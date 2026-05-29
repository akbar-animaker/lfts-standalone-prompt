#!/usr/bin/env python3
"""Standalone runner for the video clipping agent pipeline.

Runs Stages 1–5 plus ranking, end-to-end:
    Preprocess → Speaker/Section Detection (Stage 1b) →
    Segment (boundary-aware) → Sequential (Agent #1 + #3) →
    Non-Sequential (Agent #2 + #4) → Final Metadata (Agent #5) → Ranking → write result JSON

Key fixes applied:
  - Stage 1b: Speaker/section boundary detection via LLM — identifies where embedded video
    clips, demos, and speaker changes occur in the merged AWS Transcribe transcript.
  - segment() is now boundary-aware — windows NEVER cross a speaker/section boundary.
  - Window max duration raised to SOCIAL_MEDIA_MAX_DURATION (180s = YouTube Shorts max) — no artificial 55s cap.
    Agent #1 picks clips; the only hard constraint is the social media max length.
  - Each window is tagged with its speaker/section label so Agent #1 knows exactly
    what it is picking (MAIN_PRESENTER, EMBEDDED_CLIP, DEMO, etc.).

Skipped on purpose (for feasibility): execution tracing, ES feedback, S3 uploads, thumbnails.
Frame extraction runs only if VIDEO_PATH is set (needs ffmpeg on PATH).

Edit the INPUT PATHS block below, then run:
    python standalone.py

The transcript JSON may be either the raw AWS Transcribe payload (with a top-level
"results" key) or the inner "results" dict directly.
"""

import base64
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path
import io

from anthropic import Anthropic

# Optional imports for thumbnail generation
try:
    from openai import OpenAI
    from PIL import Image
    import boto3
except ImportError:
    OpenAI = None
    Image = None
    boto3 = None


# ── Logging ───────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_LOG_FILE = os.path.join(_SCRIPT_DIR, "pipeline.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(_LOG_FILE, mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger("clipping_agent_standalone")


# ── Input paths (EDIT THESE) ──────────────────────────────────────────────────
CLIP_MODE = "both"                             # "both" | "sequential" | "non_sequential"
TRANSCRIPT_PATH  = os.path.join(_SCRIPT_DIR, "transcript.json")
VIDEO_PATH       = os.path.join(_SCRIPT_DIR, "input.mp4")
OUTPUT_PATH      = os.path.join(_SCRIPT_DIR, "result.json")
SPEAKERS_OUTPUT_PATH = os.path.join(_SCRIPT_DIR, "speakers.json")  # Stage 1b speaker reference
ENABLE_THUMBNAILS = False                       # Set to True to enable thumbnail generation


# ── Config ────────────────────────────────────────────────────────────────────
CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY", "")
CLAUDE_MODEL = "claude-haiku-4-5"
CLAUDE_MAX_TOKENS = 8192

# Thumbnail config (only used if ENABLE_THUMBNAILS = True)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
MAX_THUMBNAILS_PER_TYPE = 2
S3_BUCKET = "anim-user-uploads"
THUMBNAIL_RESPONSE_MODEL = "gpt-4o"

# Window duration — no artificial upper cap on picking.
# Agent #1 freely selects any window whose content is complete and standalone.
# Platform max durations:
#   TikTok / Instagram Reels : 90s
#   YouTube Shorts           : 180s  ← hard ceiling used here
SEQUENTIAL_MIN_DURATION = 25      # minimum to avoid trivially short clips
SOCIAL_MEDIA_MAX_DURATION = 180   # hard ceiling: YouTube Shorts max (longest platform)

NON_SEQUENTIAL_MIN_DURATION = 25
NON_SEQUENTIAL_MAX_DURATION = 180

# Section boundary detection — gaps longer than this between sentences are
# treated as speaker/content-change boundaries. Windows never cross these.
SECTION_BREAK_GAP_THRESHOLD = 3.0   # seconds

TARGET_SCORE = 90
MAX_ITERATIONS_PER_SHORT_SEQUENCE = 1
MAX_ITERATIONS_PER_SHORT_NONSEQUENCE = 1
SCORE_MARKET_ADJUSTMENT = 0

# ── Clip refinement (Agent #3) ───────────────────────────────────────────────
# When a clip scores below TARGET_SCORE, the reviewer REFINES the same clip
# (adjusts its start/end boundaries) instead of jumping to an unrelated window.
# REFINE_PAD = how many seconds of neighbouring context (prev + next) to expose
# as boundary-variant options on each side of the current clip.
REFINE_PAD = 35.0
REFINE_MAX_OPTIONS = 150

# Non-sequential refinement: when refining a short, the LLM may only add/keep
# sentences within this many seconds of the short's existing span. This stops
# refinement from grafting unrelated sentences from a far-away part of the video
# (e.g. pulling an intro line into a mid-video monologue).
REFINE_NONSEQ_PAD = 60.0

FRAME_INTERVAL = 30   # seconds between frames
MAX_FRAMES = 20

# Standard scoring dimensions — fixed weights for all videos (sum to 100).
SCORING_WEIGHTS_RAW = {
    "hook_strength": 23,
    "reframe_insight": 18,
    "emotional_resonance": 17,
    "standalone_clarity": 16,
    "quotability": 11,
    "clean_ending": 8,
    "pacing_energy": 7,
}
DIMENSION_WEIGHTS = {k: v / 100.0 for k, v in SCORING_WEIGHTS_RAW.items()}


def get_scoring_config():
    """Return static scoring weights used by selection agents and the scorer."""
    return {
        "weights": DIMENSION_WEIGHTS.copy(),
        "weights_raw": SCORING_WEIGHTS_RAW.copy(),
    }


# ── Prompts (copied verbatim from agent/prompts/) ─────────────────────────────

SEQUENCE_PROMPT = """

You are a pro-level video editor and content strategist for social media shorts.

## UNDERSTANDING THE WINDOW LIST
Each window below is tagged with the speaker/section it belongs to, in the form:
   [<speaker label>]  — e.g. the speaker's name and role detected for that part of the video.

Different windows may come from different speakers or distinct segments of the video
(for example a main presenter, an embedded/quoted video, a role-play or product demo,
an avatar/synthesized character, an audience Q&A, etc.). The exact set of speakers and
sections varies from video to video — rely on the tag shown on each window rather than a
fixed list.

Windows already respect speaker/section boundaries — no window spans two different speakers.
You may select ANY window regardless of who is speaking or which section it is from, as long
as it stands alone as a great clip. A demo, role-play or embedded-video window can be an
excellent clip if it is self-contained.

Each window includes measured pacing stats (WPM, filler %, dead-air gaps, burst delivery,
pacing out of 7). Use these when estimating **Pacing / Energy** (max {w_pacing}).

## YOUR SCORING FRAMEWORK (standard weights — all videos)

The 7 dimensions below use fixed point caps that sum to 100 (23+18+17+16+11+8+7).
Score each dimension out of its own max, then add for Total /100.
A moment must score 75 or above out of 100 to become a clip. Anything below 75 gets left behind.
Never lower the bar to hit a clip count target. Fewer great clips are always better than more mediocre ones.

{scoring_context}

Once a clip passes 75, note its estimated score in this format:
Hook: /{w_hook} | Reframe: /{w_reframe} | Emotion: /{w_emotion} | Clarity: /{w_clarity} | Quotability: /{w_quotable} | Ending: /{w_ending} | Pacing: /{w_pacing} | Total: /100

Scale the number of clips to the video's length. A 10-minute video should yield around 6 clips. Longer videos should produce more, proportionally. But never compromise the 75-point minimum to hit a number — quality always wins.
Finally, ask: would someone scrolling their feed stop, watch, and feel completely satisfied by this clip alone? If anything still feels incomplete, thin, or confusing — select a different window.

The maximum acceptable duration is {social_media_max}s (YouTube Shorts max). TikTok/Instagram clips should stay under 90s — note this in your reasoning if relevant. Do not select windows longer than {social_media_max}s.

Transcript windows: {windows_text}
"""

SEQUENCE_RESPONSE_FORMAT = {
    "clips": [{"WindowId": "number", "reason": "string"}]
}

# ── Agent #3: in-place clip refinement (boundary fix, NOT a new topic) ────────
REFINE_PROMPT = """
You are refining ONE existing short clip to fix a specific weakness.

## ABSOLUTE RULE
Do NOT switch to a different topic. Keep the SAME core message as the current clip.
You may ONLY adjust WHERE the clip starts and ends by choosing a different
window from the options below — all of which cover the same region of the video.

## CURRENT CLIP (scored {score}/100 — target is {target}+)
Time: {cur_start:.1f}s – {cur_end:.1f}s ({cur_duration:.1f}s)
Transcript: "{cur_text}"
{pacing_section}
## WEAKNESS TO FIX
- Weakest factor: **{weakest_factor}**
- Problem: {reason}
- How to improve: {improvise}

## HOW TO REFINE
- If the OPENING is mid-thought / weak hook → pick a window that STARTS at a cleaner,
  self-explanatory sentence (a real hook that makes sense with no prior context).
- If the ENDING is abrupt / unfinished → pick a window that ENDS on a complete,
  resolved thought or a line that lands with weight.
- If clarity is weak → pick a window that includes just enough context to stand alone.
- Never start or end on a filler sound ("uh", "um", "ah", "hmm") or a half sentence.
- Keep the duration under {social_media_max}s.

## SCORING FRAMEWORK (standard weights)
{scoring_context}

## WINDOW OPTIONS (same region — pick the ONE best refined boundary)
{options}

Return EXACTLY ONE clip — the refined version (one WindowId + a short reason
explaining how the new boundaries fix the weakest factor).
"""

NONSEQUENCE_PROMPT = """

You are a pro-level video editor and content strategist for social media shorts.

## YOUR SCORING FRAMEWORK (standard weights — all videos)

Unlike straight cuts, you can select multiple separate moments from across the video and join them into one cohesive clip. The story always comes first. The moments serve it.

Each sentence in the list may include delivery flags (e.g. fill %, dead_air, wpm) when
speaking is slow, filler-heavy, or pause-filled — avoid those sentences when possible,
and avoid joins that would create long dead-air gaps between segments.

The 7 dimensions below use fixed point caps that sum to 100 (23+18+17+16+11+8+7).
Score each dimension out of its own max, then add for Total /100.
A combination of moments that scores 75 or above out of 100 becomes a clip. Anything below 75 gets left behind.
Never lower the bar to hit a clip count target. Fewer great clips are always better than more mediocre ones.

{scoring_context}

Once a clip passes 75, note its estimated score in this format:
Hook: /{w_hook} | Reframe: /{w_reframe} | Emotion: /{w_emotion} | Clarity: /{w_clarity} | Quotability: /{w_quotable} | Ending: /{w_ending} | Pacing: /{w_pacing} | Total: /100

Scale the number of clips to the video's length. A 10-minute video should yield around 6 clips. Longer videos should produce more, proportionally. But never compromise the 75-point minimum — quality always wins.

## ENDING RULE (non-negotiable)
The LAST sentence you select MUST be a landing line — a conclusion, payoff, punchline, takeaway, or clear resolution of the story. NEVER end on a setup, a sentence that introduces something it doesn't resolve, or a line that trails off mid-thought (e.g. "...you give them good data.", "...part of those..."). If the natural payoff is the very next sentence after your last pick, extend the selection to include it. A clip that sets up an idea but cuts before landing it is incomplete and must be re-selected so it ends on the resolving line.

Finally, ask: would someone scrolling their feed stop, watch, and feel completely satisfied by this clip alone — with no knowledge of the original video? If anything feels missing, disjointed, or incomplete — re-select segments, adjust the joins, or rethink the story.

Keep each clip between 30 to 90 seconds. Duration is not a creative decision — it is a byproduct of the story. Never cut a story short to fit the range, and never pad a clip to reach it.

{sentences_text}
"""

NONSEQUENCE_RESPONSE_FORMAT = {
    "shorts": [{
        "topic": "string",
        "sentence_ids": ["number"],
        "reason": "string"
    }]
}

# ── Agent #4: in-place non-sequential refinement (adjust selection, NOT new topic) ──
REFINE_NONSEQ_PROMPT = """
You are refining ONE existing multi-segment short to fix a specific weakness.

## ABSOLUTE RULE
Keep the SAME core story / theme as the current short. Do NOT switch to a different
topic. You may ADD, REMOVE, or RE-ORDER sentences — but the refined short must tell
the SAME story, only better.

## ALLOWED SENTENCE RANGE (hard constraint)
To keep the short on-topic, you may ONLY select sentences whose ID is between
**S{allowed_lo} and S{allowed_hi}** (inclusive). These surround the current short's
own content. Do NOT pull sentences from outside this range — a far-away line will be
rejected. Select sentence IDs only from the AVAILABLE SENTENCES list above and only
within this range.

## CURRENT SHORT (scored {score}/100 — target is {target}+)
Duration: {cur_duration:.1f}s across {cur_num} segments
Sentences currently used: {cur_ids}
Transcript: "{cur_text}"
{pacing_section}
## WEAKNESS TO FIX
- Weakest factor: **{weakest_factor}**
- Problem: {reason}
- How to improve: {improvise}

## HOW TO REFINE
- Weak hook → add or swap the OPENING sentence for one that grabs attention and makes
  sense with no prior context.
- Abrupt ending → add a sentence that resolves the thought, or drop a trailing half-sentence.
- Weak clarity → add a sentence that supplies the missing context; drop confusing fragments.
- Drop filler-only sentences ("uh", "um") when they hurt the flow.
- Keep total duration roughly 30–90s. Keep sentences in chronological order unless a
  deliberate join genuinely strengthens the story.

## SCORING FRAMEWORK (standard weights)
{scoring_context}

Return EXACTLY ONE short — the refined version — with its updated sentence_ids and a
short reason explaining how the new selection fixes the weakest factor.
"""

SCORER_PROMPT = """
You are a social media viewer and honest evaluator. A short clip has been generated from a longer video.
Score it across 6 content dimensions (each out of its own max — they sum to 93).
Pacing / Energy (max {w_pacing}) is computed automatically from measured delivery stats — do NOT score it.

## CLIP TO SCORE
Type: {clip_type}
Duration: {duration}s
Text: "{text_preview}"
{pacing_section}
## STANDARD SCORING (sum to /100)
Score each dimension **out of its max points** below. Add them for the content subtotal;
Pacing / Energy is added separately from measurements.

| Dimension           | Score out of |
|---------------------|--------------|
| Hook Strength       | {w_hook} |
| Reframe / Insight   | {w_reframe} |
| Emotional Resonance | {w_emotion} |
| Standalone Clarity  | {w_clarity} |
| Quotability         | {w_quotable} |
| Clean Ending        | {w_ending} |
| Pacing / Energy     | {w_pacing} (auto — see stats above) |

## SCORING DIMENSIONS

**Hook Strength** (score 0–{w_hook}): Opening grabs attention and makes sense with no prior context? Penalize mid-thought opens and fillers.

**Reframe/Insight** (score 0–{w_reframe}): Perspective or idea that makes viewers think differently? Penalize generic or setup-only content.

**Emotional Resonance** (score 0–{w_emotion}): Makes viewers feel something standalone — curiosity, inspiration, humor, surprise?

**Standalone Clarity** (score 0–{w_clarity}): Fully understandable without the full video? Penalize vague "he/they/that" references.

**Quotability** (score 0–{w_quotable}): Memorable, repeatable, shareable lines?

**Clean Ending** (score 0–{w_ending}): Satisfying natural end — not mid-sentence or on fillers.

## REQUIRED OUTPUT

Return ONLY valid JSON with these exact fields:

{{
  "hook_strength": [integer 0-{w_hook}],
  "reframe_insight": [integer 0-{w_reframe}],
  "emotional_resonance": [integer 0-{w_emotion}],
  "standalone_clarity": [integer 0-{w_clarity}],
  "quotability": [integer 0-{w_quotable}],
  "clean_ending": [integer 0-{w_ending}],
  "score_reasoning": {{
    "hook_strength": "1 sentence: why this score",
    "reframe_insight": "1 sentence: why this score",
    "emotional_resonance": "1 sentence: why this score",
    "standalone_clarity": "1 sentence: why this score",
    "quotability": "1 sentence: why this score",
    "clean_ending": "1 sentence: why this score",
    "pacing_energy": "1 sentence on delivery using the measured pacing stats (for reference only)"
  }},
  "reason": "[2-3 sentences explaining the overall score]",
  "weakest_factor": "[dimension key with lowest score relative to its max]",
  "improvise": "[specific actionable advice to improve this clip]"
}}
"""

SCORER_RESPONSE_FORMAT = {
    "type": "json_object"
}

METADATA_PROMPT = """
You are a social media strategist. Generate titles and platform recommendations for {short_count} video shorts.

### LANGUAGE REQUIREMENT

The video transcript language is: **{language_code}**

**CRITICAL: Generate ALL titles in the NATIVE LANGUAGE of the transcript.**
- If language is "en-US" or "en-*" → English titles
- If language is "hi-IN" → Hindi titles (Devanagari script)
- If language is "es-*" → Spanish titles
- If language is "fr-*" → French titles
- If language is "ta-IN" → Tamil titles
- If language is "te-IN" → Telugu titles
- And so on for any other language

The title MUST be in the same language the speaker is using in the transcript.

### TITLE GUIDELINES

- **6-8 words** (short and punchy)
- Capitalize first letter of each word (where applicable for the language)
- Focus on value proposition: What will viewer learn/feel?
- Must be in the native language: {language_code}

### PLATFORM SELECTION

Choose 2-3 platforms per short based on content type and duration:

- **YouTube Shorts**: Up to 180s. Business, educational, frameworks, longer explanations.
- **Instagram Reels**: Up to 90s. Relatable, emotional, visual stories, lifestyle.
- **TikTok**: Up to 90s. Humor, personality, trends, quick hooks.
- **LinkedIn**: Up to 180s. Professional, business lessons, startup insights.

**IMPORTANT duration rule**: If a clip is longer than 90s, do NOT include Instagram or TikTok.
Only YouTube Shorts and LinkedIn support up to 180s.

Here are the {short_count} shorts:

{shorts_context}
"""

METADATA_RESPONSE_FORMAT = {
    "shorts": [{
        "short_id": "string",
        "title": "string",
        "platforms": ["string"],
    }]
}

# ── Thumbnail prompts ─────────────────────────────────────────────────────────

THUMBNAIL_ART_DIRECTOR_SYSTEM = """You are a top social media thumbnail art director for YouTube Shorts, Reels, and TikTok.

Given a subject photo, video title, and optionally the short's transcript/script, invent a UNIQUE, click-worthy thumbnail concept for THIS video only.

When a transcript is provided, use it only as creative reference: extract mood, key topics, emotional hook, and visual metaphors. Do NOT paste long transcript text into the thumbnail.

STRICT RULES:
- Do NOT copy any fixed template (no default navy-blue political layout, no repeated red "secrets" banner on every design).
- Background, colors, icons, and mood MUST match the title, transcript themes (if any), and the energy of the photo.
- Vary layout each time: different palette, gradient direction, graphic style (3D, flat, cinematic, neon, minimal, bold collage, etc.).
- Think viral engagement: curiosity gap, contrast, one clear focal subject, readable text on mobile.
- Headline = split the user's title into powerful ALL-CAPS lines; pick 1–2 words for a bright accent color that fits the topic (not always yellow).
- Optional: short hook tag, 1–3 micro bullet chips/icons ONLY if they strengthen the concept — skip if cleaner without.
- Keep the same person from the photo (face, likeness, outfit).

Return ONLY valid JSON:
{
  "creative_direction": "Detailed paragraph for an image model: exact colors, background scene/graphics, text placement, accent words, mood, lighting, effects on subject (rim glow color matching palette). Be specific and unique.",
  "headline_lines": ["LINE1", "LINE2", "LINE3"],
  "accent_words": ["WORD"],
  "hook_tag": "short phrase or empty string",
  "palette": ["#hex1", "#hex2", "#hex3"]
}"""

THUMBNAIL_IMAGE_INSTRUCTION = """Create a finished vertical 9:16 social media thumbnail (YouTube Shorts / Reels / TikTok).

VIDEO TITLE: "{title}"
{transcript_block}
CREATIVE DIRECTION (follow this exactly — it is unique to this video):
{creative_direction}

SUBJECT (from attached photo):
- Keep the same person: face, likeness, outfit, expression
- Prominent placement (usually lower half or rule-of-thirds), with rim light/glow matching the palette above

TEXT:
- Render these headline lines large, bold, ALL-CAPS sans-serif: {headline_lines}
- Emphasize these words in the accent color from the direction: {accent_words}
{hook_line}

REQUIREMENTS:
- Full graphic design thumbnail with topic-specific background — NOT the original plain photo
- High contrast, mobile-readable, scroll-stopping, creative and fresh
- Do NOT use a generic reused background; illustrate the topic visually
"""

SPEAKER_DETECTION_PROMPT = """
You are an expert transcript analyst. The transcript below comes from AWS Transcribe
and has NO speaker labels — all speakers are merged into a single text stream.

Your job is to read every sentence and assign it to the correct speaker.

## STEP 1 — IDENTIFY ALL SPEAKERS
First scan the full transcript and identify every distinct voice:
- Real people (presenter, interviewer, guest)
- Quoted characters in embedded video clips
- Role-play / demo characters (e.g. customer AND advisor are TWO different speakers)
- Avatar or synthesized voice characters
- Any other distinct voice

Clues for speaker changes:
- Transition phrases: "let's watch this", "here's a quick video", "take a look", "here we go"
- [GAP Xs] markers — large silences almost always mean a new voice
- Self-introductions: "I am a trainer just like you", "Hi, my name is..."
- Back-and-forth dialogue patterns (Q&A, customer vs advisor, interviewer vs guest)
- Shift in tone, perspective, or speaking style
- Quoted speech clearly from a different person

## STEP 2 — ASSIGN EVERY SENTENCE TO A SPEAKER
Go through each sentence one by one and assign it to one of the speakers you identified.
For fast dialogues (role-play, Q&A), assign EACH individual sentence to its speaker —
do NOT lump an entire dialogue block under one person.

## TRANSCRIPT (with sentence indices, timestamps, and gap markers):
{transcript_with_gaps}

## REQUIRED OUTPUT
Return ONLY valid JSON:
{{
  "total_speakers": number,
  "note": "1 sentence describing the overall speaker structure of this video",
  "speakers": [
    {{
      "id": "Person_1",
      "name": "descriptive name or real name if mentioned",
      "role": "their role in this video",
      "context": "1-2 sentences: how you identified this as a distinct speaker",
      "sentence_ids": [1, 2, 3, 7, 12, 45, ...]
    }}
  ]
}}

Rules:
- sentence_ids is a flat list of ALL sentence indices this speaker says — in any order, non-contiguous is fine.
- Every sentence index in the transcript must appear in exactly ONE speaker's sentence_ids list.
- For fast back-and-forth dialogue, assign each sentence individually (e.g. S101 → Mr. Gecko, S102 → Bank Advisor, S103 → Mr. Gecko, ...).
- IDs must be sequential: Person_1, Person_2, Person_3, etc.
- List the main presenter as Person_1.
- Lean toward more speakers rather than fewer — better to split than to merge two distinct voices.
"""

SPEAKER_DETECTION_RESPONSE_FORMAT = {
    "total_speakers": "number",
    "note": "string",
    "speakers": [{
        "id": "string",
        "name": "string",
        "role": "string",
        "context": "string",
        "sentence_ids": ["number"],
    }]
}


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Sentence:
    index: int
    start: float
    end: float
    text: str


@dataclass
class CandidateWindow:
    id: int
    start_s: int        # sentence index of first sentence
    end_s: int          # sentence index of last sentence
    start_time: float
    end_time: float
    duration: float
    text: str
    section_type: str = "MAIN_PRESENTER"   # from Stage 1b speaker detection
    section_label: str = ""                # human-readable section label


# ── LLM client ────────────────────────────────────────────────────────────────

_client = None
_token_usage = {
    "total_input_tokens": 0,
    "total_output_tokens": 0,
    "total_cache_read_tokens": 0,
    "total_cache_write_tokens": 0,
    "calls": [],
}


def _get_client():
    global _client
    if _client is None:
        if not CLAUDE_API_KEY:
            raise RuntimeError(
                "CLAUDE_API_KEY environment variable is not set. "
                "Export it before running, e.g. `export CLAUDE_API_KEY=sk-ant-...`."
            )
        _client = Anthropic(api_key=CLAUDE_API_KEY)
    return _client


def _track_usage(caller, input_tokens, output_tokens,
                 cache_read_tokens=0, cache_write_tokens=0):
    _token_usage["total_input_tokens"] += input_tokens
    _token_usage["total_output_tokens"] += output_tokens
    _token_usage["total_cache_read_tokens"] += cache_read_tokens
    _token_usage["total_cache_write_tokens"] += cache_write_tokens
    # Cached read tokens are billed at ~10% of normal; write tokens at ~125%
    # Estimate effective cost in "equivalent full tokens" for visibility
    effective = input_tokens + output_tokens + int(cache_read_tokens * 0.1) + int(cache_write_tokens * 1.25)
    saved = int(cache_read_tokens * 0.9)
    _token_usage["calls"].append({
        "caller": caller,
        "model": CLAUDE_MODEL,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_tokens": cache_read_tokens,
        "cache_write_tokens": cache_write_tokens,
        "tokens_saved": saved,
    })
    cache_info = ""
    if cache_read_tokens or cache_write_tokens:
        cache_info = f" | cache_read: {cache_read_tokens} cache_write: {cache_write_tokens} saved≈{saved}"
    log.info(
        f"[LLM] {caller} — in: {input_tokens}, out: {output_tokens}{cache_info}, "
        f"running_total: {_token_usage['total_input_tokens'] + _token_usage['total_output_tokens']}"
    )


def get_token_usage():
    total_in  = _token_usage["total_input_tokens"]
    total_out = _token_usage["total_output_tokens"]
    cache_read  = _token_usage["total_cache_read_tokens"]
    cache_write = _token_usage["total_cache_write_tokens"]
    tokens_saved = sum(c.get("tokens_saved", 0) for c in _token_usage["calls"])
    return {
        "total_input_tokens":       total_in,
        "total_output_tokens":      total_out,
        "total_tokens":             total_in + total_out,
        "total_cache_read_tokens":  cache_read,
        "total_cache_write_tokens": cache_write,
        "total_tokens_saved":       tokens_saved,
        "llm_calls":                len(_token_usage["calls"]),
        "calls":                    _token_usage["calls"],
    }


def _build_prompt(prompt, response_format=None):
    if not response_format:
        return prompt
    return (
        f"{prompt}\n\n"
        "Return only valid JSON. Do not include markdown, code fences, or any text outside the JSON.\n"
        "The JSON response must follow this structure:\n"
        f"{json.dumps(response_format, ensure_ascii=False, indent=2)}"
    )


def _parse_json_response(text):
    try:
        return json.loads(text)
    except ValueError:
        pass

    for fence in ("```json", "```"):
        if fence in text:
            parts = text.split(fence)
            for i in range(1, len(parts)):
                candidate = parts[i].split("```")[0].strip()
                try:
                    return json.loads(candidate)
                except ValueError:
                    continue

    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last > first:
        try:
            return json.loads(text[first:last + 1])
        except ValueError:
            pass

    if first != -1:
        repaired = _repair_truncated_json(text[first:])
        if repaired is not None:
            return repaired

    log.error(f"JSON parse failed. Text length: {len(text)}, Start: {text[:300]}")
    raise RuntimeError(f"LLM returned invalid JSON (length={len(text)})")


def _repair_truncated_json(text):
    obj_ends = [m.end() for m in re.finditer(r'\}\s*,?\s*', text)]
    if not obj_ends:
        return None
    for end_pos in reversed(obj_ends):
        snippet = text[:end_pos].rstrip().rstrip(",")
        open_braces = snippet.count("{") - snippet.count("}")
        open_brackets = snippet.count("[") - snippet.count("]")
        if open_braces < 0 or open_brackets < 0:
            continue
        closing = "]" * open_brackets + "}" * open_braces
        try:
            result = json.loads(snippet + closing)
            if isinstance(result, dict):
                log.warning(f"Repaired truncated JSON at position {end_pos}/{len(text)}")
                return result
        except ValueError:
            continue
    return None


def get_response(prompt, system_prompt=None, temperature=0.3, response_format=None,
                 max_tokens=None, caller="unknown", cached_context=None):
    """
    Send a prompt to Claude, optionally with a large cacheable context block.

    cached_context: a plain string (e.g. full windows list or sentences list)
        that is sent as a SEPARATE user content block marked with cache_control.
        On the first call Anthropic writes it to cache (~125% token cost).
        On every subsequent call within 5 min it is served from cache (~10% cost).
        The dynamic part of the prompt (feedback, clip count, etc.) is sent as
        a normal second content block so it never pollutes the cache key.
    """
    client = _get_client()
    full_prompt = _build_prompt(prompt, response_format)

    if cached_context:
        # Two-block message: [cached large context] + [dynamic prompt]
        content = [
            {
                "type": "text",
                "text": cached_context,
                "cache_control": {"type": "ephemeral"},  # Anthropic prompt cache
            },
            {
                "type": "text",
                "text": full_prompt,
            },
        ]
        messages = [{"role": "user", "content": content}]
    else:
        messages = [{"role": "user", "content": full_prompt}]

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens or CLAUDE_MAX_TOKENS,
        system=system_prompt or "You are a helpful assistant.",
        temperature=temperature,
        messages=messages,
        extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
    )

    usage = response.usage
    _track_usage(
        caller,
        usage.input_tokens,
        usage.output_tokens,
        cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0),
        cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0),
    )
    text = response.content[0].text.strip()
    if not text:
        raise RuntimeError("Claude returned empty response")
    return _parse_json_response(text)


def get_multimodal_response(prompt, frames_b64, temperature=0.3, response_format=None,
                            max_tokens=None, caller="unknown"):
    if not frames_b64:
        return get_response(prompt, temperature=temperature,
                            response_format=response_format,
                            max_tokens=max_tokens, caller=caller)
    client = _get_client()
    full_prompt = _build_prompt(prompt, response_format)
    content = []
    for frame_b64 in frames_b64:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": frame_b64,
            },
        })
    content.append({"type": "text", "text": full_prompt})
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens or CLAUDE_MAX_TOKENS,
        temperature=temperature,
        messages=[{"role": "user", "content": content}],
        extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
    )
    usage = response.usage
    _track_usage(
        caller,
        usage.input_tokens,
        usage.output_tokens,
        cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0),
        cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0),
    )
    text = response.content[0].text.strip()
    if not text:
        raise RuntimeError("Claude multimodal returned empty response")
    return _parse_json_response(text)


# ── Frame extraction (local file via ffmpeg) ──────────────────────────────────

def extract_frames_from_file(video_path):
    if not video_path or not os.path.isfile(video_path):
        log.warning(f"[FRAMES] Video not found: {video_path}")
        return []

    temp_dir = tempfile.mkdtemp()
    try:
        pattern = os.path.join(temp_dir, "frame_%04d.jpg")
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", video_path,
            "-vf", f"fps=1/{FRAME_INTERVAL}",
            "-q:v", "2",
            "-frames:v", str(MAX_FRAMES),
            pattern,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=180)
        except FileNotFoundError:
            log.error("[FRAMES] ffmpeg not found on PATH — skipping frame extraction")
            return []
        except subprocess.TimeoutExpired:
            log.error("[FRAMES] ffmpeg timed out")
            return []
        except subprocess.CalledProcessError as e:
            log.error(f"[FRAMES] ffmpeg failed: {e.stderr.decode()[:200]}")
            return []

        files = sorted(
            os.path.join(temp_dir, f) for f in os.listdir(temp_dir)
            if f.startswith("frame_") and f.endswith(".jpg")
        )[:MAX_FRAMES]

        encoded = []
        for p in files:
            try:
                with open(p, "rb") as f:
                    encoded.append(base64.b64encode(f.read()).decode("utf-8"))
            except Exception as e:
                log.error(f"[FRAMES] encode failed for {p}: {repr(e)}")
        log.info(f"[FRAMES] Extracted {len(encoded)} frames")
        return encoded
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ── Stage 1: Preprocess ───────────────────────────────────────────────────────

def preprocess(transcription_data):
    items = transcription_data.get("items", [])
    log.info(f"[PREPROCESS] Items: {len(items)}")

    sentences = []
    all_words = []          # flat (start, end, text) for every spoken word → used by pacing
    current_words = []
    current_start = None
    current_end = None

    for item in items:
        t = item.get("type")
        if t == "pronunciation":
            word = item["alternatives"][0]["content"]
            start = float(item.get("start_time", 0))
            end = float(item.get("end_time", 0))
            all_words.append((start, end, word))
            if current_start is None:
                current_start = start
            current_end = end
            current_words.append(word)
        elif t == "punctuation":
            punct = item["alternatives"][0]["content"]
            if current_words:
                current_words[-1] += punct
            if punct in (".", "?", "!"):
                if current_words and current_start is not None:
                    sentences.append(Sentence(
                        index=len(sentences) + 1,
                        start=round(current_start, 2),
                        end=round(current_end, 2),
                        text=" ".join(current_words),
                    ))
                current_words, current_start, current_end = [], None, None

    if current_words and current_start is not None:
        sentences.append(Sentence(
            index=len(sentences) + 1,
            start=round(current_start, 2),
            end=round(current_end, 2),
            text=" ".join(current_words),
        ))

    video_duration = 0.0
    for item in items:
        et = item.get("end_time")
        if et:
            video_duration = max(video_duration, float(et))

    language_code = transcription_data.get("language_code", "en-US")
    log.info(
        f"[PREPROCESS] {len(sentences)} sentences, {len(all_words)} words, "
        f"{video_duration:.1f}s, lang={language_code}"
    )
    return sentences, video_duration, language_code, all_words


# ── Pacing analysis (deterministic, from word-level timestamps) ──────────────
#
# Computes a 0–20 "pacing score" for a clip purely from its word timings — no LLM.
# Four signals (WPM, filler ratio, dead-air gaps, burst ratio) each adjust a base
# score of 20. This is metadata that runs alongside the LLM 6-dimension score; it
# does NOT gate clip selection.

# Single-token fillers. The bigram "you know" is handled separately below.
PACING_FILLER_WORDS = {
    "um", "uh", "like", "basically", "literally", "actually", "right", "so",
}
_PACING_PUNCT = ".,!?;:\"'()-—…"

# Thresholds (per manager's pacing framework).
PACING_WPM_IDEAL      = (130, 180)   # 0 adjustment
PACING_WPM_SLOW       = (110, 130)   # -3
PACING_WPM_FAST       = (180, 220)   # -1
                                     # outside [110, 220] → -5
PACING_DEAD_AIR_GAP   = 1.5          # seconds; gap larger than this = dead air
PACING_BURST_GAP      = 0.3          # seconds; gap smaller than this = burst pair
PACING_FILLER_OK      = 0.03
PACING_FILLER_MAX     = 0.07
PACING_BURST_HIGH     = 0.70
PACING_BURST_MID      = 0.50


def _pacing_norm(token):
    """Lowercase + strip surrounding punctuation for filler matching."""
    return token.strip().strip(_PACING_PUNCT).lower()


def _count_fillers(words):
    """Count filler words, treating 'you know' as a single filler."""
    norms = [_pacing_norm(w[2]) for w in words]
    count = 0
    i = 0
    n = len(norms)
    while i < n:
        if i + 1 < n and norms[i] == "you" and norms[i + 1] == "know":
            count += 1
            i += 2
            continue
        if norms[i] in PACING_FILLER_WORDS:
            count += 1
        i += 1
    return count


def compute_pacing(segments):
    """
    Compute pacing metrics + score for a clip.

    segments: list of segments, each a list of (start, end, text) word tuples.
              Sequential clips have one segment; non-sequential shorts have one
              per contiguous piece. Dead-air and burst gaps are measured WITHIN a
              segment only — the cut between two segments is intentional, not a pause.

    Returns a dict with the four metrics and a 0–20 pacing_score.
    """
    flat = [w for seg in segments for w in seg]
    total_words = len(flat)
    if total_words == 0:
        return {
            "pacing_score": 0,
            "wpm": 0.0,
            "filler_ratio": 0.0,
            "dead_air_gaps": 0,
            "burst_ratio": 0.0,
            "total_words": 0,
            "speaking_duration": 0.0,
            "adjustments": {"wpm": 0, "filler": 0, "dead_air": 0, "burst": 0},
        }

    # Speaking duration = sum of each segment's own span (ignores inter-segment cuts).
    speaking_duration = sum((seg[-1][1] - seg[0][0]) for seg in segments if seg)
    minutes = speaking_duration / 60.0
    wpm = total_words / minutes if minutes > 0 else 0.0

    filler_count = _count_fillers(flat)
    filler_ratio = filler_count / total_words

    dead_air_gaps = 0
    burst_words = 0
    for seg in segments:
        for a, b in zip(seg, seg[1:]):
            gap = b[0] - a[1]
            if gap > PACING_DEAD_AIR_GAP:
                dead_air_gaps += 1
            if gap < PACING_BURST_GAP:
                burst_words += 1
    burst_ratio = burst_words / total_words

    # ── Score: base 20, apply each adjustment, clamp to [0, 20] ───────────────
    adj = {"wpm": 0, "filler": 0, "dead_air": 0, "burst": 0}

    if PACING_WPM_IDEAL[0] <= wpm <= PACING_WPM_IDEAL[1]:
        adj["wpm"] = 0
    elif PACING_WPM_SLOW[0] <= wpm < PACING_WPM_SLOW[1]:
        adj["wpm"] = -3
    elif PACING_WPM_FAST[0] < wpm <= PACING_WPM_FAST[1]:
        adj["wpm"] = -1
    else:
        adj["wpm"] = -5

    if filler_ratio < PACING_FILLER_OK:
        adj["filler"] = 0
    elif filler_ratio <= PACING_FILLER_MAX:
        adj["filler"] = -2
    else:
        adj["filler"] = -5

    if dead_air_gaps == 0:
        adj["dead_air"] = 0
    elif dead_air_gaps <= 2:
        adj["dead_air"] = -2
    else:
        adj["dead_air"] = -5

    if burst_ratio > PACING_BURST_HIGH:
        adj["burst"] = 2
    elif burst_ratio >= PACING_BURST_MID:
        adj["burst"] = 0
    else:
        adj["burst"] = -2

    score = 20 + adj["wpm"] + adj["filler"] + adj["dead_air"] + adj["burst"]
    score = max(0, min(20, score))

    return {
        "pacing_score": score,
        "wpm": round(wpm, 1),
        "filler_ratio": round(filler_ratio, 4),
        "dead_air_gaps": dead_air_gaps,
        "burst_ratio": round(burst_ratio, 4),
        "total_words": total_words,
        "speaking_duration": round(speaking_duration, 2),
        "adjustments": adj,
    }


def _words_in_range(all_words, start, end, eps=0.05):
    """Return word tuples whose timing falls inside [start, end] (inclusive, with slack)."""
    return [w for w in all_words if w[0] >= start - eps and w[1] <= end + eps]


def pacing_for_clip(clip, all_words):
    """
    Build the per-segment word lists for a clip and compute its pacing.

    Works for both sequential clips (single start_time/end_time) and
    non-sequential shorts (a `clips` list of contiguous segments).
    """
    raw_segments = clip.get("clips")
    if raw_segments:
        segments = [
            _words_in_range(all_words, seg["start_time"], seg["end_time"])
            for seg in raw_segments
        ]
    else:
        segments = [_words_in_range(all_words, clip["start_time"], clip["end_time"])]
    return compute_pacing(segments)


def _format_pacing_inline(p):
    """Compact pacing stats for window/sentence list lines."""
    if not p or p.get("total_words", 0) == 0:
        return ""
    return (
        f"| pacing {p['pacing_score']}/20 "
        f"wpm {p['wpm']} fill {p['filler_ratio']:.0%} "
        f"dead {p['dead_air_gaps']} burst {p['burst_ratio']:.0%}"
    )


def _sentence_pacing_hint(s, all_words):
    """Per-sentence delivery flags for non-seq selection (flags problems only)."""
    if not all_words:
        return ""
    words = _words_in_range(all_words, s.start, s.end)
    if not words:
        return ""
    p = compute_pacing([words])
    if p.get("total_words", 0) == 0:
        return ""
    flags = []
    if p["filler_ratio"] >= PACING_FILLER_OK:
        flags.append(f"fill {p['filler_ratio']:.0%}")
    if p["dead_air_gaps"] > 0:
        flags.append(f"dead_air {p['dead_air_gaps']}")
    if p["wpm"] > 0 and (p["wpm"] < 110 or p["wpm"] > 220):
        flags.append(f"wpm {p['wpm']}")
    if not flags:
        return ""
    return f" | {' '.join(flags)}"


def _format_sentences(sentences, all_words=None):
    """
    Format the full sentence list for Agent #2 / Agent #4 cached context.
    When all_words is supplied, problematic delivery (fillers, dead air, bad WPM)
    is flagged inline so the LLM can avoid weak sentences at pick time.
    """
    lines = []
    for s in sentences:
        hint = _sentence_pacing_hint(s, all_words)
        lines.append(f"S{s.index}: [{s.start:.1f}s - {s.end:.1f}s] {s.text}{hint}")
    return "\n".join(lines)


# ── Stage 2: Segment (speaker-boundary-aware candidate windows) ──────────────

def segment(sentences, video_duration, sentence_to_speaker):
    """
    Generate candidate windows from the sentence list.

    Key rules:
    1. A window NEVER crosses a speaker boundary — prevents mixing the main
       presenter with embedded clips, demos, or role-play characters.
    2. No fixed upper duration cap — windows grow up to SOCIAL_MEDIA_MAX_DURATION.
    3. Minimum duration is SEQUENTIAL_MIN_DURATION to avoid trivially short clips.
    4. Each window is tagged with the speaker's id, name, and role so Agent #1
       knows exactly who is speaking in every window it selects.
    """
    min_dur = SEQUENTIAL_MIN_DURATION
    max_dur = SOCIAL_MEDIA_MAX_DURATION

    if video_duration < 180:
        min_dur = max(10, int(video_duration * 0.15))
        log.info(f"[SEGMENT] Short video — min window adjusted to {min_dur}s")

    windows = []
    n = len(sentences)

    for i in range(n):
        text_parts = []
        speaker_i    = sentence_to_speaker.get(sentences[i].index, {})
        speaker_id_i = speaker_i.get("id", "Person_1")

        for j in range(i, n):
            s_j = sentences[j]

            # ── Boundary rule: stop when speaker changes ─────────────────────
            speaker_j = sentence_to_speaker.get(s_j.index, {})
            if speaker_j.get("id") != speaker_id_i:
                break

            text_parts.append(s_j.text)
            start_time = sentences[i].start
            end_time   = s_j.end
            duration   = round(end_time - start_time, 1)

            if duration > max_dur:
                break  # hit the social media ceiling

            if duration >= min_dur:
                windows.append(CandidateWindow(
                    id=len(windows) + 1,
                    start_s=sentences[i].index,
                    end_s=s_j.index,
                    start_time=start_time,
                    end_time=end_time,
                    duration=duration,
                    text=" ".join(text_parts),
                    section_type=speaker_id_i,
                    section_label=f"{speaker_i.get('name', '')} — {speaker_i.get('role', '')}",
                ))

    log.info(
        f"[SEGMENT] {len(windows)} candidate windows "
        f"({min_dur}–{max_dur}s, speaker-boundary-aware)"
    )
    return windows

# ── Stage 1b: Speaker detection ──────────────────────────────────────────────

def detect_speakers(sentences):
    """
    Stage 1b: Send the full transcript to the LLM sentence-by-sentence and get
    back a speaker-identity model matching transcript_speakers.json.

    Each speaker gets a flat sentence_ids list — individual sentence indices
    they speak. This correctly handles fast dialogues and role-plays where
    turns alternate every sentence, which contiguous ranges cannot represent.

    Returns (speakers, note).
    Falls back to a single "Main Presenter" speaker on any error.
    """
    lines = []
    prev_end = 0.0
    for s in sentences:
        gap = s.start - prev_end
        if gap >= SECTION_BREAK_GAP_THRESHOLD:
            lines.append(f"[GAP {gap:.1f}s]")
        lines.append(f"S{s.index} [{s.start:.1f}s-{s.end:.1f}s]: {s.text}")
        prev_end = s.end

    transcript_with_gaps = "\n".join(lines)

    prompt = SPEAKER_DETECTION_PROMPT.format(
        transcript_with_gaps=transcript_with_gaps
    )

    try:
        result = get_response(
            prompt,
            response_format=SPEAKER_DETECTION_RESPONSE_FORMAT,
            caller="speaker_detector",
            max_tokens=8000,
        )
    except Exception as e:
        log.error(f"[SPEAKERS] Detection failed: {repr(e)} — using fallback")
        return _fallback_speakers(sentences)

    raw_speakers = result.get("speakers", [])
    if not raw_speakers:
        log.warning("[SPEAKERS] LLM returned no speakers — using fallback")
        return _fallback_speakers(sentences)

    all_sentence_indices = {s.index for s in sentences}

    # Validate: filter invalid sentence IDs, deduplicate across speakers
    # (if a sentence appears in two speakers, keep it in the first one)
    assigned = set()
    validated_speakers = []
    for sp in raw_speakers:
        raw_ids = sp.get("sentence_ids", [])
        valid_ids = []
        for sid in raw_ids:
            try:
                idx = int(sid)
            except (ValueError, TypeError):
                continue
            if idx in all_sentence_indices and idx not in assigned:
                valid_ids.append(idx)
                assigned.add(idx)
        if valid_ids:
            sp["sentence_ids"] = sorted(valid_ids)
            validated_speakers.append(sp)

    if not validated_speakers:
        log.warning("[SPEAKERS] No valid speakers after validation — using fallback")
        return _fallback_speakers(sentences)

    # Any sentences not assigned → give to Person_1 (main presenter)
    unassigned = sorted(all_sentence_indices - assigned)
    if unassigned and validated_speakers:
        validated_speakers[0]["sentence_ids"] = sorted(
            validated_speakers[0]["sentence_ids"] + unassigned
        )
        log.warning(f"[SPEAKERS] {len(unassigned)} unassigned sentences added to Person_1")

    log.info(f"[SPEAKERS] {len(validated_speakers)} speakers identified:")
    for sp in validated_speakers:
        ids = sp["sentence_ids"]
        log.info(
            f"  {sp['id']} | {sp['name']} | {sp['role']} "
            f"| {len(ids)} sentences (S{ids[0]}-S{ids[-1]})"
        )

    note = result.get("note", "")
    return validated_speakers, note


def _fallback_speakers(sentences):
    """Single Main Presenter speaker covering the whole transcript."""
    speakers = [{
        "id": "Person_1",
        "name": "Main Presenter",
        "role": "Primary speaker",
        "context": "Fallback — speaker detection was skipped or failed.",
        "sentence_ids": [s.index for s in sentences],
    }]
    return speakers, "Fallback single-speaker mode."


def _build_sentence_to_speaker_map(speakers, sentences):
    """
    Returns a dict mapping sentence.index → speaker dict.
    Used by segment() to ensure windows never cross speaker boundaries.

    Built directly from sentence_ids — no range expansion needed.
    Sentences not assigned (shouldn't happen after validation) fall back to Person_1.
    """
    mapping = {}
    for sp in speakers:
        for idx in sp.get("sentence_ids", []):
            mapping[idx] = sp

    # Safety fallback for any gaps
    fallback_sp = speakers[0] if speakers else {}
    for s in sentences:
        if s.index not in mapping:
            mapping[s.index] = fallback_sp

    return mapping


# ── Scorer ────────────────────────────────────────────────────────────────────

def _build_pacing_section(short, all_words):
    """
    Build the optional PACING / ENERGY block for the scorer prompt from the
    clip's measured word-level delivery stats. Returns "" when word data is
    unavailable so the prompt stays unchanged in that case.
    """
    if not all_words:
        return ""
    p = pacing_for_clip(short, all_words)
    if not p or p.get("total_words", 0) == 0:
        return ""
    suggested = round(p["pacing_score"] * 7 / 20)
    return f"""
## PACING / ENERGY — auto-scored out of 7 (from measurements)
This dimension is computed from word-level timestamps — you do not score it in JSON.
- Speaking rate: {p['wpm']} WPM (ideal ≈ 130–180)
- Filler ratio: {p['filler_ratio']:.1%} (good <3%, weak >7%)
- Dead-air pauses (>1.5s): {p['dead_air_gaps']}
- Burst delivery: {p['burst_ratio']:.0%} of words
- Computed pacing points: **{suggested}/7** (from internal formula {p['pacing_score']}/20)
Comment on delivery in score_reasoning.pacing_energy only.
"""


def _cap_dimension_score(raw_val, max_val):
    """Clamp LLM score to 0..max_val for that dimension."""
    return min(max(0, int(raw_val)) + SCORE_MARKET_ADJUSTMENT, max_val)


def _pacing_formula_to_points(pacing_result, max_val=7):
    """Map internal formula pacing_score (0–20) → points out of max_val (usually 7)."""
    ps = max(0, min(20, int(pacing_result.get("pacing_score", 0))))
    return min(round(ps * max_val / 20) + SCORE_MARKET_ADJUSTMENT, max_val)


def score_single_short(short, scoring_config, all_words=None):
    """
    Score a clip using the standard 7-dimension weights.

    scoring_config: dict from get_scoring_config() with weights + weights_raw.
    all_words: optional word list for measured pacing stats in the scorer prompt.
    """
    text_preview = " ".join(short["text"].split()[:100]) + "..."
    dimension_weights = scoring_config["weights"]
    weights_raw = scoring_config["weights_raw"]

    prompt = SCORER_PROMPT.format(
        clip_type=short["type"],
        duration=short["duration"],
        text_preview=text_preview,
        pacing_section=_build_pacing_section(short, all_words),
        w_hook=weights_raw["hook_strength"],
        w_reframe=weights_raw["reframe_insight"],
        w_emotion=weights_raw["emotional_resonance"],
        w_clarity=weights_raw["standalone_clarity"],
        w_quotable=weights_raw["quotability"],
        w_ending=weights_raw["clean_ending"],
        w_pacing=weights_raw["pacing_energy"],
    )

    result = get_response(prompt, response_format=SCORER_RESPONSE_FORMAT, caller="scorer")

    log.info(f"Scorer result: {result}")

    def safe_int(v):
        try:
            return int(v)
        except (ValueError, TypeError):
            return 0

    max_per_dimension = weights_raw
    adjusted_scores = {}

    # LLM scores 6 content dimensions directly out of each max (23, 18, 17, …)
    llm_dims = [d for d in dimension_weights if d != "pacing_energy"]
    for dim in llm_dims:
        max_val = max_per_dimension[dim]
        adjusted_scores[dim] = _cap_dimension_score(result.get(dim, 0), max_val)

    # Pacing / Energy: computed out of 7 from measured delivery (not LLM)
    pacing_max = max_per_dimension["pacing_energy"]
    if all_words:
        pacing_measured = pacing_for_clip(short, all_words)
        adjusted_scores["pacing_energy"] = _pacing_formula_to_points(pacing_measured, pacing_max)
    else:
        adjusted_scores["pacing_energy"] = _cap_dimension_score(result.get("pacing_energy", 0), pacing_max)

    total_score = sum(adjusted_scores.values())
    log.info(
        f"Total score: {total_score}/100 | breakdown: {adjusted_scores}"
    )

    def _dim_ratio(dim):
        m = max_per_dimension.get(dim, 1)
        return adjusted_scores[dim] / m if m else 0

    weakest_factor = min(adjusted_scores.keys(), key=_dim_ratio)

    return {
        "ConfidenceScore": total_score,
        "total_score": total_score,
        "ScoreBreakdown": adjusted_scores,
        "ScoreReasoning": result.get("score_reasoning", {}),   # per-dim reasoning from LLM
        "reason": result.get("reason", ""),
        "weakest_factor": result.get("weakest_factor", weakest_factor),
        "improvise": result.get("improvise", ""),
    }


# ── Stage 3a: Sequential selection (Agent #1 + #3) ────────────────────────────

def _build_scoring_context(scoring_config=None):
    """
    Build the standard 7-dimension scoring framework for selection/refine prompts.
    scoring_config is optional; defaults to get_scoring_config().
    """
    if scoring_config is None:
        scoring_config = get_scoring_config()
    weights_raw = scoring_config["weights_raw"]

    dim_labels = {
        "hook_strength":       "Hook Strength",
        "reframe_insight":     "Reframe / Insight",
        "emotional_resonance": "Emotional Resonance",
        "standalone_clarity":  "Standalone Clarity",
        "quotability":         "Quotability",
        "clean_ending":        "Clean Ending",
        "pacing_energy":       "Pacing / Energy",
    }

    # Detailed guidance per dimension — kept from the original prompts but
    # now prefixed with the dynamic point value so agents know the real weight.
    # Hook Strength (25 points) — As you move through the video, ask whether this moment could open a clip with a line that immediately grabs attention and makes complete sense without any prior context. The very first line must establish who, what, and why — without the viewer needing anything from the full video. If the opening raises an unanswered "who?", "where?", "what?" or "when?" — it is not a hook, it is a mid-entry. Never begin with a filler sound like "uh," "um," "ah," or "hmm." A good hook sounds like: "My father was a chain smoker for thirty years — and I made him a deal he couldn't refuse." or "I failed the exam three times. The fourth time changed everything." A bad hook sounds like: "He used to be a chain smoker" — who is he? Or "And that's when I realized..." — realized what? If the moment doesn't have a strong enough natural opening, go back further in the video until it does.
    # Reframe / Insight (20 points) — Once a strong opening is found, ask whether what follows offers a perspective, idea, or revelation that makes the viewer think or see something differently. This must come from the core of what the video is delivering — not from a transitional, repetitive, or setup segment that only exists to lead into something else. If the moment is purely preamble, it scores low here regardless of how well it opens.
    # Emotional Resonance (18 points) — Ask whether this moment makes the viewer feel something — curiosity, inspiration, humour, surprise, or genuine emotion that relates to the viewers of the identified category of the video. The feeling must be self-contained. A viewer who has never seen the full video should feel it just as strongly as someone who has watched everything. If the emotion only lands because of what came before in the full video, this moment is not ready to stand alone.
    # Standalone Clarity (17 points) — At this point, hold the clip as identified so far and ask: does the viewer know what is being talked about, why it matters, and how it ends? If any of these feel missing, extend the clip in either direction until it does. A clip that needs the full video to make sense scores zero here and must be reworked before moving on.
    # Quotability (12 points) — Ask whether the clip contains a line or moment that someone would want to remember, repeat, or share. This naturally follows when Hook and Insight are strong — but check that the moment lands cleanly without trailing off into filler or an unfinished thought.
    # Clean Ending (8 points) — Finally, ask whether the clip ends at an emotional or narrative peak — a punchline, a powerful takeaway, a resolved story, or a line that lands with weight. Never end while a thought is still unfinished. Never end on a filler sound like "uh," "um," or "hmm." The last word the viewer hears should feel like the right last word.

    dim_guidance = {
        "hook_strength": (
            "As you move through the video, ask whether this moment could open a clip with a line that immediately grabs attention and makes complete sense without any prior context." 
            "The very first line must establish who, what, and why — without the viewer needing anything from the full video. If the opening raises an unanswered 'who?', 'where?', 'what?' or 'when?' — it is not a hook, it is a mid-entry. "
            "Never begin with a filler sound like 'uh,' 'um,' 'ah,' or 'hmm.' A good hook sounds like: 'My father was a chain smoker for thirty years — and I made him a deal he couldn't refuse.' or 'I failed the exam three times. "
            "The fourth time changed everything."
            "A bad hook sounds like: `He used to be a chain smoker` — who is he? Or `And that's when I realized...` — realized what? If the moment doesn't have a strong enough natural opening, go back further in the video until it does."
        ),
        "reframe_insight": (
            "Once a strong opening is found, ask whether what follows offers a perspective, idea, or revelation that makes the viewer think or see something differently." 
            "This must come from the core of what the video is delivering — not from a transitional, repetitive, or setup segment that only exists to lead into something else. If the moment is purely preamble, it scores low here regardless of how well it opens."
        ),
        "emotional_resonance": (
            "Ask whether this moment makes the viewer feel something — curiosity, inspiration, humour, surprise, or genuine emotion."
            "The feeling must be self-contained. A viewer who has never seen the full video should feel it just as strongly as someone who has watched everything. If the emotion only lands because of what came before in the full video, this moment is not ready to stand alone."
        ),
        "standalone_clarity": (
            " At this point, hold the clip as identified so far and ask: does the viewer know what is being talked about, why it matters, and how it ends? If any of these feel missing, extend the clip in either direction until it does."
            "A clip that needs the full video to make sense scores zero here and must be reworked before moving on."
        ),
        "quotability": (
            "Ask whether the clip contains a line or moment that someone would want to remember, repeat, or share."
            "This naturally follows when Hook and Insight are strong — but check that the moment lands cleanly without trailing off into filler or an unfinished thought."
        ),
        "clean_ending": (
           "Finally, ask whether the clip ends at an emotional or narrative peak — a punchline, a powerful takeaway, a resolved story, or a line that lands with weight." 
           "Never end while a thought is still unfinished. Never end on a filler sound like 'uh,' 'um,' or 'hmm.' The last word the viewer hears should feel like the right last word."
        ),
        "pacing_energy": (
            "Evaluate delivery quality using measured pacing hints on windows/sentences when shown: "
            "ideal WPM 130–180, filler ratio <3%, minimal dead-air gaps (>1.5s), energetic burst delivery. "
            "Penalize draggy, filler-heavy, or pause-filled stretches. When content is equal, prefer better pacing."
        ),
    }

    lines = []
    for key, label in dim_labels.items():
        pts = weights_raw.get(key, 0)
        guidance = dim_guidance.get(key, "")
        lines.append(
            f"**{label} (score 0–{pts} points)**\n"
            f"How to evaluate: {guidance}\n"
        )

    return "\n".join(lines)


def _format_windows(windows, max_windows=500, all_words=None):
    if len(windows) > max_windows:
        step = max(1, len(windows) // max_windows)
        thinned = windows[::step][:max_windows]
    else:
        thinned = windows
    lines = []
    for w in thinned:
        words = w.text.split()
        if len(words) <= 25:
            preview = w.text
        else:
            preview = " ".join(words[:15]) + " ... " + " ".join(words[-10:])
        # Include section tag so the agent knows exactly what it is selecting
        section_tag = f"[{w.section_type}]" + (f" {w.section_label}" if w.section_label else "")
        pacing_hint = ""
        if all_words:
            p = pacing_for_clip({"start_time": w.start_time, "end_time": w.end_time}, all_words)
            pacing_hint = " " + _format_pacing_inline(p)
        lines.append(
            f"W{w.id} ({w.duration}s) [{w.start_time:.1f}s - {w.end_time:.1f}s] "
            f"{section_tag} {preview}{pacing_hint}"
        )
    return "\n".join(lines)


def _calculate_clips_count(video_duration, available_windows):
    if video_duration <= 0 or available_windows == 0:
        return 1
    avg_clip = (SEQUENTIAL_MIN_DURATION + SOCIAL_MEDIA_MAX_DURATION) / 2
    max_possible = max(1, int(video_duration / avg_clip))
    if video_duration < 180:
        clips = max(1, min(int(video_duration * 0.15 / avg_clip), 12))
    else:
        clips = max(4, min(int(video_duration * 0.15 / avg_clip), 12))
    return min(clips, available_windows, max_possible)


def _agent_sequence(windows, clips_count, scoring_config, regenerate_feedback=None,
                    used_window_ids=None, frames_b64=None, all_words=None):
    if used_window_ids is None:
        used_window_ids = set()
    windows_text = _format_windows(windows, all_words=all_words)

    # ── Prompt caching ────────────────────────────────────────────────────────
    # windows_text is large and identical on every iteration (only the feedback
    # section changes). Splitting it into a separate cached_context block means
    # Anthropic stores it once and serves it at ~10% of the normal token cost
    # on every subsequent call within the same 5-minute session.
    cached_context = (
        "## AVAILABLE WINDOWS (transcript segments)\n\n"
        "The windows below are the only segments you may select from.\n\n"
        + windows_text
    )

    feedback_section = ""
    if regenerate_feedback:
        used_text = ", ".join(f"W{wid}" for wid in sorted(used_window_ids)) if used_window_ids else "None yet"
        feedback_section = f"""
### REGENERATION CONTEXT

**CRITICAL: Windows already used (DO NOT SELECT THESE)**:
{used_text}

You must pick a DIFFERENT window that is NOT in the list above.

---

You previously generated a short that scored {regenerate_feedback['score']}/100 (target: {TARGET_SCORE}+).

**Issues identified**:
- Weakest factor: {regenerate_feedback['weakest_factor']}
- Problem: {regenerate_feedback['reason']}

**How to improve**: {regenerate_feedback['improvise']}

YOUR TASK: Pick a DIFFERENT window that addresses these issues.
"""
        clips_count = 1

    visual_section = ""
    if frames_b64:
        visual_section = """
### VISUAL CONTEXT PROVIDED

You have access to video frames extracted every 30 seconds. Use these to:
- **Detect interview format**: Look for multiple people in frame (interviewer + guest)
- **Identify speaker changes**: Visual cues when question transitions to answer
- **Verify content type**: Is this Q&A format or monologue?

**CRITICAL for Interview Content**:
- If you see 2+ people in frames → This is likely interview format
- For interview windows: MUST include both question AND answer
- **REJECT any window that is just an interviewer asking a question**
- The valuable content is the GUEST'S RESPONSE, not the question
- A good interview clip: 10% question + 90% answer
"""

    # Dynamic part of the prompt — does NOT include windows_text (it's cached above)
    weights_raw = scoring_config["weights_raw"]
    prompt = SEQUENCE_PROMPT.format(
        feedback_section=feedback_section + visual_section,
        windows_text="[See the AVAILABLE WINDOWS context provided above]",
        scoring_context=_build_scoring_context(scoring_config),
        w_hook=weights_raw["hook_strength"],
        w_reframe=weights_raw["reframe_insight"],
        w_emotion=weights_raw["emotional_resonance"],
        w_clarity=weights_raw["standalone_clarity"],
        w_quotable=weights_raw["quotability"],
        w_ending=weights_raw["clean_ending"],
        w_pacing=weights_raw["pacing_energy"],
        social_media_max=SOCIAL_MEDIA_MAX_DURATION,
    )

    if frames_b64:
        # Multimodal path: frames are already big; pass cached_context as text prefix
        result = get_multimodal_response(
            cached_context + "\n\n" + _build_prompt(prompt, SEQUENCE_RESPONSE_FORMAT),
            frames_b64,
            response_format=None,   # already embedded in prompt string above
            caller="agent1_sequence",
        )
    else:
        result = get_response(
            prompt,
            response_format=SEQUENCE_RESPONSE_FORMAT,
            caller="agent1_sequence",
            cached_context=cached_context,   # ← cache the windows block
        )

    window_by_id = {w.id: w for w in windows}
    seq_clips = []
    for item in result.get("clips", []):
        wid = _parse_window_id(item.get("WindowId"))
        if not wid or wid not in window_by_id:
            continue
        used_window_ids.add(wid)
        seq_clips.append(_window_to_clip(window_by_id[wid], item.get("reason", "")))
    return seq_clips


def _parse_window_id(wid):
    """Normalise a WindowId value (e.g. 'W12', '12', 12) to an int, or None."""
    if isinstance(wid, str):
        wid = wid.replace("W", "").replace("w", "").strip()
    if wid is None or wid == "":
        return None
    try:
        return int(wid)
    except (ValueError, TypeError):
        return None


def _window_to_clip(w, reason=""):
    """Build a clip dict from a CandidateWindow."""
    return {
        "type": "sequence",
        "text": w.text,
        "start_time": w.start_time,
        "end_time": w.end_time,
        "duration": w.duration,
        "section_type": w.section_type,
        "section_label": w.section_label,
        "clips": [{"start_time": w.start_time, "end_time": w.end_time, "text": w.text}],
        "reason": reason,
    }


def _refine_clip(current, windows, scoring_config, feedback, frames_b64=None, all_words=None):
    """
    Agent #3 — in-place boundary refinement.

    Instead of asking the LLM for a brand-new window (which jumps to a different
    topic and risks duplicates), we give it ONLY the windows that overlap or
    neighbour the current clip within the SAME speaker section — at full
    resolution (no thinning) — and ask it to pick the boundary variant that
    fixes the weakest factor while keeping the same core message.

    Returns a refined clip dict, or None if no suitable neighbour set exists.
    """
    cur_start   = current["start_time"]
    cur_end     = current["end_time"]
    cur_section = current.get("section_type")

    lo = cur_start - REFINE_PAD
    hi = cur_end + REFINE_PAD
    neighbours = [
        w for w in windows
        if (cur_section is None or w.section_type == cur_section)
        and w.end_time >= lo and w.start_time <= hi
    ]
    if not neighbours:
        log.info(f"[REFINE] No neighbour windows around {cur_start:.1f}s–{cur_end:.1f}s")
        return None

    # Full-resolution option list (small region → no aggressive thinning).
    options = _format_windows(neighbours, max_windows=REFINE_MAX_OPTIONS, all_words=all_words)

    weights_raw = scoring_config["weights_raw"]
    prompt = REFINE_PROMPT.format(
        score=feedback["score"],
        target=TARGET_SCORE,
        cur_start=cur_start,
        cur_end=cur_end,
        cur_duration=current.get("duration", cur_end - cur_start),
        cur_text=current.get("text", ""),
        pacing_section=_build_pacing_section(current, all_words),
        weakest_factor=feedback["weakest_factor"],
        reason=feedback["reason"],
        improvise=feedback["improvise"],
        scoring_context=_build_scoring_context(scoring_config),
        social_media_max=SOCIAL_MEDIA_MAX_DURATION,
        options=options,
    )

    if frames_b64:
        result = get_multimodal_response(
            _build_prompt(prompt, SEQUENCE_RESPONSE_FORMAT),
            frames_b64,
            response_format=None,
            caller="agent3_refine",
        )
    else:
        result = get_response(
            prompt,
            response_format=SEQUENCE_RESPONSE_FORMAT,
            caller="agent3_refine",
        )

    window_by_id = {w.id: w for w in neighbours}
    for item in result.get("clips", []):
        wid = _parse_window_id(item.get("WindowId"))
        if wid and wid in window_by_id:
            return _window_to_clip(window_by_id[wid], item.get("reason", ""))
    return None


def _agent_sequence_reviewer(seq_shorts, windows, scoring_config, frames_b64=None, all_words=None):
    """
    Agent #3 — score every clip and, when it falls below TARGET_SCORE, REFINE
    the same clip in place (adjust its start/end boundaries) rather than picking
    an unrelated window. We keep the best-scoring version seen across all
    refinement attempts, so a refinement can never make a clip worse.
    """
    iteration_log = {"iterations": 0, "regenerated": 0, "final_scores": []}
    reviewed = []

    for idx, short in enumerate(seq_shorts):
        short_id = f"seq_{idx + 1}"
        short["short_id"] = short_id
        current = short
        best = None  # best-scoring snapshot seen so far

        # Budget: 1 initial score + MAX_ITERATIONS_PER_SHORT_SEQUENCE refinement rounds.
        for iteration in range(MAX_ITERATIONS_PER_SHORT_SEQUENCE + 1):
            try:
                score_result = score_single_short(current, scoring_config, all_words=all_words)
            except (RuntimeError, ValueError) as e:
                log.error(f"[{short_id}] Scoring failed: {e}. Keeping current version.")
                break

            total = score_result["total_score"]
            log.info(f"{short_id} iter {iteration + 1}: score={total}/100")

            snapshot = dict(current)
            snapshot.update(score_result)
            snapshot["short_id"]   = short_id
            snapshot["iterations"] = iteration
            if best is None or snapshot["total_score"] > best["total_score"]:
                best = snapshot

            if total >= TARGET_SCORE:
                break
            if iteration >= MAX_ITERATIONS_PER_SHORT_SEQUENCE:
                break  # refinement budget exhausted

            feedback = {
                "score":          total,
                "weakest_factor": score_result["weakest_factor"],
                "reason":         score_result["reason"],
                "improvise":      score_result["improvise"],
            }
            log.info(f"{short_id} refine {iteration + 1} before: {current.get('text', '')[:120]}...")
            iteration_log["iterations"] += 1
            refined = _refine_clip(current, windows, scoring_config, feedback, frames_b64=frames_b64, all_words=all_words)
            if not refined:
                break  # no neighbour options — keep best so far
            # No progress (LLM returned the same boundaries) → stop refining.
            if (abs(refined["start_time"] - current["start_time"]) < 0.05 and
                    abs(refined["end_time"] - current["end_time"]) < 0.05):
                log.info(f"{short_id} refine {iteration + 1}: unchanged boundaries — stopping")
                break
            iteration_log["regenerated"] += 1
            refined["short_id"] = short_id
            log.info(f"{short_id} refine {iteration + 1} after:  {refined.get('text', '')[:120]}...")
            current = refined

        final = best if best is not None else current
        reviewed.append(final)
        iteration_log["final_scores"].append({
            "short_id":    short_id,
            "final_score": final.get("ConfidenceScore", 0),
            "iterations":  final.get("iterations", 0),
        })

    return reviewed, iteration_log


def sequential_selection(sentences, windows, video_duration, scoring_config, frames_b64=None, all_words=None):
    try:
        clips_count = _calculate_clips_count(video_duration, len(windows))
        log.info(f"[SEQ] Targeting {clips_count} clips from {len(windows)} windows")
        seq_clips = _agent_sequence(windows, clips_count, scoring_config=scoring_config, frames_b64=frames_b64, all_words=all_words)
        log.info(f"[SEQ] Agent #1 generated {len(seq_clips)} clips")
        reviewed, log_data = _agent_sequence_reviewer(seq_clips, windows, scoring_config, frames_b64=frames_b64, all_words=all_words)
        log.info(f"[SEQ] Agent #3 reviewed {len(reviewed)}, regenerated {log_data['regenerated']}")
        return reviewed, log_data
    except Exception as e:
        log.error(f"[SEQ] FAILED: {repr(e)}\n{traceback.format_exc()}")
        return [], {}


# ── Stage 3b: Non-sequential selection (Agent #2 + #4) ────────────────────────

def _calculate_target_count(video_duration):
    avg_short_dur = 40
    max_possible = max(1, int(video_duration / avg_short_dur))
    if video_duration < 180:
        target = max(1, min(int(video_duration * 0.12 / avg_short_dur), 10))
    else:
        target = max(4, min(int(video_duration * 0.12 / avg_short_dur), 10))
    return min(target, max_possible)


def _agent_nonsequence(sentences, clips_count, scoring_config, regenerate_feedback=None, all_words=None):
    sentences_text = _format_sentences(sentences, all_words)

    # ── Prompt caching ────────────────────────────────────────────────────────
    # The full sentences list is identical on every iteration; only the feedback
    # section changes. Caching it separately saves ~90% of its token cost after
    # the first call within the same 5-minute window.
    cached_context = (
        "## AVAILABLE SENTENCES (full transcript)\n\n"
        "Select sentence IDs only from the list below.\n\n"
        + sentences_text
    )

    feedback_section = ""
    if regenerate_feedback:
        old_note = ""
        if "old_sentence_ids" in regenerate_feedback:
            old_note = (
                f"\n**Sentences used in previous attempt**: "
                f"{', '.join(f'S{sid}' for sid in regenerate_feedback['old_sentence_ids'])}"
            )
        feedback_section = f"""
### REGENERATION CONTEXT
{old_note}

You must pick a DIFFERENT set of sentences (avoid the ones above if provided).

---

You previously generated a short that scored {regenerate_feedback['score']}/100 (target: {TARGET_SCORE}+).

**Issues identified**:
- Weakest factor: {regenerate_feedback['weakest_factor']}
- Problem: {regenerate_feedback['reason']}

**How to improve**: {regenerate_feedback['improvise']}

YOUR TASK: Pick a DIFFERENT set of sentences that addresses these issues.
"""
        clips_count = 1

    # Dynamic prompt — sentences body replaced by reference to the cached block
    weights_raw = scoring_config["weights_raw"]
    prompt = NONSEQUENCE_PROMPT.format(
        feedback_section=feedback_section,
        sentences_text="[See the AVAILABLE SENTENCES context provided above]",
        scoring_context=_build_scoring_context(scoring_config),
        w_hook=weights_raw["hook_strength"],
        w_reframe=weights_raw["reframe_insight"],
        w_emotion=weights_raw["emotional_resonance"],
        w_clarity=weights_raw["standalone_clarity"],
        w_quotable=weights_raw["quotability"],
        w_ending=weights_raw["clean_ending"],
        w_pacing=weights_raw["pacing_energy"],
    )

    result = get_response(
        prompt,
        response_format=NONSEQUENCE_RESPONSE_FORMAT,
        caller="agent2_nonsequence",
        cached_context=cached_context,   # ← cache the sentences block
    )

    sentence_by_idx = {s.index: s for s in sentences}
    nonseq_shorts = []
    raw_shorts = result.get("shorts", [])
    log.info(f"[NONSEQ] LLM returned {len(raw_shorts)} raw shorts")

    for short_data in raw_shorts:
        assembled_short = _assemble_nonseq_short(
            short_data.get("sentence_ids", []),
            sentence_by_idx,
            topic=short_data.get("topic", ""),
            reason=short_data.get("reason", ""),
        )
        if assembled_short:
            nonseq_shorts.append(assembled_short)
    return nonseq_shorts


def _assemble_nonseq_short(sentence_ids, sentence_by_idx, topic="", reason=""):
    """Build a non-sequential short dict from a list of sentence IDs."""
    assembled = []
    full_text_parts = []
    clean_ids = []
    for sid in sentence_ids:
        try:
            sid_int = int(sid)
        except (ValueError, TypeError):
            continue
        if sid_int in sentence_by_idx:
            s = sentence_by_idx[sid_int]
            assembled.append({
                "start_time": s.start,
                "end_time": s.end,
                "duration": round(s.end - s.start, 1),
                "text": s.text,
            })
            full_text_parts.append(s.text)
            clean_ids.append(sid_int)
    if not assembled:
        return None
    total_dur = sum(c["duration"] for c in assembled)
    return {
        "type": "non-sequence",
        "topic": topic,
        "text": " ".join(full_text_parts),
        "start_time": assembled[0]["start_time"],
        "end_time": assembled[-1]["end_time"],
        "duration": round(total_dur, 1),
        "num_clips": len(assembled),
        "clips": assembled,
        "reason": reason,
        "_sentence_ids": clean_ids,
    }


def _refine_nonseq_short(current, sentences, scoring_config, feedback, all_words=None):
    """
    Agent #4 — in-place refinement for non-sequential shorts.

    Unlike sequential clips (a single window), a non-sequential short is assembled
    from any sentences across the video, so the LLM is given the FULL sentence list
    (reusing the same cached_context block Agent #2 uses → cache hit, low token cost)
    and asked to ADD / REMOVE / RE-ORDER sentences to fix the weakest factor while
    keeping the SAME core story — never a brand-new topic.

    Returns a refined short dict, or None.
    """
    sentences_text = _format_sentences(sentences, all_words)
    # Byte-identical to Agent #2's cached_context → same Anthropic prompt-cache entry.
    cached_context = (
        "## AVAILABLE SENTENCES (full transcript)\n\n"
        "Select sentence IDs only from the list below.\n\n"
        + sentences_text
    )

    # ── Graft-distance guard (B plan) ──────────────────────────────────────────
    # Only sentences within REFINE_NONSEQ_PAD seconds of the short's current span
    # are allowed, so refinement can't pull in an unrelated line from elsewhere.
    clip_starts = [c["start_time"] for c in current.get("clips", [])]
    clip_ends   = [c["end_time"]   for c in current.get("clips", [])]
    if clip_starts and clip_ends:
        lo_time = min(clip_starts) - REFINE_NONSEQ_PAD
        hi_time = max(clip_ends)   + REFINE_NONSEQ_PAD
    else:
        lo_time, hi_time = float("-inf"), float("inf")

    allowed_ids = sorted(
        s.index for s in sentences if s.start >= lo_time and s.end <= hi_time
    )
    if not allowed_ids:
        allowed_ids = sorted(current.get("_sentence_ids", []))
    allowed_set = set(allowed_ids)
    allowed_lo, allowed_hi = allowed_ids[0], allowed_ids[-1]

    cur_ids = current.get("_sentence_ids", [])
    prompt = REFINE_NONSEQ_PROMPT.format(
        score=feedback["score"],
        target=TARGET_SCORE,
        allowed_lo=allowed_lo,
        allowed_hi=allowed_hi,
        cur_duration=current.get("duration", 0),
        cur_num=current.get("num_clips", len(cur_ids)),
        cur_ids=", ".join(f"S{i}" for i in cur_ids) if cur_ids else "unknown",
        cur_text=current.get("text", ""),
        pacing_section=_build_pacing_section(current, all_words),
        weakest_factor=feedback["weakest_factor"],
        reason=feedback["reason"],
        improvise=feedback["improvise"],
        scoring_context=_build_scoring_context(scoring_config),
    )

    result = get_response(
        prompt,
        response_format=NONSEQUENCE_RESPONSE_FORMAT,
        caller="agent4_refine",
        cached_context=cached_context,   # ← reuses Agent #2's cached sentence list
    )

    sentence_by_idx = {s.index: s for s in sentences}
    for short_data in result.get("shorts", []):
        # Enforce the graft-distance guard: drop any out-of-range sentence the LLM
        # picked despite the instruction.
        raw_ids = short_data.get("sentence_ids", [])
        filtered_ids = []
        dropped = []
        for sid in raw_ids:
            try:
                sid_int = int(sid)
            except (ValueError, TypeError):
                continue
            if sid_int in allowed_set:
                filtered_ids.append(sid_int)
            else:
                dropped.append(sid_int)
        if dropped:
            log.info(f"[NONSEQ-REFINE] Dropped out-of-range sentences "
                     f"{dropped} (allowed S{allowed_lo}-S{allowed_hi})")
        refined = _assemble_nonseq_short(
            filtered_ids,
            sentence_by_idx,
            topic=short_data.get("topic", current.get("topic", "")),
            reason=short_data.get("reason", ""),
        )
        if refined:
            return refined
    return None


def _agent_nonsequence_reviewer(nonseq_shorts, sentences, scoring_config, all_words=None):
    """
    Agent #4 — score every non-sequential short and, when it falls below
    TARGET_SCORE, REFINE the same short in place (adjust which sentences are
    included) rather than assembling an unrelated new short. The best-scoring
    version across all refinement attempts is kept, so refinement can never make
    a short worse.
    """
    iteration_log = {"iterations": 0, "regenerated": 0, "final_scores": []}
    reviewed = []
    for idx, short in enumerate(nonseq_shorts):
        short_id = f"nonseq_{idx + 1}"
        short["short_id"] = short_id
        current = short
        best = None  # best-scoring snapshot seen so far

        for iteration in range(MAX_ITERATIONS_PER_SHORT_NONSEQUENCE + 1):
            try:
                score_result = score_single_short(current, scoring_config, all_words=all_words)
            except (RuntimeError, ValueError) as e:
                log.error(f"[{short_id}] Scoring failed: {e}. Keeping current version.")
                break

            total = score_result["total_score"]
            log.info(f"{short_id} iter {iteration + 1}: score={total}/100")

            snapshot = dict(current)
            snapshot.update(score_result)
            snapshot["short_id"]   = short_id
            snapshot["iterations"] = iteration
            if best is None or snapshot["total_score"] > best["total_score"]:
                best = snapshot

            if total >= TARGET_SCORE:
                break
            if iteration >= MAX_ITERATIONS_PER_SHORT_NONSEQUENCE:
                break  # refinement budget exhausted

            feedback = {
                "score":          total,
                "weakest_factor": score_result["weakest_factor"],
                "reason":         score_result["reason"],
                "improvise":      score_result["improvise"],
            }
            log.info(f"{short_id} refine {iteration + 1} before: {current.get('text', '')[:120]}...")
            iteration_log["iterations"] += 1
            refined = _refine_nonseq_short(current, sentences, scoring_config, feedback, all_words=all_words)
            if not refined:
                break  # no refinement produced — keep best so far
            # No change in selection → stop refining (avoids a wasted re-score).
            if sorted(refined.get("_sentence_ids", [])) == sorted(current.get("_sentence_ids", [])):
                log.info(f"{short_id} refine {iteration + 1}: unchanged selection — stopping")
                break
            iteration_log["regenerated"] += 1
            refined["short_id"] = short_id
            log.info(f"{short_id} refine {iteration + 1} after:  {refined.get('text', '')[:120]}...")
            current = refined

        final = best if best is not None else current
        reviewed.append(final)
        iteration_log["final_scores"].append({
            "short_id": short_id,
            "final_score": final.get("ConfidenceScore", 0),
            "iterations": final.get("iterations", 0),
        })

    return reviewed, iteration_log


def non_sequential_selection(sentences, video_duration, scoring_config, all_words=None):
    try:
        if video_duration < NON_SEQUENTIAL_MIN_DURATION:
            log.info(f"[NONSEQ] Video too short ({video_duration}s) — skipping")
            return [], {}
        clips_count = _calculate_target_count(video_duration)
        log.info(f"[NONSEQ] Targeting {clips_count} shorts")
        nonseq_shorts = _agent_nonsequence(sentences, clips_count, scoring_config=scoring_config, all_words=all_words)
        log.info(f"[NONSEQ] Agent #2 generated {len(nonseq_shorts)} shorts")
        reviewed, log_data = _agent_nonsequence_reviewer(nonseq_shorts, sentences, scoring_config, all_words=all_words)
        log.info(f"[NONSEQ] Agent #4 reviewed {len(reviewed)}, regenerated {log_data['regenerated']}")
        return reviewed, log_data
    except Exception as e:
        log.error(f"[NONSEQ] FAILED: {repr(e)}\n{traceback.format_exc()}")
        return [], {}


# ── Stage 4: Final metadata (Agent #5) ────────────────────────────────────────

def final_metadata(sequential, non_sequential, language_code):
    all_shorts = sequential + non_sequential
    if not all_shorts:
        return sequential, non_sequential

    parts = []
    for s in all_shorts:
        parts.append(
            f"Short (ID: {s.get('short_id', '')})\n"
            f"Type: {s['type']} | Duration: {s['duration']:.1f}s | Score: {s.get('ConfidenceScore', 0)}/100\n"
            f"Text: \"{' '.join(s['text'].split()[:80])}...\""
        )
    context_text = "\n\n".join(parts)

    prompt = METADATA_PROMPT.format(
        short_count=len(all_shorts),
        shorts_context=context_text,
        language_code=language_code,
    )

    result = get_response(
        prompt, temperature=0.3,
        response_format=METADATA_RESPONSE_FORMAT, caller="agent5_metadata",
    )

    by_id = {m["short_id"]: m for m in result.get("shorts", [])}
    for short in all_shorts:
        sid = short.get("short_id", "")
        if sid in by_id:
            meta = by_id[sid]
            short["Title"] = meta.get("title", "")
            short["SocialMedia"] = meta.get("platforms", ["instagram", "tiktok", "youtube"])
        else:
            short["Title"] = short.get("Title", "Untitled Short")
            short["SocialMedia"] = short.get("SocialMedia", ["instagram", "tiktok", "youtube"])

    seq_count = len(sequential)
    log.info(f"[META] Generated metadata for {len(all_shorts)} shorts")
    return all_shorts[:seq_count], all_shorts[seq_count:]


# ── Stage 5: Ranking ──────────────────────────────────────────────────────────

def ranking(sequential, non_sequential):
    sequential = sorted(sequential, key=lambda c: c.get("start_time", 0))
    non_sequential = sorted(non_sequential, key=lambda s: s.get("ConfidenceScore", 0), reverse=True)
    log.info(f"[RANK] {len(sequential)} sequential, {len(non_sequential)} non-sequential")
    return sequential, non_sequential


# ── Stage 6: Thumbnail Generation (Optional) ──────────────────────────────────

def generate_thumbnails(sequential, non_sequential, frames_b64):
    """Generate thumbnails for top clips if ENABLE_THUMBNAILS is True."""
    if not ENABLE_THUMBNAILS:
        log.info("[THUMBNAIL] Skipped (ENABLE_THUMBNAILS = False)")
        return sequential, non_sequential
    
    if not all([OpenAI, Image, boto3]):
        log.error("[THUMBNAIL] Missing dependencies (openai, PIL, boto3) - skipping")
        return sequential, non_sequential
    
    if not OPENAI_API_KEY:
        log.error("[THUMBNAIL] OPENAI_API_KEY not set - skipping")
        return sequential, non_sequential
    
    if not frames_b64:
        log.error("[THUMBNAIL] No video frames available - skipping")
        return sequential, non_sequential

    log.info(f"[THUMBNAIL] Starting generation for top clips")
    
    # Get top clips by confidence score
    top_seq = sorted(sequential, key=lambda c: c.get("confidence_score", 0), reverse=True)[:MAX_THUMBNAILS_PER_TYPE]
    top_nonseq = sorted(non_sequential, key=lambda s: s.get("confidence_score", 0), reverse=True)[:MAX_THUMBNAILS_PER_TYPE]
    
    all_clips = top_seq + top_nonseq
    if not all_clips:
        log.info("[THUMBNAIL] No clips to process")
        return sequential, non_sequential
    
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        for clip in all_clips:
            clip_id = clip.get("debug_id", "unknown")
            title = clip.get("title", "")
            text = clip.get("text", "")
            
            if not title:
                continue
                
            log.info(f"[THUMBNAIL] Generating for {clip_id}: {title[:60]}")
            
            try:
                # Get frame at 30% into clip
                start_time = clip.get("video_start_time", 0)
                end_time = clip.get("video_end_time", 0)
                target_time = start_time + (end_time - start_time) * 0.30
                frame_index = max(0, min(int(target_time / FRAME_INTERVAL), len(frames_b64) - 1))
                
                # Create temp image file
                temp_dir = tempfile.mkdtemp()
                image_path = os.path.join(temp_dir, f"{clip_id}_frame.jpg")
                
                with open(image_path, "wb") as f:
                    f.write(base64.b64decode(frames_b64[frame_index]))
                
                # Generate thumbnail URL (simplified - just store the path)
                clip["thumbnail_url"] = f"thumbnail_{clip_id}.jpg"
                log.info(f"[THUMBNAIL] ✓ Generated for {clip_id}")
                
                # Cleanup
                os.remove(image_path)
                os.rmdir(temp_dir)
                
            except Exception as e:
                log.error(f"[THUMBNAIL] ✗ Failed for {clip_id}: {repr(e)}")
                clip["thumbnail_url"] = ""
    
    except Exception as e:
        log.error(f"[THUMBNAIL] Generation failed: {repr(e)}")
    
    return sequential, non_sequential


# ── Output formatting ─────────────────────────────────────────────────────────

def format_sequential(clips, all_words=None):
    out = []
    for clip in clips:
        pacing = pacing_for_clip(clip, all_words) if all_words else None
        if pacing:
            log.info(
                f"[PACING] seq {clip.get('short_id', '')}: score {pacing['pacing_score']}/20 "
                f"(wpm {pacing['wpm']}, filler {pacing['filler_ratio']:.1%}, "
                f"dead_air {pacing['dead_air_gaps']}, burst {pacing['burst_ratio']:.1%})"
            )
        out.append({
            "debug_id": clip.get("short_id", ""),
            "title": clip.get("Title", ""),
            "text": clip["text"],
            "confidence_score": clip.get("ConfidenceScore", 0),
            "pacing": pacing,
            "category": "standard",
            "section_type": clip.get("section_type", ""),
            "section_label": clip.get("section_label", ""),
            "social_media": clip.get("SocialMedia", []),
            "video_start_time": clip["start_time"],
            "video_end_time": clip["end_time"],
            "score_breakdown": clip.get("ScoreBreakdown", {}),
            "score_reasoning": clip.get("ScoreReasoning", {}),
            "reason": clip.get("reason", ""),
            "weakest_factor": clip.get("weakest_factor", ""),
            "iterations": clip.get("iterations", 0),
            "thumbnail_url": clip.get("thumbnail_url", ""),
        })
    return out


def format_non_sequential(shorts, all_words=None):
    out = []
    for short in shorts:
        pacing = pacing_for_clip(short, all_words) if all_words else None
        if pacing:
            log.info(
                f"[PACING] nonseq {short.get('short_id', '')}: score {pacing['pacing_score']}/20 "
                f"(wpm {pacing['wpm']}, filler {pacing['filler_ratio']:.1%}, "
                f"dead_air {pacing['dead_air_gaps']}, burst {pacing['burst_ratio']:.1%})"
            )
        out.append({
            "short_id": short.get("short_id", ""),
            "debug_id": short.get("short_id", ""),
            "title": short.get("Title", ""),
            "confidence_score": short.get("ConfidenceScore", 0),
            "pacing": pacing,
            "category": "standard",
            "social_media": short.get("SocialMedia", []),
            "total_duration": short.get("duration", 0),
            "num_clips": short.get("num_clips", 0),
            "clips": [
                {
                    "startTime": c["start_time"],
                    "endTime": c["end_time"],
                    "duration": c["duration"],
                    "text": c["text"],
                }
                for c in short.get("clips", [])
            ],
            "score_breakdown": short.get("ScoreBreakdown", {}),
            "score_reasoning": short.get("ScoreReasoning", {}),
            "reason": short.get("reason", ""),
            "weakest_factor": short.get("weakest_factor", ""),
            "iterations": short.get("iterations", 0),
            "thumbnail_url": short.get("thumbnail_url", ""),
        })
    return out


# ── Speaker reference JSON ────────────────────────────────────────────────────

def _sentence_ids_to_contiguous_segments(sentence_ids, sentence_by_index):
    """
    Convert a flat list of sentence indices into contiguous time-range segments.
    Non-contiguous blocks (e.g. presenter speaks, then clip, then presenter again)
    become separate segment dicts each with start/end time and sentence range.
    """
    if not sentence_ids:
        return []

    sorted_ids = sorted(sentence_ids)
    segments = []
    seg_start = sorted_ids[0]
    seg_prev  = sorted_ids[0]

    for idx in sorted_ids[1:]:
        if idx == seg_prev + 1:
            seg_prev = idx
        else:
            # gap — close current segment and start new one
            s0 = sentence_by_index.get(seg_start)
            s1 = sentence_by_index.get(seg_prev)
            if s0 and s1:
                segments.append({
                    "start_sentence_index": seg_start,
                    "end_sentence_index":   seg_prev,
                    "start_time":           s0.start,
                    "end_time":             s1.end,
                    "duration":             round(s1.end - s0.start, 2),
                })
            seg_start = idx
            seg_prev  = idx

    # close final segment
    s0 = sentence_by_index.get(seg_start)
    s1 = sentence_by_index.get(seg_prev)
    if s0 and s1:
        segments.append({
            "start_sentence_index": seg_start,
            "end_sentence_index":   seg_prev,
            "start_time":           s0.start,
            "end_time":             s1.end,
            "duration":             round(s1.end - s0.start, 2),
        })
    return segments


def save_speakers_json(speakers, note, sentences, output_path):
    """
    Build and write a speakers.json reference file from Stage 1b speaker detection.

    Matches the shape of transcript_speakers.json:
      id, name, role, context, segments (contiguous time ranges derived from
      sentence_ids), full_transcript, and per-sentence detail.

    Written immediately after Stage 1b — available even if later stages fail.
    Use it to inspect and verify speaker boundaries before re-running.
    """
    sentence_by_index = {s.index: s for s in sentences}

    output_speakers = []
    for sp in speakers:
        sentence_ids = sp.get("sentence_ids", [])

        # Build per-sentence list (ordered)
        sp_sentences = []
        for idx in sorted(sentence_ids):
            s = sentence_by_index.get(idx)
            if s:
                sp_sentences.append({
                    "index": s.index,
                    "start": s.start,
                    "end":   s.end,
                    "text":  s.text,
                })

        full_text = " ".join(s["text"] for s in sp_sentences)

        # Derive contiguous segments from sentence_ids
        segments = _sentence_ids_to_contiguous_segments(sentence_ids, sentence_by_index)
        total_duration = round(sum(seg["duration"] for seg in segments), 2)

        output_speakers.append({
            "id":             sp.get("id"),
            "name":           sp.get("name", ""),
            "role":           sp.get("role", ""),
            "context":        sp.get("context", ""),
            "total_duration": total_duration,
            "sentence_count": len(sp_sentences),
            "segments":       segments,
            "full_transcript": full_text,
            "sentences":      sp_sentences,
        })

    payload = {
        "source": "Stage 1b — LLM speaker detection (sentence-level assignment)",
        "note": note,
        "total_speakers": len(output_speakers),
        "speakers": output_speakers,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    log.info(f"[SPEAKERS] Reference JSON written → {output_path} ({len(output_speakers)} speakers)")
    return payload


# ── Entry point ───────────────────────────────────────────────────────────────

def run(transcription_data, video_path=None, clip_mode="both"):
    """Programmatic entry point. Returns the result dict."""
    frames_b64 = []
    if video_path:
        frames_b64 = extract_frames_from_file(video_path)
        log.info(f"[AGENT] Multimodal {'enabled' if frames_b64 else 'disabled'}")

    sentences, video_duration, language_code, all_words = preprocess(transcription_data)

    scoring_config = get_scoring_config()
    log.info(f"[AGENT] Standard dimension weights: {scoring_config['weights_raw']}")

    # ── Stage 1b: Speaker detection ──────────────────────────────────────────
    log.info("[AGENT] Stage 1b: Detecting speakers and boundaries...")
    speakers, speakers_note = detect_speakers(sentences)
    sentence_to_speaker = _build_sentence_to_speaker_map(speakers, sentences)
    log.info(f"[AGENT] {len(speakers)} speakers identified")

    # Write speaker reference JSON immediately after detection so it is
    # available even if the clipping pipeline fails in a later stage.
    speakers_payload = save_speakers_json(speakers, speakers_note, sentences, SPEAKERS_OUTPUT_PATH)

    # ── Stage 2: Speaker-boundary-aware segmentation ──────────────────────────
    windows = segment(sentences, video_duration, sentence_to_speaker)

    sequential_clips, seq_log = [], {}
    non_sequential_shorts, nonseq_log = [], {}

    if clip_mode in ("both", "sequential"):
        sequential_clips, seq_log = sequential_selection(
            sentences, windows, video_duration,
            scoring_config=scoring_config,
            frames_b64=frames_b64,
            all_words=all_words,
        )

    if clip_mode in ("both", "non_sequential"):
        non_sequential_shorts, nonseq_log = non_sequential_selection(
            sentences, video_duration,
            scoring_config=scoring_config,
            all_words=all_words,
        )

    sequential_clips, non_sequential_shorts = final_metadata(
        sequential_clips, non_sequential_shorts, language_code,
    )
    sequential_clips, non_sequential_shorts = ranking(sequential_clips, non_sequential_shorts)
    
    # ── Stage 6: Thumbnail generation (optional) ───────────────────────────────
    if ENABLE_THUMBNAILS:
        sequential_clips, non_sequential_shorts = generate_thumbnails(
            sequential_clips, non_sequential_shorts, frames_b64
        )

    return {
        "video_duration": video_duration,
        "language_code": language_code,
        "dimension_weights": scoring_config["weights_raw"],
        # ── Stage 1b outputs ──────────────────────────────────────────────────
        "speakers": speakers_payload,
        "speakers_file": SPEAKERS_OUTPUT_PATH,
        # ─────────────────────────────────────────────────────────────────────
        "sentence_count": len(sentences),
        "window_count": len(windows),
        "sequential_clips": format_sequential(sequential_clips, all_words),
        "non_sequential_shorts": format_non_sequential(non_sequential_shorts, all_words),
        "iteration_log": {
            "sequential": seq_log,
            "non_sequential": nonseq_log,
        },
        "token_usage": get_token_usage(),
    }


def main():
    if not os.path.isfile(TRANSCRIPT_PATH):
        raise FileNotFoundError(f"TRANSCRIPT_PATH not found: {TRANSCRIPT_PATH}")

    with open(TRANSCRIPT_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    transcription_data = raw.get("results", raw)

    video_path = VIDEO_PATH or None
    result = run(transcription_data, video_path=video_path, clip_mode=CLIP_MODE)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    log.info(
        f"[DONE] {len(result['sequential_clips'])} sequential, "
        f"{len(result['non_sequential_shorts'])} non-sequential — written to {OUTPUT_PATH}"
    )
    log.info(f"[WEIGHTS] Standard dimensions: {result['dimension_weights']}")
    log.info(f"[SPEAKERS] {result.get('speakers', {}).get('total_speakers', 0)} speakers detected")
    log.info(f"[SPEAKERS] Reference file → {result.get('speakers_file', '')}")
    usage = result["token_usage"]
    log.info(
        f"[TOKENS] total: {usage['total_tokens']:,} | "
        f"cache_read: {usage['total_cache_read_tokens']:,} | "
        f"cache_write: {usage['total_cache_write_tokens']:,} | "
        f"saved≈: {usage['total_tokens_saved']:,} tokens"
    )


if __name__ == "__main__":
    main()

"""
STT prompts.

All prompt strings used by stt/gemini.py live here.
To change transcription behaviour, edit this file — not the STT implementation.
"""

TRANSCRIBE = """\
Transcribe this audio with speaker diarization and timestamps.

Output a JSON array. Each element is one continuous speaker turn:
  start_ms      — turn start in milliseconds (integer)
  end_ms        — turn end in milliseconds (integer)
  speaker_label — speaker identifier (e.g. "Speaker A", "Speaker B")
  text          — verbatim transcript in the original script of whatever language is spoken

Script rules:
- Write Hindi in Devanagari, Telugu in Telugu script, English in Latin, etc.
- Preserve code-switching exactly as spoken — do not transliterate or translate.
- Do not add language labels, translations, or explanations anywhere.

Return ONLY the JSON array. No markdown, no code fences, no commentary.\
"""

# IndraAstra BD Pitch — Targeting Mem0 Partnership

IndraAstra AI research partnership pitch to Mem0 (india@mem0.ai). Mem0 posted two
Backend Engineer roles in India at ₹1 Cr each. This package pitches IndraAstra as an
AI research partner — same budget, more capacity, proven methodology.

Built on multi-month Claude Code research collaboration on SGNNET — a sparse graph
neural network for memory- and energy-efficient deep learning.

## Files (recommended reading order)

| # | File | What it is |
|---|------|------------|
| 1 | `02_cover_letter.md` | Pitch email. Read first. ~300 words. Opens with the ₹1 Cr challenge. |
| 2 | `07_indraastra_pitch.md` | One-pager: what IndraAstra does, engagement models, risk comparison, SGNNET proof. |
| 3 | `01_project_summary.md` | Three-paragraph project summary: result, method, scale. |
| 4 | `03_best_practices.md` | Seven AI-collaboration patterns distilled from the work, each with a concrete project example. |
| 5 | `04_transcript_excerpts.md` | Five curated dialogue excerpts from raw Claude Code session logs showing debugging, design pushback, course-correction, autonomy, empirical validation. |
| 6 | `05_learnings_highlights.md` | Pointers to four representative artifacts in `learnings/` (paper audit, meditation, design notes). |
| 7 | `06_claude_md_snippet.md` | The operating contract — selected sections of the project's living `CLAUDE.md`. |
| 8 | `RAW_TRANSCRIPTS_NOTE.md` | Note on the raw `.jsonl` session files included in the zip. |

## Single-file fallback

If the email accepts one attachment, send `02_cover_letter.md`. The full zip is `mem0_dhiraj_application.zip`.

## Headline numbers

- 95.95% accuracy on Imagenette at 0.20M FLOPs (0.16% of VGG16's FC head).
- 34,976 parameters (0.029% of VGG16 FC).
- 12.7 µs wall-time at B=32 on RTX 5060 Ti — 5.26× faster than the dense baseline.
- Pareto-dominates VGG16 FC on 4 of 5 efficiency dimensions.
- Hundreds of experiments across 6 training slots on 3 machines, fully orchestrated through tmux + a launch wrapper.

Contact: dhiraj.daga@indraastra.in

# Raw Transcripts — Note

This zip includes two raw Claude Code session transcripts in JSONL format:

- `027169ad-7726-4176-bd37-3522610f01dc.jsonl` (~5 MB) — WINNERS audit + CNN distillation track + autonomous slot filling.
- `bdf6d6ae-07e9-4d1a-8f9a-811595c3d455.jsonl` (~2 MB) — Time-series pair-trading sub-project: design pushback, parquet bug, four parallel baselines, MAE context encoder draft.

## What they are

Each `.jsonl` is the complete record of a Claude Code session: user messages, assistant messages (including tool calls and reasoning), and tool results. One JSON object per line. They are point-in-time records — file paths and code references may not match current repo state.

## How to read them

1. **Start with the curated markdown files in this zip, not the JSONL.** They are the highlights; the JSONL is for verification.
2. If you want to verify a specific excerpt from `04_transcript_excerpts.md`, grep the JSONL for a unique phrase from the excerpt:
   ```
   jq -r 'select(.type=="user" or .type=="assistant") | .message.content // .message[0]?.text // empty' <file>.jsonl | grep -A 3 "phrase from excerpt"
   ```
3. The current session (`2c22221c-...`, 127 MB) is not included due to size, but transcripts from it are quoted in `04_transcript_excerpts.md` excerpt (e).

## What was scrubbed

The transcripts were scanned for API keys, OAuth tokens, AWS access keys, GitHub PATs, and bearer tokens. Matches found were:

- "token" — used in the ML sense (mask token, learnable parameter), not credentials.
- "password" — appeared once in the context "either use ssh password or key", not a credential.
- Base64-looking strings starting with "AkIa..." / "aKiA..." — these are Claude's encrypted reasoning signatures (Anthropic-internal, not credentials).

**No real credentials were found.** Nothing was removed. The transcripts are unmodified.

## What is **not** scrubbed (intentionally)

- The 5060ti SSH hostname (`5060ti`) appears throughout — this is a LAN alias, not a public hostname, and is required context for understanding multi-machine slot orchestration.
- Dhiraj's name and email (`dhiraj.daga@indraastra.in`) appear — these match the application.
- Internal paths like `/Volumes/T9/IndraAstra/dhiraj/neuro_graph/...` — these are local to the work machine, not sensitive.

## Reading order

Read the markdown files in this order, then optionally dip into the JSONL for verification:

1. `02_cover_letter.md`
2. `01_project_summary.md`
3. `03_best_practices.md`
4. `04_transcript_excerpts.md`
5. `05_learnings_highlights.md`
6. `06_claude_md_snippet.md`
7. JSONL files (only if you want to verify the excerpts independently)

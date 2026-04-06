# InfraNodus Trial — Instructions

Local knowledge graph analysis tool for SGNNET experiment gap detection.
Reverse-engineered from InfraNodus (Paranyushkin 2019, WWW paper).

---

## Method: Three-Way Parallel Gap Analysis

Run three methods in parallel on the same corpus, then aggregate:

### Step 1: Run all three analyses

```bash
# V1 — Co-occurrence baseline (5s, $0)
python infranodus_trial/v1_baseline/run.py > infranodus_trial/comparison_v1_baseline.txt 2>&1

# V2 — TF-IDF weighted co-occurrence (6s, $0)
python infranodus_trial/run.py --tfidf > infranodus_trial/comparison_v2_tfidf.txt 2>&1

# V3 — Full-context LLM reading (~5min, uses context window)
# Ask Claude to read ALL learnings/*.md + results/train_step*.json and produce:
#   - Topical clusters (8-15 themes)
#   - Structural gaps (pairs that SHOULD connect but don't)
#   - Unexplored combinations (mechanisms never tested together)
#   - Contradictions/tensions
# Save output to: infranodus_trial/comparison_v3_context.txt
```

### Step 2: Aggregate and distill

Cross-reference all three outputs:
- V1/V2 provide **cluster structure** (what terms group together)
- V3 provides **semantic reasoning** (WHY gaps matter, WHAT to do about them)
- Deduplicate: if V2 detects a gap that V3 also found, it's cross-validated
- Discard: V1/V2 gaps involving noise clusters (ops, meta-text, result prose)

Write distilled gaps to `DISTILLED_GAPS.md` with:
- Priority tiers (HIGH blocks next generation, MEDIUM is opportunity, LOW is research)
- For each gap: the problem, what's missing, specific experiment to run
- Cross-method validation table showing which methods found which gaps

### Step 3: Feed back into experiment design

The DISTILLED_GAPS.md directly informs `learnings/EXPERIMENT_QUEUE.md`.

---

## When to Run

- **After every generation** of experiments completes (new results land in results/)
- **After major learnings updates** (new phase learnings files)
- **Before designing compound experiments** (ensure no untested combinations are missed)

---

## CLI Reference

```bash
# Basic analysis
python infranodus_trial/run.py

# With TF-IDF (stronger clusters, recommended)
python infranodus_trial/run.py --tfidf

# Save a named graph for later
python infranodus_trial/run.py --save my-graph

# Load and query a saved graph
python infranodus_trial/run.py --load my-graph
python infranodus_trial/run.py --load my-graph --statements coupling
python infranodus_trial/run.py --load my-graph --statements coupling --radius 2

# List and search saved graphs
python infranodus_trial/run.py --list
python infranodus_trial/run.py --search "inhibition"

# Interactive visualization (HTML + D3.js)
python infranodus_trial/run.py --viz infranodus_trial/sgnnet.html

# LLM features (prints prompt if no API key, calls Claude if ANTHROPIC_API_KEY set)
python infranodus_trial/run.py --questions     # bridging research questions
python infranodus_trial/run.py --overview      # topical summary
python infranodus_trial/run.py --context       # compact graph context for LLM injection

# Tuning
python infranodus_trial/run.py --resolution 1.5  # more/smaller clusters
python infranodus_trial/run.py --window 6        # wider co-occurrence window
python infranodus_trial/run.py --no-results      # learnings only, exclude result JSONs
python infranodus_trial/run.py --no-queue        # exclude EXPERIMENT_QUEUE.md
```

---

## File Structure

```
infranodus_trial/
  engine.py              # Core analysis engine (tokenizer, graph, communities, centrality, gaps)
  run.py                 # CLI runner (loads corpus, runs analysis, all flags)
  store.py               # Graph persistence (save/load/search as JSON)
  viz.py                 # D3.js HTML visualization generator
  ai.py                  # LLM integration (research questions, overview, context)
  trial.py               # Cloud InfraNodus API client (reference only)
  INSTRUCTIONS.md        # This file
  EVALUATION.md          # Qualitative scoring of V1 vs V2 vs V3
  DISTILLED_GAPS.md      # Aggregated gaps influencing next experiments
  v1_baseline/           # Frozen V1 code for reproducible comparison
  graphs/                # Saved graph JSON files
  stopwords/             # Per-graph custom stopword files
  comparison_v1_baseline.txt   # V1 output
  comparison_v2_tfidf.txt      # V2 output
  comparison_v3_context.txt    # V3 output (LLM deep reading)
```

---

## How the Algorithm Works

1. **Tokenize** text: strip markdown, remove stopwords, extract meaningful terms
2. **4-gram sliding window**: pairs within window get weighted edges (adjacent=3, dist2=2, dist3=1)
3. **Optional TF-IDF reweighting**: multiply edges by IDF of both endpoints (downweight common terms)
4. **Louvain community detection**: modularity-based clustering (gamma=1.0)
5. **Betweenness centrality**: identify terms that bridge communities
6. **Structural holes**: community pairs with low/zero cross-edge density = research gaps
7. **Jenks elbow**: natural cutoff for top-tier bridging concepts

Based on: Paranyushkin (2019) "InfraNodus: Generating Insight Using Text Network Analysis" WWW'19

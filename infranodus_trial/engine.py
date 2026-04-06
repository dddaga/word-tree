"""
Text Network Analysis Engine — local implementation of InfraNodus / Textexture algorithm.

Algorithm (Paranyushkin 2011, 2019 WWW paper):
  1. Tokenize + remove stopwords
  2. Sliding 4-gram window → weighted co-occurrence edges
       adjacent (dist=1) → weight 3
       dist=2           → weight 2
       dist=3           → weight 1
  3. Louvain community detection (modularity-based, γ=1)
  4. Betweenness centrality → bridging concepts
  5. Structural holes → community pairs with no/few cross-edges
  6. Jenks elbow cutoff → natural split of top-influence nodes
  7. Diversivity = BC / degree → high-influence with few connections
"""

import re
import json
from collections import defaultdict
from pathlib import Path

import networkx as nx
from networkx.algorithms.community import louvain_communities, modularity

# ─── Stopwords ────────────────────────────────────────────────────────────────

ENGLISH_STOPWORDS = frozenset({
    # Articles, conjunctions, prepositions
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "into", "onto", "upon", "over",
    "under", "about", "after", "before", "between", "among", "across",
    "along", "through", "during", "above", "below", "beyond", "beside",
    "within", "without", "per", "via", "except", "against", "following",
    "around", "near", "since", "until", "plus", "off",
    # Verbs
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might",
    "shall", "can", "get", "got", "make", "made", "take", "taken", "give",
    "given", "use", "used", "using", "show", "shows", "showed", "shown",
    "find", "found", "see", "seen", "note", "noted", "include", "included",
    "apply", "applied", "run", "runs", "running", "ran", "set", "sets",
    "add", "added", "update", "updated", "define", "defined", "compute",
    "computed", "train", "trained", "test", "tested", "run",
    # Pronouns
    "i", "me", "my", "we", "our", "you", "your", "he", "him", "his",
    "she", "her", "it", "its", "they", "them", "their", "this", "that",
    "these", "those", "which", "who", "whom", "what",
    # Adverbs / discourse
    "also", "well", "very", "just", "only", "even", "still", "here",
    "there", "now", "then", "too", "quite", "rather", "so", "yet",
    "how", "when", "where", "why", "not", "no", "nor", "both", "either",
    "neither", "each", "every", "all", "any", "few", "more", "most",
    "other", "some", "such", "than", "however", "although", "though",
    "while", "because", "thus", "hence", "therefore", "moreover",
    "furthermore", "nevertheless", "instead", "otherwise", "meanwhile",
    "indeed", "however", "already", "always", "never", "often",
    "etc", "eg", "ie", "vs",
    # Numbers as words
    "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten",
    # Markdown / paper structure
    "table", "figure", "section", "eq", "see", "ref",
    # Generic ML research words that appear everywhere and carry no signal
    # "accuracy", "result" — keep, they signal outcome clusters
    # "baseline" — keep, anchors reference cluster
    # "gain" — keep, signals improvement cluster
    "epoch", "epochs", "run", "runs", "elapsed", "elapsed_s",
    "true", "false", "none", "null",
    # Generic words that flood from section headers and result prose
    "step", "steps", "document", "section", "source",
    "same", "full", "new", "current", "total",
    "versus", "reached", "achieved", "used", "given",
    "following", "above", "already", "specific",
    "reading", "outperforms", "defaults", "throughout",
    # Ops noise (from LEARNINGS_ops.md)
    "ram", "memory", "old", "avoid", "disk",
    # Purely structural discourse words
    "included", "include", "show", "shows", "showed",
    "config", "note", "notes",
})

# Domain-specific additions: single-letter Greek/Latin param names that
# flood the graph without meaning when tokenized
_SINGLE_LETTER = frozenset("abcdefghijklmnopqrstuvwxyz")


# ─── Tokenizer ────────────────────────────────────────────────────────────────

def tokenize(text: str, min_len: int = 3) -> list[str]:
    """
    Clean and tokenize text into meaningful terms.

    Steps:
      - Strip markdown fences, URLs, headers, formatting symbols
      - Lowercase
      - Extract tokens matching [a-z][a-z0-9_-]*[a-z0-9] (allows underscores, hyphens)
      - Drop stopwords, pure numbers, single-letter tokens, tokens < min_len
    """
    # Strip markdown code fences
    text = re.sub(r"```[\s\S]*?```", " ", text)
    # Strip inline code
    text = re.sub(r"`[^`\n]+`", " ", text)
    # Collapse markdown links to link text
    text = re.sub(r"\[([^\]]*)\]\([^\)]*\)", r"\1", text)
    # Remove URLs
    text = re.sub(r"https?://\S+", " ", text)
    # Remove markdown formatting characters
    text = re.sub(r"[#|*_~^>\\]", " ", text)
    # Remove arrows and punctuation used as separators
    text = re.sub(r"[-=+→←↔•·]+", " ", text)
    # Remove anything in parentheses that's just numbers/symbols like (0.3) or (+8.00pp)
    text = re.sub(r"\([^a-zA-Z]{0,20}\)", " ", text)

    text = text.lower()

    # Extract tokens: start with letter, body may contain letters/digits/underscore/hyphen,
    # must end with letter or digit (no trailing hyphens/underscores)
    tokens = re.findall(r"[a-z][a-z0-9_\-]*[a-z0-9]", text)

    # Also pick up plain 3+ letter words for safety
    tokens += re.findall(r"[a-z]{3,}", text)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    tokens = unique

    result = []
    for tok in tokens:
        if tok in ENGLISH_STOPWORDS:
            continue
        if len(tok) < min_len:
            continue
        if re.fullmatch(r"\d+", tok):
            continue
        if re.fullmatch(r"[a-z]", tok):
            continue
        result.append(tok)

    return result


def tokenize_sequence(text: str, min_len: int = 3) -> list[str]:
    """
    Tokenize text into an ordered sequence (preserving position for sliding window).
    Unlike tokenize(), this keeps duplicate occurrences.
    """
    text = re.sub(r"```[\s\S]*?```", " ", text)
    text = re.sub(r"`[^`\n]+`", " ", text)
    text = re.sub(r"\[([^\]]*)\]\([^\)]*\)", r"\1", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[#|*_~^>\\]", " ", text)
    text = re.sub(r"[-=+→←↔•·]+", " ", text)
    text = re.sub(r"\([^a-zA-Z]{0,20}\)", " ", text)

    text = text.lower()

    tokens = re.findall(r"[a-z][a-z0-9_\-]*[a-z0-9]|[a-z]{3,}", text)

    result = []
    for tok in tokens:
        if tok in ENGLISH_STOPWORDS:
            continue
        if len(tok) < min_len:
            continue
        if re.fullmatch(r"\d+", tok):
            continue
        result.append(tok)

    return result


# ─── Graph builder ────────────────────────────────────────────────────────────

PROXIMITY_WEIGHTS = {1: 3, 2: 2, 3: 1}


def build_graph(tokens: list[str], window_size: int = 4) -> nx.Graph:
    """
    Build weighted co-occurrence graph.

    For each sliding window of `window_size` consecutive tokens, create edges
    between all pairs weighted by their proximity:
      distance 1 (adjacent): +3
      distance 2:             +2
      distance 3:             +1

    Self-loops and same-node pairs are excluded.
    Weights accumulate across the full corpus.
    """
    edge_acc: dict[tuple[str, str], float] = defaultdict(float)

    for i in range(len(tokens)):
        window = tokens[i: i + window_size]
        for j in range(len(window)):
            for k in range(j + 1, len(window)):
                a, b = window[j], window[k]
                if a == b:
                    continue
                dist = k - j  # 1, 2, or 3
                w = PROXIMITY_WEIGHTS.get(dist, 0)
                if w > 0:
                    key = (min(a, b), max(a, b))
                    edge_acc[key] += w

    G = nx.Graph()
    for (a, b), w in edge_acc.items():
        G.add_edge(a, b, weight=w)

    return G


# ─── Community detection ──────────────────────────────────────────────────────

def detect_communities(
    G: nx.Graph, resolution: float = 1.0, seed: int = 42
) -> list[frozenset]:
    """
    Louvain community detection.
    Returns list of node sets sorted by size descending.
    """
    if len(G.nodes) == 0:
        return []
    comms = louvain_communities(G, weight="weight", resolution=resolution, seed=seed)
    return sorted(comms, key=len, reverse=True)


def compute_modularity(G: nx.Graph, communities: list[frozenset]) -> float:
    if not communities or len(G.edges) == 0:
        return 0.0
    return modularity(G, communities, weight="weight")


# ─── Centrality + influence ───────────────────────────────────────────────────

def compute_betweenness(G: nx.Graph) -> dict[str, float]:
    """Normalized betweenness centrality. Higher = bridges more communities."""
    if len(G.nodes) < 3:
        return {n: 0.0 for n in G.nodes}
    return nx.betweenness_centrality(G, weight="weight", normalized=True)


def compute_degree(G: nx.Graph) -> dict[str, float]:
    """Weighted degree (sum of edge weights)."""
    return dict(G.degree(weight="weight"))


def compute_diversivity(
    bc: dict[str, float], degree: dict[str, float]
) -> dict[str, float]:
    """
    Diversivity = BC / degree.
    Identifies nodes that bridge communities efficiently — high influence per connection.
    """
    return {
        n: (bc[n] / degree[n] if degree.get(n, 0) > 0 else 0.0)
        for n in bc
    }


def jenks_elbow(values: list[float]) -> float:
    """
    Jenks natural breaks elbow: find the value at the largest drop in sorted list.
    Everything >= this cutoff is considered a "top node".
    """
    if len(values) < 3:
        return 0.0
    sv = sorted(values, reverse=True)
    diffs = [sv[i] - sv[i + 1] for i in range(len(sv) - 1)]
    elbow_idx = diffs.index(max(diffs))
    return sv[elbow_idx]


# ─── Structural holes ─────────────────────────────────────────────────────────

def find_structural_holes(
    G: nx.Graph,
    communities: list[frozenset],
    top_n: int = 20,
) -> list[dict]:
    """
    Find community pairs with no or few cross-edges.
    Structural holes = places where a new connection would bridge isolated discourse clusters.

    Returns list of gap dicts sorted by density ascending (lowest = biggest gap).
    """
    gaps = []
    for i in range(len(communities)):
        for j in range(i + 1, len(communities)):
            ca, cb = communities[i], communities[j]
            # Ignore tiny communities (likely noise)
            if len(ca) < 2 or len(cb) < 2:
                continue

            cross_weight = sum(
                G[a][b]["weight"]
                for a in ca
                for b in cb
                if G.has_edge(a, b)
            )
            possible = len(ca) * len(cb)
            density = cross_weight / possible if possible > 0 else 0.0

            gaps.append({
                "community_a": i,
                "community_b": j,
                "size_a": len(ca),
                "size_b": len(cb),
                "cross_weight": cross_weight,
                "possible": possible,
                "density": density,
            })

    # Sort: no-connection gaps first, then by ascending density
    gaps.sort(key=lambda g: (g["density"], -(g["size_a"] * g["size_b"])))
    return gaps[:top_n]


# ─── Community labeling ───────────────────────────────────────────────────────

def label_community(
    community: frozenset,
    bc: dict[str, float],
    degree: dict[str, float],
    top_n: int = 6,
) -> str:
    """
    Label a community by its most central members (betweenness + degree combined).
    """
    def score(n: str) -> float:
        return bc.get(n, 0) * 0.7 + (degree.get(n, 0) / (max(degree.values()) + 1e-9)) * 0.3

    ranked = sorted(community, key=score, reverse=True)
    return ", ".join(ranked[:top_n])


# ─── Full analysis ────────────────────────────────────────────────────────────

class TextNetwork:
    """
    Full text network analysis pipeline, mirroring InfraNodus/Textexture.

    Usage:
        tn = TextNetwork.from_text(corpus_text)
        tn.print_report()
    """

    def __init__(
        self,
        G: nx.Graph,
        communities: list[frozenset],
        bc: dict[str, float],
        degree: dict[str, float],
        mod: float,
    ):
        self.G = G
        self.communities = communities
        self.bc = bc
        self.degree = degree
        self.mod = mod
        self.diversivity = compute_diversivity(bc, degree)

        # Build reverse map: node → community index
        self.node_community: dict[str, int] = {}
        for idx, comm in enumerate(communities):
            for node in comm:
                self.node_community[node] = idx

        # Community labels
        self.community_labels: list[str] = [
            label_community(c, bc, degree) for c in communities
        ]

        # Structural holes
        self.holes = find_structural_holes(G, communities)

    @classmethod
    def from_text(cls, text: str, window_size: int = 4, resolution: float = 1.0) -> "TextNetwork":
        tokens = tokenize_sequence(text)
        G = build_graph(tokens, window_size=window_size)

        # Remove isolates (nodes with no edges) — they don't contribute
        isolates = list(nx.isolates(G))
        G.remove_nodes_from(isolates)

        communities = detect_communities(G, resolution=resolution)
        bc = compute_betweenness(G)
        degree = compute_degree(G)
        mod = compute_modularity(G, communities)
        return cls(G, communities, bc, degree, mod)

    def top_nodes(self, n: int = 25, by: str = "betweenness") -> list[tuple[str, float]]:
        """Return top N nodes by betweenness, degree, or diversivity."""
        source = {
            "betweenness": self.bc,
            "degree": self.degree,
            "diversivity": self.diversivity,
        }.get(by, self.bc)
        ranked = sorted(source.items(), key=lambda x: x[1], reverse=True)
        return ranked[:n]

    def top_nodes_elbow(self, by: str = "betweenness") -> list[tuple[str, float]]:
        """Return nodes above Jenks elbow cutoff."""
        source = {
            "betweenness": self.bc,
            "degree": self.degree,
            "diversivity": self.diversivity,
        }.get(by, self.bc)
        cutoff = jenks_elbow(list(source.values()))
        ranked = sorted(
            [(n, v) for n, v in source.items() if v >= cutoff],
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked

    def community_weight(self, idx: int) -> float:
        """Total internal edge weight of a community — proxy for its 'size' in discourse."""
        comm = self.communities[idx]
        return sum(
            self.G[a][b]["weight"]
            for a in comm
            for b in comm
            if self.G.has_edge(a, b)
        )

    def print_report(self, top_n: int = 20, show_holes: int = 15):
        """Print a full structured analysis report."""
        W = 65

        def header(title: str):
            print(f"\n{'─' * 4} {title} {'─' * max(0, W - len(title) - 6)}")

        print(f"\n{'=' * W}")
        print(f"  SGNNET TEXT NETWORK ANALYSIS")
        print(f"{'=' * W}")

        # ── Stats
        header("GRAPH STATS")
        total_weight = sum(d["weight"] for _, _, d in self.G.edges(data=True))
        print(f"  nodes:       {self.G.number_of_nodes()}")
        print(f"  edges:       {self.G.number_of_edges()}")
        print(f"  total_weight:{total_weight:.0f}")
        print(f"  communities: {len(self.communities)}")
        print(f"  modularity:  {self.mod:.4f}  "
              f"({'strong' if self.mod > 0.4 else 'moderate' if self.mod > 0.2 else 'weak'} structure)")

        # ── Topical clusters
        header("TOPICAL CLUSTERS")
        total_w = sum(self.community_weight(i) for i in range(len(self.communities)))
        for i, comm in enumerate(self.communities):
            if len(comm) < 2:
                continue
            cw = self.community_weight(i)
            pct = 100 * cw / total_w if total_w > 0 else 0
            label = self.community_labels[i]
            print(f"\n  [{i}] {pct:.1f}%  size={len(comm)}")
            print(f"       Top terms: {label}")

        # ── Structural holes / gaps
        header(f"STRUCTURAL GAPS  (top {show_holes} community pairs with low connectivity)")
        zero_gaps = [h for h in self.holes if h["cross_weight"] == 0]
        weak_gaps = [h for h in self.holes if h["cross_weight"] > 0]

        if zero_gaps:
            print(f"\n  ZERO-CONNECTION gaps (true structural holes):")
            for h in zero_gaps[:show_holes]:
                la = self.community_labels[h["community_a"]]
                lb = self.community_labels[h["community_b"]]
                print(f"    ·  [{h['community_a']}] {la[:40]}")
                print(f"       ↔  [{h['community_b']}] {lb[:40]}")

        if weak_gaps:
            print(f"\n  WEAK-CONNECTION gaps (low cross-density):")
            for h in weak_gaps[:min(8, show_holes)]:
                la = self.community_labels[h["community_a"]]
                lb = self.community_labels[h["community_b"]]
                dens = h["density"]
                print(f"    ·  [{h['community_a']}] {la[:35]}  ↔  [{h['community_b']}] {lb[:35]}  "
                      f"(density={dens:.3f})")

        # ── Top bridging concepts
        header(f"TOP BRIDGING CONCEPTS  [betweenness centrality — elbow cutoff]")
        top = self.top_nodes_elbow(by="betweenness")
        if len(top) < 5:
            # Elbow too aggressive (single dominant node) — fall back to top N
            top = self.top_nodes(top_n, by="betweenness")
        for node, score in top[:top_n]:
            comm_idx = self.node_community.get(node, -1)
            print(f"  · {node:<30} BC={score:.4f}  cluster={comm_idx}")

        # ── Diversivity (high BC, low degree = efficient bridges)
        # Min degree threshold to filter out rare words that appear only once
        min_deg = max(50, sorted(self.degree.values(), reverse=True)[min(50, len(self.degree) - 1)])
        header("DIVERSIVITY  [betweenness / degree — efficient bridges]")
        div_candidates = sorted(
            [(n, v) for n, v in self.diversivity.items() if self.degree.get(n, 0) >= min_deg],
            key=lambda x: x[1], reverse=True
        )
        div_top = div_candidates[:top_n // 2]
        for node, score in div_top:
            deg = self.degree.get(node, 0)
            bc_val = self.bc.get(node, 0)
            comm_idx = self.node_community.get(node, -1)
            print(f"  · {node:<30} div={score:.5f}  BC={bc_val:.4f}  deg={deg:.0f}  cluster={comm_idx}")

        print()

    # ─── Serialization ────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize the full TextNetwork to a JSON-compatible dict."""
        return {
            "stats": {
                "nodes": self.G.number_of_nodes(),
                "edges": self.G.number_of_edges(),
                "communities": len(self.communities),
                "modularity": round(self.mod, 4),
            },
            "edges": [
                {"a": a, "b": b, "weight": round(d["weight"], 2)}
                for a, b, d in self.G.edges(data=True)
            ],
            "communities": [
                {
                    "id": i,
                    "label": self.community_labels[i],
                    "members": sorted(c),
                }
                for i, c in enumerate(self.communities)
            ],
            "node_metrics": {
                node: {
                    "betweenness": round(self.bc.get(node, 0), 5),
                    "degree": round(self.degree.get(node, 0), 1),
                    "diversivity": round(self.diversivity.get(node, 0), 6),
                    "community": self.node_community.get(node, -1),
                }
                for node in self.G.nodes()
            },
            "structural_holes": [
                {
                    "community_a": h["community_a"],
                    "label_a": self.community_labels[h["community_a"]],
                    "community_b": h["community_b"],
                    "label_b": self.community_labels[h["community_b"]],
                    "density": round(h["density"], 5),
                    "cross_weight": h["cross_weight"],
                }
                for h in self.holes
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TextNetwork":
        """Reconstruct a TextNetwork from a serialized dict."""
        G = nx.Graph()
        for e in data["edges"]:
            G.add_edge(e["a"], e["b"], weight=e["weight"])

        communities = [
            frozenset(c["members"]) for c in data["communities"]
        ]

        bc = {}
        degree = {}
        for node, m in data.get("node_metrics", {}).items():
            bc[node] = m["betweenness"]
            degree[node] = m["degree"]

        mod = data.get("stats", {}).get("modularity", 0.0)
        return cls(G, communities, bc, degree, mod)

    # ─── Statement retrieval ──────────────────────────────────────────────

    def get_statements(self, node: str, radius: int = 1) -> list[dict]:
        """
        Get all edges/neighbors within `radius` hops of a node.
        Returns list of relation dicts: {source, target, weight, community_source, community_target}.
        """
        if node not in self.G:
            return []

        visited = {node}
        frontier = {node}
        relations = []

        for _ in range(radius):
            next_frontier = set()
            for n in frontier:
                for neighbor in self.G.neighbors(n):
                    w = self.G[n][neighbor]["weight"]
                    relations.append({
                        "source": n,
                        "target": neighbor,
                        "weight": round(w, 2),
                        "community_source": self.node_community.get(n, -1),
                        "community_target": self.node_community.get(neighbor, -1),
                    })
                    if neighbor not in visited:
                        next_frontier.add(neighbor)
                        visited.add(neighbor)
            frontier = next_frontier

        return relations


# ─── TF-IDF edge reweighting ─────────────────────────────────────────────────

def tfidf_reweight(G: nx.Graph, documents: list[str]) -> nx.Graph:
    """
    Reweight graph edges by TF-IDF scores of endpoint nodes.

    Each document is a text chunk (e.g., one learnings file). IDF is computed
    across all documents. Edge weight is multiplied by `idf[a] * idf[b]`,
    so edges between globally common terms are downweighted.

    Returns the same graph (mutated in place).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Use same tokenizer as the engine for consistency
    vectorizer = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"[a-z][a-z0-9_\-]*[a-z0-9]|[a-z]{3,}",
        lowercase=True,
    )
    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()

    # Build IDF lookup: term → IDF score
    idf_lookup = {}
    for idx, name in enumerate(feature_names):
        idf_lookup[name] = vectorizer.idf_[idx]

    # Default IDF for terms not in any document (shouldn't happen, but safety)
    default_idf = max(vectorizer.idf_) if len(vectorizer.idf_) > 0 else 1.0

    for a, b, d in G.edges(data=True):
        idf_a = idf_lookup.get(a, default_idf)
        idf_b = idf_lookup.get(b, default_idf)
        # Normalize: divide by max possible IDF product to keep weights in same scale
        d["weight"] *= (idf_a * idf_b) / (default_idf ** 2)

    return G


# ─── Custom stopwords ────────────────────────────────────────────────────────

def load_custom_stopwords(path: str) -> frozenset:
    """
    Load custom stopword additions/removals from a JSON file.
    Format: {"add": ["term1", "term2"], "remove": ["term3"]}
    Returns the merged stopword set.
    """
    p = Path(path)
    if not p.exists():
        return ENGLISH_STOPWORDS

    data = json.loads(p.read_text())
    result = set(ENGLISH_STOPWORDS)
    for term in data.get("add", []):
        result.add(term.lower())
    for term in data.get("remove", []):
        result.discard(term.lower())
    return frozenset(result)

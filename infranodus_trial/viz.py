"""
Interactive knowledge graph visualization — self-contained HTML with D3.js.

Generates a single HTML file with force-directed graph layout.
Nodes colored by community, sized by betweenness centrality.
"""

import json
import math
from pathlib import Path


def generate_viz(
    tn,
    output_path: str,
    max_nodes: int = 300,
    min_nodes_per_community: int = 10,
):
    """
    Generate an interactive HTML visualization of the text network.

    Args:
        tn: TextNetwork instance
        output_path: Where to write the HTML file
        max_nodes: Maximum nodes to render (top by BC, with community coverage)
        min_nodes_per_community: Ensure at least this many nodes per community
    """
    # Select nodes: top by BC + ensure community coverage
    selected = _select_nodes(tn, max_nodes, min_nodes_per_community)

    # Build D3 data
    nodes_data = []
    node_set = set(selected)
    for node in selected:
        nodes_data.append({
            "id": node,
            "bc": round(tn.bc.get(node, 0), 5),
            "degree": round(tn.degree.get(node, 0), 1),
            "community": tn.node_community.get(node, -1),
            "label": tn.community_labels[tn.node_community.get(node, 0)][:40],
        })

    edges_data = []
    for a, b, d in tn.G.edges(data=True):
        if a in node_set and b in node_set:
            edges_data.append({
                "source": a,
                "target": b,
                "weight": round(d["weight"], 1),
            })

    # Community summary for legend
    communities = []
    for i, comm in enumerate(tn.communities):
        if len(comm) < 2:
            continue
        communities.append({
            "id": i,
            "label": tn.community_labels[i],
            "size": len(comm),
        })

    # Structural holes for display
    holes = []
    for h in tn.holes[:10]:
        holes.append({
            "a": h["community_a"],
            "b": h["community_b"],
            "label_a": tn.community_labels[h["community_a"]][:30],
            "label_b": tn.community_labels[h["community_b"]][:30],
            "density": round(h["density"], 4),
        })

    graph_json = json.dumps({
        "nodes": nodes_data,
        "links": edges_data,
        "communities": communities,
        "holes": holes,
        "stats": {
            "total_nodes": tn.G.number_of_nodes(),
            "total_edges": tn.G.number_of_edges(),
            "shown_nodes": len(nodes_data),
            "shown_edges": len(edges_data),
            "modularity": round(tn.mod, 4),
        },
    })

    html = _HTML_TEMPLATE.replace("/*GRAPH_DATA*/", graph_json)
    Path(output_path).write_text(html)
    print(f"Visualization saved → {output_path}")


def _select_nodes(tn, max_nodes: int, min_per_comm: int) -> list[str]:
    """Select top nodes by BC while ensuring community representation."""
    # First pass: ensure min_per_comm nodes from each community
    selected = set()
    for i, comm in enumerate(tn.communities):
        if len(comm) < 2:
            continue
        ranked = sorted(comm, key=lambda n: tn.bc.get(n, 0), reverse=True)
        for n in ranked[:min_per_comm]:
            selected.add(n)

    # Fill remaining slots with top BC nodes
    remaining = max_nodes - len(selected)
    if remaining > 0:
        all_by_bc = sorted(tn.bc.items(), key=lambda x: x[1], reverse=True)
        for node, _ in all_by_bc:
            if node not in selected:
                selected.add(node)
                remaining -= 1
                if remaining <= 0:
                    break

    return list(selected)


_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>SGNNET Knowledge Graph</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0a0a0a; color: #e0e0e0; display: flex; height: 100vh; overflow: hidden; }
#sidebar { width: 300px; background: #141414; border-right: 1px solid #333; padding: 16px; overflow-y: auto; flex-shrink: 0; }
#graph-container { flex: 1; position: relative; }
svg { width: 100%; height: 100%; }
h2 { font-size: 14px; color: #888; text-transform: uppercase; letter-spacing: 1px; margin: 16px 0 8px; }
h2:first-child { margin-top: 0; }
.stat { font-size: 13px; margin: 2px 0; }
.stat span { color: #4fc3f7; }
.cluster { font-size: 12px; margin: 4px 0; padding: 4px 8px; border-radius: 4px; cursor: pointer; }
.cluster:hover { background: #222; }
.cluster-dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; }
.hole { font-size: 11px; margin: 3px 0; padding: 3px 6px; background: #1a1a2e; border-radius: 3px; color: #b0b0b0; }
.hole em { color: #ff7043; font-style: normal; }
#tooltip { position: absolute; background: rgba(20,20,20,0.95); border: 1px solid #444; padding: 8px 12px; border-radius: 6px; font-size: 12px; pointer-events: none; display: none; z-index: 10; }
#tooltip .tt-name { font-size: 14px; font-weight: 600; color: #fff; }
#tooltip .tt-detail { color: #aaa; margin-top: 2px; }
.link { stroke-opacity: 0.15; }
.link.highlighted { stroke-opacity: 0.6; stroke: #fff !important; }
.node text { font-size: 10px; fill: #ccc; pointer-events: none; }
.node text.highlighted { fill: #fff; font-weight: 600; }
.node circle.dimmed { opacity: 0.1; }
.node text.dimmed { opacity: 0.1; }
.link.dimmed { stroke-opacity: 0.02; }
</style>
</head>
<body>
<div id="sidebar">
  <h2>Graph Stats</h2>
  <div id="stats"></div>
  <h2>Clusters</h2>
  <div id="clusters"></div>
  <h2>Structural Gaps</h2>
  <div id="holes"></div>
</div>
<div id="graph-container">
  <svg id="graph"></svg>
  <div id="tooltip"></div>
</div>
<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
const data = /*GRAPH_DATA*/;

const colors = d3.scaleOrdinal(d3.schemeTableau10);
const bcMax = d3.max(data.nodes, d => d.bc) || 0.01;
const wMax = d3.max(data.links, d => d.weight) || 1;
const nodeRadius = d => 4 + Math.sqrt(d.bc / bcMax) * 16;

// Stats
const statsEl = document.getElementById('stats');
statsEl.innerHTML = `
  <div class="stat">Nodes: <span>${data.stats.total_nodes}</span> (showing ${data.stats.shown_nodes})</div>
  <div class="stat">Edges: <span>${data.stats.total_edges}</span> (showing ${data.stats.shown_edges})</div>
  <div class="stat">Modularity: <span>${data.stats.modularity}</span></div>
`;

// Clusters
const clustersEl = document.getElementById('clusters');
data.communities.forEach(c => {
  const div = document.createElement('div');
  div.className = 'cluster';
  div.innerHTML = `<span class="cluster-dot" style="background:${colors(c.id)}"></span>[${c.id}] ${c.label.substring(0, 35)} <span style="color:#666">(${c.size})</span>`;
  div.onclick = () => highlightCommunity(c.id);
  clustersEl.appendChild(div);
});

// Holes
const holesEl = document.getElementById('holes');
data.holes.forEach(h => {
  const div = document.createElement('div');
  div.className = 'hole';
  div.innerHTML = `<em>[${h.a}]</em> ${h.label_a} <em>↔</em> <em>[${h.b}]</em> ${h.label_b}`;
  holesEl.appendChild(div);
});

// Graph
const svg = d3.select('#graph');
const width = document.getElementById('graph-container').clientWidth;
const height = document.getElementById('graph-container').clientHeight;

const g = svg.append('g');

const zoom = d3.zoom().scaleExtent([0.1, 8]).on('zoom', e => g.attr('transform', e.transform));
svg.call(zoom);

const simulation = d3.forceSimulation(data.nodes)
  .force('link', d3.forceLink(data.links).id(d => d.id).distance(60).strength(d => Math.min(d.weight / wMax * 0.3, 0.2)))
  .force('charge', d3.forceManyBody().strength(-80))
  .force('center', d3.forceCenter(width / 2, height / 2))
  .force('collision', d3.forceCollide().radius(d => nodeRadius(d) + 2));

const link = g.append('g').selectAll('line')
  .data(data.links).join('line')
  .attr('class', 'link')
  .attr('stroke', '#555')
  .attr('stroke-width', d => Math.max(0.5, Math.sqrt(d.weight / wMax) * 3));

const node = g.append('g').selectAll('g')
  .data(data.nodes).join('g')
  .attr('class', 'node')
  .call(d3.drag().on('start', dragStarted).on('drag', dragged).on('end', dragEnded));

node.append('circle')
  .attr('r', d => nodeRadius(d))
  .attr('fill', d => colors(d.community))
  .attr('stroke', '#000')
  .attr('stroke-width', 0.5);

node.append('text')
  .attr('dx', d => nodeRadius(d) + 3)
  .attr('dy', '0.35em')
  .text(d => d.id)
  .style('font-size', d => d.bc > bcMax * 0.3 ? '12px' : '9px');

// Tooltip
const tooltip = document.getElementById('tooltip');
node.on('mouseover', (e, d) => {
  tooltip.style.display = 'block';
  tooltip.innerHTML = `<div class="tt-name">${d.id}</div>
    <div class="tt-detail">BC: ${d.bc.toFixed(4)}</div>
    <div class="tt-detail">Degree: ${d.degree}</div>
    <div class="tt-detail">Cluster: [${d.community}] ${d.label}</div>`;
}).on('mousemove', e => {
  tooltip.style.left = (e.pageX + 12) + 'px';
  tooltip.style.top = (e.pageY - 12) + 'px';
}).on('mouseout', () => { tooltip.style.display = 'none'; });

// Click to highlight neighborhood
node.on('click', (e, d) => highlightNode(d));

function highlightNode(d) {
  const neighbors = new Set();
  data.links.forEach(l => {
    const s = typeof l.source === 'object' ? l.source.id : l.source;
    const t = typeof l.target === 'object' ? l.target.id : l.target;
    if (s === d.id) neighbors.add(t);
    if (t === d.id) neighbors.add(s);
  });
  neighbors.add(d.id);

  node.select('circle').classed('dimmed', n => !neighbors.has(n.id));
  node.select('text').classed('dimmed', n => !neighbors.has(n.id)).classed('highlighted', n => neighbors.has(n.id));
  link.classed('dimmed', l => {
    const s = typeof l.source === 'object' ? l.source.id : l.source;
    const t = typeof l.target === 'object' ? l.target.id : l.target;
    return !neighbors.has(s) || !neighbors.has(t);
  }).classed('highlighted', l => {
    const s = typeof l.source === 'object' ? l.source.id : l.source;
    const t = typeof l.target === 'object' ? l.target.id : l.target;
    return (s === d.id || t === d.id);
  });
}

function highlightCommunity(cid) {
  const members = new Set(data.nodes.filter(n => n.community === cid).map(n => n.id));
  node.select('circle').classed('dimmed', n => !members.has(n.id));
  node.select('text').classed('dimmed', n => !members.has(n.id)).classed('highlighted', n => members.has(n.id));
  link.classed('dimmed', l => {
    const s = typeof l.source === 'object' ? l.source.id : l.source;
    const t = typeof l.target === 'object' ? l.target.id : l.target;
    return !members.has(s) && !members.has(t);
  }).classed('highlighted', () => false);
}

// Double-click to reset
svg.on('dblclick.reset', () => {
  node.select('circle').classed('dimmed', false);
  node.select('text').classed('dimmed', false).classed('highlighted', false);
  link.classed('dimmed', false).classed('highlighted', false);
});

simulation.on('tick', () => {
  link.attr('x1', d => d.source.x).attr('y1', d => d.source.y)
      .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
  node.attr('transform', d => `translate(${d.x},${d.y})`);
});

function dragStarted(e, d) { if (!e.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; }
function dragged(e, d) { d.fx = e.x; d.fy = e.y; }
function dragEnded(e, d) { if (!e.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; }
</script>
</body>
</html>"""

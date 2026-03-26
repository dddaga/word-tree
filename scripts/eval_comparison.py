"""Compare all Phase 4 experimental stages and produce wave_comparison outputs.

Loads Stage A, Exp 1, Exp 2 results and VGG16 baseline, then generates:
- results/wave_comparison.json (structured side-by-side metrics)
- results/wave_comparison.md (human-readable comparison table)
"""

import json
import os

# -- Constants ---------------------------------------------------------------

CLASS_NAMES = [
    "tench", "english_springer", "cassette_player", "chain_saw", "church",
    "french_horn", "garbage_truck", "gas_pump", "golf_ball", "parachute",
]

# Baseline uses different casing for class names
BASELINE_CLASS_MAP = {
    "tench": "tench",
    "english_springer": "English springer",
    "cassette_player": "cassette player",
    "chain_saw": "chain saw",
    "church": "church",
    "french_horn": "French horn",
    "garbage_truck": "garbage truck",
    "gas_pump": "gas pump",
    "golf_ball": "golf ball",
    "parachute": "parachute",
}


def main():
    """Load results, build comparison JSON and markdown."""
    baseline = _load_json("results/baseline_vgg16.json")
    stage_a = _load_json("results/stageA_full.json")
    exp1 = _load_json("results/exp1_full.json")
    exp2 = _load_json("results/exp2_full.json")

    phase3_config = _load_optional("results/sgnnet_config.json")

    comparison = _build_comparison(baseline, stage_a, exp1, exp2, phase3_config)
    _write_json(comparison, "results/wave_comparison.json")

    md_text = _build_markdown(baseline, stage_a, exp1, exp2, phase3_config)
    _write_text(md_text, "results/wave_comparison.md")

    print("Saved results/wave_comparison.json and results/wave_comparison.md")


# -- Data loading ------------------------------------------------------------

def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _load_optional(path):
    if os.path.exists(path):
        return _load_json(path)
    return None


# -- Comparison JSON ---------------------------------------------------------

def _build_comparison(baseline, stage_a, exp1, exp2, phase3_config):
    comparison = {
        "vgg16_baseline": {
            "model": "VGG16 FC (frozen)",
            "params": baseline["fc_params"],
            "top1_accuracy": baseline["top1_accuracy"],
            "mAP": baseline["mAP"],
        },
        "stageA": _stage_entry("Stage A (static binary C, no dynamic)", stage_a),
        "exp1": _stage_entry("Exp 1 (spatial path-length phase)", exp1),
        "exp2": _stage_entry("Exp 2 (spatial phase + W_phase)", exp2),
        "deltas": {
            "exp1_vs_stageA": _compute_delta(exp1, stage_a),
            "exp2_vs_exp1": _compute_delta(exp2, exp1),
        },
    }
    if phase3_config:
        comparison["phase3_reference"] = {
            "model": "SGNNET v1 (Phase 3)",
            "params": phase3_config["total_params"],
            "percent_of_vgg16_fc": phase3_config["percent_of_vgg16_fc"],
        }
    return comparison


def _stage_entry(model_name, data):
    return {
        "model": model_name,
        "params": data["params"],
        "percent_of_vgg16_fc": data["percent_of_vgg16_fc"],
        "top1_accuracy": data["top1_accuracy"],
        "mAP": data["mAP"],
        "per_class": data["per_class"],
        "hyperparams": data["hyperparams"],
    }


def _compute_delta(newer, older):
    return {
        "top1_delta": newer["top1_accuracy"] - older["top1_accuracy"],
        "mAP_delta": newer["mAP"] - older["mAP"],
        "param_delta": newer["params"] - older["params"],
    }


# -- Markdown report ---------------------------------------------------------

def _build_markdown(baseline, stage_a, exp1, exp2, phase3_config):
    lines = _aggregate_table(baseline, stage_a, exp1, exp2, phase3_config)
    lines += _contribution_analysis(stage_a, exp1, exp2)
    lines += _per_class_table(baseline, stage_a, exp1, exp2)
    lines += _hyperparams_section(stage_a, exp1, exp2)
    return "\n".join(lines) + "\n"


def _aggregate_table(baseline, stage_a, exp1, exp2, phase3_config):
    lines = [
        "# Wave Architecture Comparison",
        "",
        "## Aggregate Metrics",
        "",
        "| Model | Params | % VGG FC | Top-1 | mAP | Key addition |",
        "|-------|--------|----------|-------|-----|--------------|",
        (f"| VGG16 FC (frozen) | {baseline['fc_params']:,} | 100% "
         f"| {baseline['top1_accuracy']:.4f} | {baseline['mAP']:.4f} "
         f"| Reference |"),
    ]
    if phase3_config:
        lines.append(
            f"| SGNNET v1 (Phase 3) | {phase3_config['total_params']:,} "
            f"| {phase3_config['percent_of_vgg16_fc']:.2f}% "
            f"| -- | -- | Learned C, amplitude routing |"
        )
    lines.extend([
        (f"| Stage A (static) | {stage_a['params']:,} "
         f"| {stage_a['percent_of_vgg16_fc']}% "
         f"| {stage_a['top1_accuracy']:.4f} | {stage_a['mAP']:.4f} "
         f"| Binary C, no dynamic |"),
        (f"| Exp 1 (spatial phase) | {exp1['params']:,} "
         f"| {exp1['percent_of_vgg16_fc']}% "
         f"| {exp1['top1_accuracy']:.4f} | {exp1['mAP']:.4f} "
         f"| + proximity w/ path-length phase |"),
        (f"| Exp 2 (spatial + W_phase) | {exp2['params']:,} "
         f"| {exp2['percent_of_vgg16_fc']}% "
         f"| {exp2['top1_accuracy']:.4f} | {exp2['mAP']:.4f} "
         f"| + learned phase operator |"),
    ])
    return lines


def _contribution_analysis(stage_a, exp1, exp2):
    e1_top1 = exp1["top1_accuracy"] - stage_a["top1_accuracy"]
    e1_map = exp1["mAP"] - stage_a["mAP"]
    e2_top1 = exp2["top1_accuracy"] - exp1["top1_accuracy"]
    e2_map = exp2["mAP"] - exp1["mAP"]
    e1_verb = "improves" if exp1["mAP"] > stage_a["mAP"] else "does not improve"
    e2_verb = "improves" if exp2["mAP"] > exp1["mAP"] else "does not improve"
    return [
        "",
        "## Contribution Analysis",
        "",
        (f"- **Exp 1 vs Stage A:** top1 delta = {e1_top1:+.4f}, "
         f"mAP delta = {e1_map:+.4f}"),
        f"  - Proximity routing with spatial phase {e1_verb} over static wiring",
        (f"- **Exp 2 vs Exp 1:** top1 delta = {e2_top1:+.4f}, "
         f"mAP delta = {e2_map:+.4f}"),
        f"  - W_phase adds {exp2['params'] - exp1['params']} parameters",
        (f"  - Learned phase operator {e2_verb} over geometry-only phase"),
    ]


def _per_class_table(baseline, stage_a, exp1, exp2):
    lines = [
        "",
        "## Per-Class Comparison",
        "",
        "| Class | VGG16 Acc | Stage A Acc | Exp1 Acc | Exp2 Acc "
        "| Exp1 AP | Exp2 AP |",
        "|-------|-----------|-------------|----------|----------"
        "|---------|---------|",
    ]
    for cls in CLASS_NAMES:
        bl_key = BASELINE_CLASS_MAP[cls]
        vgg_acc = baseline.get("per_class", {}).get(bl_key, {}).get("accuracy", "--")
        sa_acc = stage_a["per_class"].get(cls, {}).get("accuracy", "--")
        e1_acc = exp1["per_class"].get(cls, {}).get("accuracy", "--")
        e2_acc = exp2["per_class"].get(cls, {}).get("accuracy", "--")
        e1_ap = exp1["per_class"].get(cls, {}).get("AP", "--")
        e2_ap = exp2["per_class"].get(cls, {}).get("AP", "--")
        vals = [vgg_acc, sa_acc, e1_acc, e2_acc, e1_ap, e2_ap]
        formatted = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in vals]
        lines.append(
            f"| {cls} | {' | '.join(formatted)} |"
        )
    return lines


def _hyperparams_section(stage_a, exp1, exp2):
    lines = ["", "## Hyperparameters Used", ""]
    for name, data in [("Stage A", stage_a), ("Exp 1", exp1), ("Exp 2", exp2)]:
        hp = data["hyperparams"]
        lines.append(
            f"**{name}:** K={hp['K']}, N_hidden={hp['N_hidden']}, "
            f"lr_Wpos={hp['lr_Wpos']:.2e}, "
            f"lambda_safety={hp['lambda_safety']:.3f}, "
            f"batch_size={hp['batch_size']}"
        )
    return lines


# -- File I/O ----------------------------------------------------------------

def _write_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _write_text(text, path):
    with open(path, "w") as f:
        f.write(text)


if __name__ == "__main__":
    main()

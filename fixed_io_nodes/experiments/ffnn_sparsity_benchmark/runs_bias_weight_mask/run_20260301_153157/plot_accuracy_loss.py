import matplotlib
matplotlib.use("Agg")
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = "results.csv"
df = pd.read_csv(CSV_PATH)

sparsity_levels = sorted(df["sparsity"].unique())
cmap = plt.cm.viridis
colors = [cmap(i / (len(sparsity_levels) - 1)) for i in range(len(sparsity_levels))]

# ── Plot 1: Separate Accuracy vs Hidden Size subplot for each sparsity ──
n_sp = len(sparsity_levels)
ncols = 4
nrows = int(np.ceil(n_sp / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharex=True, sharey=True)
axes_flat = axes.flatten()

for idx, (sp, color) in enumerate(zip(sparsity_levels, colors)):
    ax = axes_flat[idx]
    sub = df[df["sparsity"] == sp].sort_values("hidden_size")
    ax.plot(sub["hidden_size"], sub["accuracy"], color=color, linewidth=1.2, alpha=0.85)
    ax.set_title(f"Sparsity = {sp:.1f}", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    if idx % ncols == 0:
        ax.set_ylabel("Accuracy", fontsize=10)
    if idx >= (nrows - 1) * ncols:
        ax.set_xlabel("Hidden Size", fontsize=10)

for idx in range(n_sp, len(axes_flat)):
    axes_flat[idx].set_visible(False)

fig.suptitle("Accuracy vs Hidden Size (separate per Sparsity level)", fontsize=15, y=1.01)
fig.tight_layout()
fig.savefig("accuracy_vs_hidden_size_per_sparsity.png", dpi=200, bbox_inches="tight")
print("Saved accuracy_vs_hidden_size_per_sparsity.png")

# ── Plot 2: Separate Val Loss vs Hidden Size subplot for each sparsity ──
fig2, axes2 = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharex=True, sharey=True)
axes2_flat = axes2.flatten()

for idx, (sp, color) in enumerate(zip(sparsity_levels, colors)):
    ax2 = axes2_flat[idx]
    sub = df[df["sparsity"] == sp].sort_values("hidden_size")
    ax2.plot(sub["hidden_size"], sub["best_val_loss"], color=color, linewidth=1.2, alpha=0.85)
    ax2.set_title(f"Sparsity = {sp:.1f}", fontsize=11)
    ax2.grid(True, alpha=0.3)
    if idx % ncols == 0:
        ax2.set_ylabel("Best Val Loss", fontsize=10)
    if idx >= (nrows - 1) * ncols:
        ax2.set_xlabel("Hidden Size", fontsize=10)

for idx in range(n_sp, len(axes2_flat)):
    axes2_flat[idx].set_visible(False)

fig2.suptitle("Validation Loss vs Hidden Size (separate per Sparsity level)", fontsize=15, y=1.01)
fig2.tight_layout()
fig2.savefig("loss_vs_hidden_size_per_sparsity.png", dpi=200, bbox_inches="tight")
print("Saved loss_vs_hidden_size_per_sparsity.png")

# ── Plot 3: Accuracy variance across sparsity levels for each hidden size ──
# Group by hidden_size, compute variance of accuracy across sparsity levels
variance_df = df.groupby("hidden_size")["accuracy"].var().reset_index()
variance_df.columns = ["hidden_size", "accuracy_variance"]

fig3, ax3 = plt.subplots(figsize=(14, 6))
ax3.bar(variance_df["hidden_size"], variance_df["accuracy_variance"], width=80, color="steelblue", alpha=0.8)
ax3.set_xlabel("Hidden Size", fontsize=13)
ax3.set_ylabel("Variance of Accuracy (across sparsity levels)", fontsize=13)
ax3.set_title("Accuracy Variance across Sparsity levels vs Hidden Size", fontsize=15)
ax3.grid(True, alpha=0.3, axis="y")
fig3.tight_layout()
fig3.savefig("accuracy_variance_vs_hidden_size.png", dpi=200)
print("Saved accuracy_variance_vs_hidden_size.png")

# ── Plot 4: Loss variance across sparsity levels for each hidden size ──
loss_var_df = df.groupby("hidden_size")["best_val_loss"].var().reset_index()
loss_var_df.columns = ["hidden_size", "loss_variance"]

fig4, ax4 = plt.subplots(figsize=(14, 6))
ax4.bar(loss_var_df["hidden_size"], loss_var_df["loss_variance"], width=80, color="indianred", alpha=0.8)
ax4.set_xlabel("Hidden Size", fontsize=13)
ax4.set_ylabel("Variance of Val Loss (across sparsity levels)", fontsize=13)
ax4.set_title("Validation Loss Variance across Sparsity levels vs Hidden Size", fontsize=15)
ax4.grid(True, alpha=0.3, axis="y")
fig4.tight_layout()
fig4.savefig("loss_variance_vs_hidden_size.png", dpi=200)
print("Saved loss_variance_vs_hidden_size.png")

print("Done.")

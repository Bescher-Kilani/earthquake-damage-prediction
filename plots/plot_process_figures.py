"""
Process figures for README.md and docs/FULL_DOCUMENTATION.md.

The repository does not publish the raw data or the generated result files,
so every figure is drawn from the result tables published in
docs/FULL_DOCUMENTATION.md. The values below are copied from there.

Run from the repository root:

	python plots/plot_process_figures.py

Writes PNGs to docs/figures/.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT_DIR = Path("docs/figures")
TEAL = "#009682"
TEAL_LIGHT = "#8FD0C6"
GREY = "#B8BEC4"
DARK = "#2B2F33"
RED = "#C8553D"

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({
	"figure.facecolor": "white",
	"axes.facecolor": "white",
	"savefig.facecolor": "white",
	"axes.edgecolor": DARK,
	"axes.labelcolor": DARK,
	"text.color": DARK,
	"xtick.color": DARK,
	"ytick.color": DARK,
})


def save(fig, name: str) -> None:
	OUT_DIR.mkdir(parents=True, exist_ok=True)
	fig.savefig(OUT_DIR / name, dpi=160, bbox_inches="tight")
	plt.close(fig)
	print(f"wrote {OUT_DIR / name}")


def node(ax, x, y, w, h, title, sub, filled=False, edge=TEAL):
	"""Rounded box with a bold title and a smaller subtitle."""
	ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.12",
		fc=TEAL if filled else "white", ec=edge, lw=2))
	col = "white" if filled else DARK
	ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=10,
		fontweight="bold", color=col)
	ax.text(x + w / 2, y + h * 0.26, sub, ha="center", va="center", fontsize=8.5, color=col)


def arrow(ax, p0, p1, color=DARK, style="-|>"):
	ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=14, color=color, lw=1.5))


def fig_pipeline() -> None:
	"""Pipeline overview: preprocessing row and two-stage classifier row (schematic)."""
	fig, ax = plt.subplots(figsize=(14, 5.4))
	ax.set_xlim(0, 14.2)
	ax.set_ylim(0, 5.4)
	ax.axis("off")
	w, h, gap = 1.9, 1.3, 0.35
	top = [
		("Load data", "260,601 buildings\n38 features"),
		("Split", "before cleaning\n(no leakage)"),
		("Clean", "age = 0 → NaN\nmedian / mode"),
		("Drop columns", "constant, corr > 0.9"),
		("Encode", "label + target\nencoding"),
		("Features", "recombine binaries,\nrare → OTHER"),
	]
	x0 = 0.15
	for i, (t, s) in enumerate(top):
		x = x0 + i * (w + gap)
		node(ax, x, 3.7, w, h, t, s)
		if i < len(top) - 1:
			arrow(ax, (x + w + 0.02, 4.35), (x + w + gap - 0.02, 4.35))
	ax.text(0.15, 5.25, "Preprocessing: all statistics fit on the training part only",
		fontsize=10.5, color=RED, fontweight="bold", va="center")
	last_x = x0 + (len(top) - 1) * (w + gap) + w / 2
	arrow(ax, (last_x, 3.68), (last_x, 3.05), style="-|>")
	ax.plot([last_x, 1.1], [3.05, 3.05], color=DARK, lw=1.5)
	arrow(ax, (1.1, 3.05), (1.1, 2.55))
	node(ax, 0.15, 1.25, 2.1, 1.3, "Stage 1:\nRandom Forest", "all samples\ndepth 16, 500 trees", filled=True)
	node(ax, 3.1, 1.25, 2.4, 1.3, "max probability\n< 0.65 ?", "uncertain samples\n(~29 % of data)", edge=RED)
	node(ax, 6.4, 1.25, 2.4, 1.3, "Stage 2:\nLightGBM", "uncertain samples only\n76 leaves, 500 rounds", filled=True)
	node(ax, 9.7, 1.25, 2.1, 1.3, "Prediction", "damage grade 1 / 2 / 3")
	arrow(ax, (2.27, 1.9), (3.07, 1.9))
	arrow(ax, (5.52, 1.9), (6.37, 1.9))
	ax.text(5.95, 2.12, "yes", ha="center", fontsize=9, color=RED, fontweight="bold")
	arrow(ax, (8.82, 1.9), (9.67, 1.9))
	# confident samples skip stage 2
	ax.plot([4.3, 4.3], [1.22, 0.6], color=DARK, lw=1.5)
	ax.plot([4.3, 10.75], [0.6, 0.6], color=DARK, lw=1.5)
	arrow(ax, (10.75, 0.6), (10.75, 1.22))
	ax.text(7.5, 0.8, "no: keep the Random Forest prediction", ha="center", fontsize=9, color=DARK)
	ax.text(12.2, 1.9, "Metric: accuracy (= micro-F1),\nMCC as secondary metric", fontsize=9,
		va="center", color=DARK)
	save(fig, "01_pipeline.png")


def fig_eda() -> None:
	"""Week 1: target distribution and mutual information (values from the Week 1 section)."""
	fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [1, 1.4]})
	grades = ["Grade 1\nlow", "Grade 2\nmedium", "Grade 3\nnear-complete\ndestruction"]
	counts = [25124, 148259, 87218]
	pct = [9.6, 56.9, 33.5]
	a.bar(grades, counts, color=[GREY, TEAL, GREY])
	for i, (c, p) in enumerate(zip(counts, pct)):
		a.text(i, c + 3000, f"{c:,}\n({p}%)", ha="center", fontsize=9)
	a.set_ylim(0, 175000)
	a.set_ylabel("buildings")
	a.set_title("Target is imbalanced and ordinal", fontsize=11)
	mi = {
		"geo_level_3_id": 0.2967, "geo_level_2_id": 0.2341, "geo_level_1_id": 0.1336,
		"foundation_type": 0.0864, "has_superstructure_mud_mortar_stone": 0.0761,
		"other_floor_type": 0.0617, "ground_floor_type": 0.0550, "roof_type": 0.0417,
	}
	names = list(mi.keys())
	vals = list(mi.values())
	cols = [TEAL if n.startswith("geo_") else GREY for n in names]
	b.barh(names, vals, color=cols)
	b.invert_yaxis()
	for i, v in enumerate(vals):
		b.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=9)
	b.set_xlim(0, 0.35)
	b.set_xlabel("mutual information with damage_grade")
	b.set_title("Geography carries the most signal", fontsize=11)
	fig.tight_layout()
	save(fig, "02_target_and_mutual_information.png")


def fig_progress() -> None:
	"""Accuracy and MCC per week (Overall Performance table)."""
	weeks = ["W2\ncleaning", "W3\nfeatures", "W4\nmodels", "W5\ntwo-stage", "W6\ntuning"]
	acc = [0.6881, 0.7327, 0.7437, 0.7423, 0.7551]
	mcc = [0.4077, 0.5023, 0.5239, 0.5211, 0.5424]
	fig, axes = plt.subplots(1, 2, figsize=(12, 4))
	for ax, vals, name, lo, hi in ((axes[0], acc, "Accuracy (micro-F1)", 0.66, 0.78),
			(axes[1], mcc, "MCC", 0.38, 0.58)):
		x = np.arange(len(weeks))
		ax.plot(x, vals, color=TEAL, lw=2.2, marker="o", ms=8)
		for xi, v in zip(x, vals):
			dy = -(hi - lo) * 0.07 if xi == 0 else (hi - lo) * 0.045
			ax.text(xi, v + dy, f"{v:.4f}", ha="center", va="center", fontsize=9, fontweight="bold")
		ax.set_xlim(-0.5, len(weeks) - 0.5)
		ax.set_xticks(x, weeks)
		ax.set_ylim(lo, hi)
		ax.set_ylabel(name)
		ax.set_title(name, fontsize=11)
	fig.suptitle("Progress per week (W4: single split, W5-W6: 5-fold CV mean)", fontsize=12, y=1.02)
	fig.tight_layout()
	save(fig, "03_progress.png")


def fig_models() -> None:
	"""Week 4: single-split model comparison (accuracy, MCC, training time)."""
	models = ["Stacking", "Random Forest", "Voting (soft)", "LightGBM", "XGBoost", "CatBoost"]
	acc = [0.7437, 0.7426, 0.7424, 0.7418, 0.7392, 0.7366]
	mcc = [0.5239, 0.5185, 0.5198, 0.5209, 0.5146, 0.5087]
	sec = [127.0, 10.3, 27.5, 3.7, 32.6, 10.4]
	fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [1.3, 1]})
	y = np.arange(len(models))
	a.barh(y - 0.2, acc, 0.4, color=TEAL, label="Accuracy")
	a.barh(y + 0.2, mcc, 0.4, color=GREY, label="MCC")
	for yi, v1, v2 in zip(y, acc, mcc):
		a.text(v1 + 0.003, yi - 0.2, f"{v1:.4f}", va="center", fontsize=8.5)
		a.text(v2 + 0.003, yi + 0.2, f"{v2:.4f}", va="center", fontsize=8.5)
	a.set_yticks(y, models)
	a.invert_yaxis()
	a.set_xlim(0.45, 0.8)
	a.set_xlabel("single split, validation set")
	a.set_title("Top six models are within 0.71 pp accuracy", fontsize=11)
	a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
	b.barh(models, sec, color=[RED if s > 100 else GREY for s in sec])
	b.invert_yaxis()
	b.set_xscale("log")
	for i, s in enumerate(sec):
		b.text(s * 1.12, i, f"{s:g} s", va="center", fontsize=9)
	b.set_xlim(2, 400)
	b.set_xlabel("training time (s, log scale)")
	b.set_title("Stacking costs 12× the Random Forest", fontsize=11)
	fig.tight_layout()
	save(fig, "04_model_comparison.png")


def fig_confusion() -> None:
	"""Week 4: Stacking confusion matrix (counts from the documentation)."""
	cm = np.array([[2642, 2325, 58], [1119, 24815, 3718], [75, 6063, 11306]])
	norm = cm / cm.sum(axis=1, keepdims=True) * 100
	fig, ax = plt.subplots(figsize=(6, 4.8))
	sns.heatmap(norm, annot=False, cmap=sns.light_palette(TEAL, as_cmap=True), vmin=0, vmax=100,
		cbar_kws={"label": "% of true class"}, ax=ax, linewidths=2, linecolor="white")
	for i in range(3):
		for j in range(3):
			col = "white" if norm[i, j] > 55 else DARK
			ax.text(j + 0.5, i + 0.5, f"{norm[i, j]:.1f}%\n{cm[i, j]:,}", ha="center", va="center",
				fontsize=10, color=col, fontweight="bold" if (i, j) == (2, 1) else "normal")
	ax.add_patch(plt.Rectangle((1, 2), 1, 1, fill=False, ec=RED, lw=3))
	ax.set_xticklabels(["1", "2", "3"])
	ax.set_yticklabels(["1", "2", "3"], rotation=0)
	ax.set_xlabel("predicted grade")
	ax.set_ylabel("true grade")
	ax.set_title("Stacking: main error is grade 3 → grade 2", fontsize=11)
	save(fig, "05_confusion_matrix.png")


def fig_sampling() -> None:
	"""Week 5: sampling trade-off, 5-fold CV (accuracy vs grade 1 recall)."""
	rows = [
		("Baseline", 0.7388, 0.497), ("SMOTE + Under", 0.7179, 0.691), ("SMOTE", 0.7178, 0.688),
		("class_weight", 0.7008, 0.740), ("ROSE", 0.6858, 0.827), ("RandomUnderSampler", 0.6768, 0.842),
		("EasyEnsemble", 0.6587, 0.793), ("ClusterCentroids", 0.4450, 0.919),
	]
	fig, ax = plt.subplots(figsize=(8.5, 4.8))
	for name, acc, rec in rows:
		ax.scatter(rec, acc, s=90, color=TEAL if name == "Baseline" else GREY, edgecolor=DARK, zorder=3)
		dx, dy, ha = 0.012, 0.0, "left"
		if name == "SMOTE":
			dx, dy, ha = -0.012, -0.004, "right"
		if name == "SMOTE + Under":
			dx, dy = 0.012, 0.010
		if name == "ROSE":
			dx, dy = 0.012, 0.014
		if name == "RandomUnderSampler":
			dy = -0.002
		if name == "ClusterCentroids":
			dx, ha = -0.012, "right"
		ax.text(rec + dx, acc + dy, name, fontsize=9, ha=ha, va="center")
	ax.set_xlim(0.45, 1.0)
	ax.set_ylim(0.42, 0.76)
	ax.set_xlabel("grade 1 recall")
	ax.set_ylabel("accuracy")
	ax.set_title("Sampling: every recall gain costs accuracy (5-fold CV, 260k samples)", fontsize=11)
	save(fig, "06_sampling_tradeoff.png")


def fig_two_stage() -> None:
	"""Week 5: two-stage classifier, threshold sweep (accuracy gain and number of uncertain samples)."""
	thr = [0.50, 0.55, 0.60, 0.65, 0.70]
	dacc = [0.0002, 0.0007, 0.0008, 0.0012, 0.0010]
	unc = [981, 5867, 10724, 15670, 20605]
	fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
	a.bar([f"{t:.2f}" for t in thr], [d * 100 for d in dacc], color=[TEAL if t == 0.65 else GREY for t in thr])
	for i, d in enumerate(dacc):
		a.text(i, d * 100 + 0.004, f"+{d * 100:.2f} pp", ha="center", fontsize=9)
	a.set_xlabel("uncertainty threshold (max probability)")
	a.set_ylabel("accuracy gain over RF baseline (pp)")
	a.set_ylim(0, 0.17)
	a.set_title("Best gain at threshold 0.65", fontsize=11)
	b.plot([f"{t:.2f}" for t in thr], unc, color=GREY, marker="o", lw=2)
	b.scatter(["0.65"], [15670], color=TEAL, s=90, zorder=5)
	for i, u in enumerate(unc):
		b.text(i, u + 900, f"{u:,}", ha="center", fontsize=9)
	b.set_xlabel("uncertainty threshold (max probability)")
	b.set_ylabel("samples sent to stage 2 (per fold)")
	b.set_ylim(0, 24000)
	b.margins(x=0.12)
	b.set_title("More uncertain samples, higher cost", fontsize=11)
	fig.tight_layout()
	save(fig, "07_two_stage_threshold.png")


def fig_tuning() -> None:
	"""Week 6: tuning methods and nested CV (values from the Week 6 tables)."""
	fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1, 1.1]})
	names = ["Default", "Random search\n(20 iter.)", "Optuna\n(30 trials)", "Nested CV\n(unbiased)"]
	vals = [0.7423, 0.7554, 0.7551, 0.7535]
	cols = [GREY, TEAL_LIGHT, TEAL_LIGHT, TEAL]
	a.bar(names, vals, color=cols)
	for i, v in enumerate(vals):
		a.text(i, v + 0.0006, f"{v:.4f}", ha="center", fontsize=9)
	a.set_ylim(0.73, 0.76)
	a.set_ylabel("accuracy (CV, axis truncated)")
	a.set_title("Tuning: +1.3 pp, optimistic bias only 0.16 pp", fontsize=11)
	folds = ["Fold 1", "Fold 2", "Fold 3", "Mean"]
	inner = [0.7526, 0.7516, 0.7511, 0.7518]
	outer = [0.7520, 0.7537, 0.7547, 0.7535]
	x = np.arange(len(folds))
	b.bar(x - 0.2, inner, 0.4, color=GREY, label="inner CV (tuning)")
	b.bar(x + 0.2, outer, 0.4, color=TEAL, label="outer fold (evaluation)")
	for xi, v1, v2 in zip(x, inner, outer):
		b.text(xi - 0.2, v1 + 0.0004, f"{v1:.4f}", ha="center", fontsize=8)
		b.text(xi + 0.2, v2 + 0.0004, f"{v2:.4f}", ha="center", fontsize=8)
	b.set_xticks(x, folds)
	b.set_ylim(0.745, 0.758)
	b.set_ylabel("accuracy (axis truncated)")
	b.set_title("Nested CV: outer folds confirm the tuned model", fontsize=11)
	b.legend(loc="upper left", fontsize=8)
	fig.tight_layout()
	save(fig, "08_tuning.png")


def main() -> None:
	fig_pipeline()
	fig_eda()
	fig_progress()
	fig_models()
	fig_confusion()
	fig_sampling()
	fig_two_stage()
	fig_tuning()


if __name__ == "__main__":
	main()

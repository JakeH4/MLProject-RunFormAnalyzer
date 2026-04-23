"""
Generate publication-quality plots for the writeup.

Outputs PNGs into plots/:
  features_phase5_per_frame.png      per-class distributions of raw features
  features_phase6_per_stride.png     same but at foot-contact moments
  confusion_phase5.png               Phase 5 LOSO confusion matrix
  confusion_phase6.png               Phase 6 LOSO confusion matrix
  f1_comparison.png                  per-class F1 across data/feature combos

Run: python3 plots.py
"""

import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score

from train_phase6 import load_all_strides
from train_combined import load_with_contact_features, COMBINED_FEATS

HERE = os.path.dirname(os.path.abspath(__file__))
RECORDINGS_DIR = os.path.join(HERE, "recordings")
V1_DIR = os.path.join(RECORDINGS_DIR, "version1data")
PLOTS_DIR = os.path.join(HERE, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")
LABELS = ["good", "overstride", "excessive_lean"]
LABEL_PALETTE = {
    "good": "#2ca02c",
    "overstride": "#d62728",
    "excessive_lean": "#1f77b4",
}


def load_frame_data(folder):
    frames = []
    for path in sorted(glob.glob(os.path.join(folder, "session_*.csv"))):
        df = pd.read_csv(path)
        if "label" not in df.columns:
            continue
        df["source_file"] = os.path.basename(path)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(out):
        out = out[out["ready"] == 1]
    return out


def plot_feature_distributions(df, features, title, outfile):
    """Side-by-side violin plots of each feature colored by label."""
    df = df[df["label"].isin(LABELS)].copy()
    fig, axes = plt.subplots(1, len(features), figsize=(4 * len(features), 5), sharey=False)
    if len(features) == 1:
        axes = [axes]
    for ax, feat in zip(axes, features):
        sub = df[[feat, "label"]].dropna()
        sns.violinplot(
            data=sub, x="label", y=feat, hue="label", ax=ax,
            order=LABELS, palette=LABEL_PALETTE, legend=False,
            cut=0, inner="quartile",
        )
        ax.set_xlabel("")
        ax.set_title(feat, fontsize=12)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outfile}")


def loso_cv(df, features):
    """Run LOSO CV, return aggregate (true, pred) arrays."""
    true_all, pred_all = [], []
    for held in sorted(df["source_file"].unique()):
        train = df[df["source_file"] != held]
        test = df[df["source_file"] == held]
        clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        clf.fit(train[features], train["label"])
        preds = clf.predict(test[features])
        true_all.extend(test["label"].tolist())
        pred_all.extend(preds.tolist())
    return true_all, pred_all


def plot_confusion(y_true, y_pred, title, outfile):
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(
        cm_norm, annot=cm, fmt="d", cmap="Blues", cbar=True,
        xticklabels=LABELS, yticklabels=LABELS, vmin=0, vmax=1, ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outfile}")


def plot_f1_comparison(rows, outfile):
    """rows: list of {experiment, label, f1}"""
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=df, x="label", y="f1", hue="experiment",
        order=LABELS, ax=ax, palette="Set2",
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1 score (LOSO CV)")
    ax.set_xlabel("")
    ax.set_title("Per-class F1 across experiments", fontsize=13)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=9, padding=3)
    ax.legend(title="Experiment", loc="upper right")
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outfile}")


def main():
    # --- Feature distributions ------------------------------------------
    new_frames = load_frame_data(RECORDINGS_DIR)
    plot_feature_distributions(
        new_frames,
        ["knee_angle", "trunk_lean", "foot_offset", "cadence_spm"],
        "Phase 5 features — per-frame distributions (new data)",
        os.path.join(PLOTS_DIR, "features_phase5_per_frame.png"),
    )

    strides = pd.DataFrame(load_all_strides())
    plot_feature_distributions(
        strides,
        ["foot_offset_at_contact", "knee_angle_at_contact",
         "trunk_lean_at_contact", "cadence_at_contact"],
        "Phase 6 features — at foot-contact moments",
        os.path.join(PLOTS_DIR, "features_phase6_per_stride.png"),
    )

    # --- Phase 5 LOSO on new data ---------------------------------------
    p5_feats = ["knee_angle", "trunk_lean", "foot_offset", "cadence_spm"]
    p5_df = new_frames.dropna(subset=p5_feats + ["label"])
    p5_true, p5_pred = loso_cv(p5_df, p5_feats)
    plot_confusion(
        p5_true, p5_pred,
        "Phase 5: per-frame features, LOSO CV (new data)",
        os.path.join(PLOTS_DIR, "confusion_phase5.png"),
    )

    # --- Phase 6 LOSO on strides ----------------------------------------
    p6_feats = ["foot_offset_at_contact", "knee_angle_at_contact",
                "trunk_lean_at_contact", "cadence_at_contact"]
    p6_true, p6_pred = loso_cv(strides, p6_feats)
    plot_confusion(
        p6_true, p6_pred,
        "Phase 6: per-stride features, LOSO CV",
        os.path.join(PLOTS_DIR, "confusion_phase6.png"),
    )

    # --- Phase 5 on OLD (v1) data, for the data-quality comparison ------
    old_frames = load_frame_data(V1_DIR)
    old_f1 = {}
    if len(old_frames):
        old_df = old_frames.dropna(subset=p5_feats + ["label"])
        o_true, o_pred = loso_cv(old_df, p5_feats)
        for label in LABELS:
            old_f1[label] = f1_score(o_true, o_pred, labels=[label], average="macro")

    # --- Combined (Option B) LOSO --------------------------------------
    combined_frames = load_with_contact_features()
    combined_frames = combined_frames[combined_frames["ready"] == 1]
    combined_df = combined_frames.dropna(subset=COMBINED_FEATS + ["label"])
    c_true, c_pred = loso_cv(combined_df, COMBINED_FEATS)
    plot_confusion(
        c_true, c_pred,
        "Combined features (Option B): LOSO CV",
        os.path.join(PLOTS_DIR, "confusion_combined.png"),
    )

    # --- F1 comparison bars ---------------------------------------------
    p5_f1 = {l: f1_score(p5_true, p5_pred, labels=[l], average="macro") for l in LABELS}
    p6_f1 = {l: f1_score(p6_true, p6_pred, labels=[l], average="macro") for l in LABELS}
    c_f1 = {l: f1_score(c_true, c_pred, labels=[l], average="macro") for l in LABELS}
    rows = []
    for label in LABELS:
        if label in old_f1:
            rows.append({"experiment": "Phase 5 on old data", "label": label, "f1": old_f1[label]})
        rows.append({"experiment": "Phase 5 on new data", "label": label, "f1": p5_f1[label]})
        rows.append({"experiment": "Phase 6 on new data", "label": label, "f1": p6_f1[label]})
        rows.append({"experiment": "Combined on new data", "label": label, "f1": c_f1[label]})
    plot_f1_comparison(rows, os.path.join(PLOTS_DIR, "f1_comparison.png"))

    print("\nDone.  All plots in plots/")


if __name__ == "__main__":
    main()

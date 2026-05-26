"""
MCASF Figure Generator
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import os
import sys

from extract_features import process_ply_file
from forensics import run_forensics

# ── Style ─────────────────────────────────────────────────────────────────────

BLUE   = "#1F4E79"
BLUE2  = "#2E75B6"
BLUE3  = "#9DC3E6"
ORANGE = "#C55A11"
GRAY   = "#A0A0A0"
GREEN  = "#375623"
BG     = "#F8FAFC"

plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'font.size':        11,
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'axes.grid':        True,
    'grid.alpha':       0.3,
    'grid.linestyle':   '--',
    'figure.dpi':       150,
    'savefig.dpi':      200,
    'savefig.bbox':     'tight',
    'savefig.facecolor': BG,
    'axes.facecolor':   BG,
    'figure.facecolor': BG,
})

OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_mos(mos_path):
    try:
        df = pd.read_excel(mos_path, engine='openpyxl')
    except Exception:
        df = pd.read_csv(mos_path)
    df['3DGS_name'] = df['3DGS_name'].str.strip()

    def assign_label(row):
        if pd.notna(row['Distortion_Type']):
            return row['Distortion_Type']
        if row['Viewports'] != 360:
            return 'Viewpoint reduction'
        if row['Iterations'] == 7000:
            return 'Iteration reduction'
        return 'Reference'

    df['Distortion_Label'] = df.apply(assign_label, axis=1)
    return df


def build_features(ply_dir, df):
    available = {f.lower(): f for f in os.listdir(ply_dir) if f.endswith('.ply')}
    X, y, labels, contents = [], [], [], []
    feat_names = None

    for _, row in df.iterrows():
        fname = row['3DGS_name']
        actual = available.get(fname.lower())
        if actual is None:
            continue
        try:
            feats, names = process_ply_file(os.path.join(ply_dir, actual))
            X.append(feats)
            y.append(row['MOS'])
            labels.append(row['Distortion_Label'])
            contents.append(row['Content'])
            if feat_names is None:
                feat_names = names
        except:
            pass

    return (np.nan_to_num(np.array(X), nan=0.0, posinf=1e6, neginf=-1e6),
            np.array(y), np.array(labels), np.array(contents), feat_names)


# ── Figure 1: SRCC / PLCC bar chart ──────────────────────────────────────────

def fig1_srcc_plcc():
    dtypes = ['All', 'Viewpoint\nreduction', 'Iteration\nreduction',
              'Downsampling', 'Color\ndistortion', 'Gaussian\nnoise']
    srcc   = [0.7260, 0.6347, 0.7000, 0.7194, 0.7153, 0.8346]
    plcc   = [0.7786, 0.8785, 0.7553, 0.7252, 0.7461, 0.8056]
    n      = [209, 60, 14, 45, 45, 45]

    x = np.arange(len(dtypes))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - w/2, srcc, w, color=BLUE2,  label='SRCC', zorder=3)
    b2 = ax.bar(x + w/2, plcc, w, color=ORANGE, label='PLCC', alpha=0.85, zorder=3)

    # Value labels
    for bar in b1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom',
                fontsize=9, color=BLUE)
    for bar in b2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom',
                fontsize=9, color=ORANGE)

    # Sample size annotations
    for i, ni in enumerate(n):
        ax.text(i, 0.02, f'N={ni}', ha='center', va='bottom',
                fontsize=8, color='#555555')

    ax.set_xticks(x)
    ax.set_xticklabels(dtypes, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Correlation Coefficient')
    ax.set_title('Quality Prediction Performance per Distortion Type\n'
                 '(XGBoost, Content-Aware 5-Fold CV)', fontsize=8, color=BLUE, pad=12)
    ax.axhline(0.7, color=GRAY, linewidth=1, linestyle=':', zorder=2)
    ax.text(5.6, 0.71, '0.70', fontsize=7, color=GRAY)
    ax.legend(loc='lower right', framealpha=0.9)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig1_srcc_plcc_bar.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ── Figure 2: Predicted vs Actual MOS scatter ─────────────────────────────────

def fig2_scatter(X, y, labels):
    from sklearn.preprocessing import RobustScaler
    from sklearn.model_selection import KFold
    from xgboost import XGBRegressor

    mask = labels != 'Reference'
    X_s, y_s, lab_s = X[mask], y[mask], labels[mask]

    if len(X_s) < 5:
        print(f"  [SKIP] fig2: need at least 5 samples, only {len(X_s)} available")
        return
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    all_true, all_pred, all_lab = [], [], []
    for train_idx, test_idx in kf.split(X_s):
        scaler = RobustScaler()
        Xtr = scaler.fit_transform(X_s[train_idx])
        Xte = scaler.transform(X_s[test_idx])
        model = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             random_state=42, verbosity=0)
        model.fit(Xtr, y_s[train_idx])
        preds = model.predict(Xte)
        all_true.extend(y_s[test_idx])
        all_pred.extend(preds)
        all_lab.extend(lab_s[test_idx])

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    all_lab  = np.array(all_lab)

    dtype_colors = {
        'Viewpoint reduction': '#1F4E79',
        'Iteration reduction': '#2E75B6',
        'Downsampling':        '#C55A11',
        'Color distortion':    '#375623',
        'Gaussian noise':      '#7030A0',
    }

    fig, ax = plt.subplots(figsize=(7, 6))

    for dt, color in dtype_colors.items():
        mask_dt = all_lab == dt
        if mask_dt.sum() == 0:
            continue
        ax.scatter(all_true[mask_dt], all_pred[mask_dt],
                   color=color, alpha=0.65, s=28, label=dt, zorder=3)

    # Diagonal
    lims = [1.0, 5.0]
    ax.plot(lims, lims, color=GRAY, linewidth=1.2, linestyle='--', zorder=2)

    # SRCC / PLCC annotation
    from scipy.stats import spearmanr, pearsonr
    srcc, _ = spearmanr(all_true, all_pred)
    plcc, _ = pearsonr(all_true, all_pred)
    ax.text(0.05, 0.93, f'SRCC = {srcc:.3f}   PLCC = {plcc:.3f}',
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax.set_xlim(0.8, 5.2)
    ax.set_ylim(0.8, 5.2)
    ax.set_xlabel('Actual MOS')
    ax.set_ylabel('Predicted MOS')
    ax.set_title('Predicted vs Actual MOS\n(All Distortion Types, 5-Fold CV)',
                 fontsize=8, color=BLUE, pad=12)
    ax.legend(loc='lower right', fontsize=7, framealpha=0.9)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig2_scatter_all.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ── Figure 3: Feature importance ──────────────────────────────────────────────

def fig3_feature_importance(X, y, labels, feat_names):
    from sklearn.preprocessing import RobustScaler
    from xgboost import XGBRegressor

    mask = labels != 'Reference'
    scaler = RobustScaler()
    X_s = scaler.fit_transform(X[mask])
    model = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8,
                         random_state=42, verbosity=0)
    model.fit(X_s, y[mask])
    imp = model.feature_importances_
    top_idx = np.argsort(imp)[::-1][:15]

    names = [feat_names[i].replace('_', ' ') for i in top_idx]
    vals  = imp[top_idx]

    # Color by feature group
    group_colors = {
        'eigval':     '#1F4E79',
        'anisotropy': '#2E75B6',
        'scale':      '#9DC3E6',
        'opacity':    '#C55A11',
        'spatial':    '#375623',
        'nn':         '#7030A0',
        'sh':         '#843C0C',
        'n gauss':    '#1F4E79',
        'log n':      '#1F4E79',
        'scene':      '#9DC3E6',
        'spread':     '#375623',
        'trace':      '#2E75B6',
    }

    def get_color(name):
        for key, col in group_colors.items():
            if key in name.lower():
                return col
        return GRAY

    colors = [get_color(n) for n in names]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(range(15), vals[::-1], color=colors[::-1], zorder=3)

    for i, (bar, val) in enumerate(zip(bars, vals[::-1])):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=8.5)

    ax.set_yticks(range(15))
    ax.set_yticklabels(names[::-1], fontsize=9.5)
    ax.set_xlabel('Feature Importance (XGBoost)')
    ax.set_title('Top 15 Feature Importances\n(XGBoost, All Distortion Types)',
                 fontsize=8, color=BLUE, pad=12)

    # Legend for groups
    legend_items = [
        mpatches.Patch(color='#1F4E79', label='Eigenvalue / Count'),
        mpatches.Patch(color='#2E75B6', label='Anisotropy / Trace'),
        mpatches.Patch(color='#9DC3E6', label='Scale / Scene MVG'),
        mpatches.Patch(color='#C55A11', label='Opacity'),
        mpatches.Patch(color='#375623', label='Spatial / Spread'),
        mpatches.Patch(color='#7030A0', label='Nearest-Neighbor'),
        mpatches.Patch(color='#843C0C', label='SH Coefficients'),
    ]
    ax.legend(handles=legend_items, loc='lower right', fontsize=8, framealpha=0.9)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig3_feature_importance.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ── Figure 4: Anomaly score distribution ──────────────────────────────────────

def fig4_anomaly(ply_dir):
    ref_path = os.path.join(ply_dir, 'bottle.ply')
    dis_path = os.path.join(ply_dir, 'bottle_25.ply')

    if not os.path.exists(ref_path) or not os.path.exists(dis_path):
        print(f"  [SKIP] fig4: bottle.ply or bottle_25.ply not found in {ply_dir}")
        return

    print("  Running forensics for fig4/5/6...")
    ref = run_forensics(ref_path, verbose=False)
    dis = run_forensics(dis_path, verbose=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)

    for ax, res, label, color in zip(
        axes,
        [ref, dis],
        ['Reference (360 views)', 'Distorted (25 views)'],
        [BLUE2, ORANGE]
    ):
        A_k = res['A_k']
        # Clip for readability
        A_clip = np.clip(A_k, 0, np.percentile(A_k, 99))
        ax.hist(A_clip, bins=60, color=color, alpha=0.75, edgecolor='white',
                linewidth=0.3, zorder=3)
        ax.axvline(np.mean(A_k), color='black', linewidth=1.5,
                   linestyle='--', label=f'Mean = {np.mean(A_k):.1f}')
        ax.axvline(np.percentile(A_k, 95), color='red', linewidth=1.2,
                   linestyle=':', label=f'95th pct = {np.percentile(A_k,95):.1f}')
        ax.set_title(label, fontsize=11, color=BLUE)
        ax.set_xlabel('Anomaly Score A_k')
        ax.set_ylabel('Count')
        ax.legend(fontsize=9)

    fig.suptitle('Distribution of Mahalanobis Anomaly Scores A_k\n'
                 '(Reference vs 25-View Distorted Bottle)',
                 fontsize=8, color=BLUE, y=1.02)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig4_anomaly_distribution.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")
    return ref, dis


# ── Figure 5: Consistency score distribution ──────────────────────────────────

def fig5_consistency(ref, dis):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)

    for ax, res, label, color in zip(
        axes,
        [ref, dis],
        ['Reference (360 views)', 'Distorted (25 views)'],
        [BLUE2, ORANGE]
    ):
        C = res['C_k_mean']
        ax.hist(C, bins=60, color=color, alpha=0.75, edgecolor='white',
                linewidth=0.3, zorder=3)
        ax.axvline(np.mean(C), color='black', linewidth=1.5,
                   linestyle='--', label=f'Mean = {np.mean(C):.4f}')
        ax.axvline(np.percentile(C, 95), color='red', linewidth=1.2,
                   linestyle=':', label=f'95th pct = {np.percentile(C,95):.4f}')
        ax.set_title(label, fontsize=11, color=BLUE)
        ax.set_xlabel('Consistency Score C_kj (mean per primitive)')
        ax.set_ylabel('Count')
        ax.legend(fontsize=9)

    fig.suptitle('Distribution of Pairwise Consistency Scores C_kj\n'
                 '(Reference vs 25-View Distorted Bottle)',
                 fontsize=8, color=BLUE, y=1.02)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig5_consistency_distribution.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ── Figure 6: Forensics radar chart ───────────────────────────────────────────

def fig6_radar(ref, dis):
    categories = [
        'Mean A_k\n(normalized)',
        'Mean C_kj\n(normalized)',
        '95th pct A_k\n(normalized)',
        '95th pct C_kj\n(normalized)',
        'Artifact\nrate (%)',
        'Std A_k\n(normalized)',
    ]

    def get_vals(res):
        return np.array([
            res['A_k'].mean(),
            res['C_k_mean'].mean(),
            np.percentile(res['A_k'], 95),
            np.percentile(res['C_k_mean'], 95),
            100 * res['n_artifact'] / res['n_gaussians'],
            res['A_k'].std(),
        ])

    ref_vals = get_vals(ref)
    dis_vals = get_vals(dis)

    # Normalize each metric relative to max of both
    maxvals = np.maximum(ref_vals, dis_vals) + 1e-10
    ref_norm = ref_vals / maxvals
    dis_norm = dis_vals / maxvals

    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    ref_plot = ref_norm.tolist() + ref_norm[:1].tolist()
    dis_plot = dis_norm.tolist() + dis_norm[:1].tolist()

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    ax.plot(angles, ref_plot, color=BLUE2,  linewidth=2, label='Reference (360-view)')
    ax.fill(angles, ref_plot, color=BLUE2,  alpha=0.15)
    ax.plot(angles, dis_plot, color=ORANGE, linewidth=2, linestyle='--',
            label='Distorted (25-view)')
    ax.fill(angles, dis_plot, color=ORANGE, alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9.5)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.50', '0.75', '1.0'], fontsize=7.5, color=GRAY)
    ax.set_title('Forensics Profile\nReference vs Distorted Scene',
                 fontsize=8, color=BLUE, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig6_forensics_radar.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    PLY_DIR  = sys.argv[1] if len(sys.argv) > 1 else "./data/model"
    MOS_PATH = sys.argv[2] if len(sys.argv) > 2 else "./data/3DGS_MOS.csv"

    print("=" * 55)
    print("MCASF Figure Generator")
    print("=" * 55)

    print("\n[1] Loading data...")
    df = load_mos(MOS_PATH)
    X, y, labels, contents, feat_names = build_features(PLY_DIR, df)

    print("\n[2] Generating figures...")

    print("  fig1: SRCC/PLCC bar chart...")
    fig1_srcc_plcc()

    print("  fig2: Predicted vs actual MOS scatter...")
    fig2_scatter(X, y, labels)

    print("  fig3: Feature importance...")
    fig3_feature_importance(X, y, labels, feat_names)

    result = fig4_anomaly(PLY_DIR)
    if result is not None:
        ref, dis = result
        print("  fig5: Consistency score distribution...")
        fig5_consistency(ref, dis)
        print("  fig6: Forensics radar chart...")
        fig6_radar(ref, dis)

    print(f"\nAll figures saved to ./{OUT_DIR}/")


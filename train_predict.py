"""
MVG-3DGS Quality Prediction Pipeline
======================================
Improvements over v1:
  1. Content-aware cross-validation (split by object, not randomly)
     - Prevents data leakage from same object appearing in train and test
     - More honest evaluation for generalization to unseen objects
  2. Hyperparameter tuning via GridSearchCV inside each fold
     - Tunes n_estimators, max_depth, learning_rate, subsample

Author: Bishr Omer
"""

import numpy as np
import pandas as pd
import os
from scipy.stats import spearmanr, pearsonr
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, Matern, RationalQuadratic, WhiteKernel, ConstantKernel as C
)
from sklearn.decomposition import PCA
from xgboost import XGBRegressor

from extract_features import process_ply_file


# ---------------------------------------------------------------------------
# 1. LOAD MOS
# ---------------------------------------------------------------------------

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

    print(f"\nDistortion type breakdown:")
    for dtype, count in df['Distortion_Label'].value_counts().items():
        print(f"  {dtype:<25}: {count} samples")

    return df


# ---------------------------------------------------------------------------
# 2. BUILD FEATURE MATRIX
# ---------------------------------------------------------------------------

def build_feature_matrix(ply_dir, df):
    available = {f.lower(): f for f in os.listdir(ply_dir) if f.endswith('.ply')}

    X, y, labels, contents, filenames = [], [], [], [], []
    feat_names = None
    matched, skipped = 0, 0

    for _, row in df.iterrows():
        fname   = row['3DGS_name']
        mos     = row['MOS']
        label   = row['Distortion_Label']
        content = row['Content']

        actual_fname = available.get(fname.lower())
        if actual_fname is None:
            skipped += 1
            continue

        fpath = os.path.join(ply_dir, actual_fname)
        try:
            feats, names = process_ply_file(fpath)
            X.append(feats)
            y.append(mos)
            labels.append(label)
            contents.append(content)
            filenames.append(fname)
            if feat_names is None:
                feat_names = names
            matched += 1
        except Exception as e:
            print(f"  [ERROR] {fname}: {e}")
            skipped += 1

    print(f"\nMatched: {matched} | Skipped/missing: {skipped}")
    return (np.array(X), np.array(y), np.array(labels),
            np.array(contents), filenames, feat_names)


# ---------------------------------------------------------------------------
# 3. EVALUATION
# ---------------------------------------------------------------------------

def evaluate(y_true, y_pred):
    srcc, _ = spearmanr(y_true, y_pred)
    plcc, _ = pearsonr(y_true, y_pred)
    rmse    = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return srcc, plcc, rmse


# ---------------------------------------------------------------------------
# 4. CONTENT-AWARE CROSS-VALIDATION WITH HYPERPARAMETER TUNING
# ---------------------------------------------------------------------------

# XGBoost hyperparameter grid
PARAM_GRID = {
    'xgb__n_estimators':   [100, 200, 300],
    'xgb__max_depth':      [3, 4, 5],
    'xgb__learning_rate':  [0.01, 0.05, 0.1],
    'xgb__subsample':      [0.7, 0.8, 1.0],
    'xgb__colsample_bytree': [0.7, 0.8, 1.0],
}

# Smaller grid for faster runs when n_samples is small
PARAM_GRID_SMALL = {
    'xgb__n_estimators':   [100, 200],
    'xgb__max_depth':      [3, 4],
    'xgb__learning_rate':  [0.05, 0.1],
    'xgb__subsample':      [0.8, 1.0],
    'xgb__colsample_bytree': [0.8, 1.0],
}


# ---------------------------------------------------------------------------
# GPR KERNEL DEFINITIONS
# ---------------------------------------------------------------------------
# GPR needs PCA first to reduce 98 dims to manageable size.
# GPR scales O(N^3) with samples so we keep n_components low.

def build_gpr_kernels():
    """
    Three kernel candidates to try:
    1. Matern-5/2 - smooth but not infinitely differentiable, good for real data
    2. RBF (squared exponential) - very smooth, good baseline
    3. RationalQuadratic - mixture of RBF at different scales
    All include WhiteKernel for noise estimation.
    """
    kernels = [
        C(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.1),
        C(1.0) * RBF(length_scale=1.0)             + WhiteKernel(noise_level=0.1),
        C(1.0) * RationalQuadratic(length_scale=1.0, alpha=1.0) + WhiteKernel(noise_level=0.1),
    ]
    return kernels


def build_gpr(kernel, n_components=20, random_state=42):
    """
    Build a GPR pipeline: RobustScaler -> PCA -> GPR.
    PCA reduces dimensionality before GPR to keep it tractable.
    """
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=5,
        normalize_y=True,
        random_state=random_state,
        alpha=1e-6,
    )
    return gpr, n_components


def run_content_aware_cv(X, y, contents, model_type='xgboost',
                          n_splits=5, tune=True,
                          n_pca_components=20, random_state=42, verbose=True):
    """
    Content-aware k-fold CV supporting XGBoost and GPR.

    Args:
        X               : (N, D) feature matrix
        y               : (N,) MOS scores
        contents        : (N,) object names
        model_type      : 'xgboost' or 'gpr'
        n_splits        : CV folds
        tune            : tune XGBoost hyperparams (ignored for GPR, which
                          auto-optimizes kernel params via marginal likelihood)
        n_pca_components: PCA dims before GPR (GPR scales O(N^3), keep low)
    """
    unique_contents = np.unique(contents)
    n_objects = len(unique_contents)

    if n_objects < n_splits:
        print(f"  [WARN] Only {n_objects} objects, reducing to {n_objects}-fold CV")
        n_splits = n_objects

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    object_folds = list(kf.split(unique_contents))

    all_srcc, all_plcc, all_rmse = [], [], []
    all_y_true, all_y_pred = [], []
    best_params_per_fold = []

    for fold, (train_obj_idx, test_obj_idx) in enumerate(object_folds):
        train_objects = unique_contents[train_obj_idx]
        test_objects  = unique_contents[test_obj_idx]

        train_mask = np.isin(contents, train_objects)
        test_mask  = np.isin(contents, test_objects)

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        if len(X_test) == 0:
            continue

        # Scale
        scaler = RobustScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        if model_type == 'gpr':
            # PCA to reduce dims for GPR tractability
            n_comp = min(n_pca_components, X_train_s.shape[0] - 1,
                         X_train_s.shape[1])
            pca = PCA(n_components=n_comp, random_state=random_state)
            X_train_pca = pca.fit_transform(X_train_s)
            X_test_pca  = pca.transform(X_test_s)

            # Try all kernels, pick best on training log-likelihood
            kernels = build_gpr_kernels()
            best_gpr, best_ll = None, -np.inf
            best_kernel_name = ''

            kernel_names = ['Matern-5/2', 'RBF', 'RationalQuadratic']
            for kname, kernel in zip(kernel_names, kernels):
                gpr = GaussianProcessRegressor(
                    kernel=kernel,
                    n_restarts_optimizer=5,
                    normalize_y=True,
                    random_state=random_state,
                    alpha=1e-6,
                )
                gpr.fit(X_train_pca, y_train)
                ll = gpr.log_marginal_likelihood_value_
                if ll > best_ll:
                    best_ll   = ll
                    best_gpr  = gpr
                    best_kernel_name = kname

            y_pred = best_gpr.predict(X_test_pca)
            best_params = {'kernel': best_kernel_name, 'pca_components': n_comp}
            best_params_per_fold.append(best_params)

            srcc, plcc, rmse = evaluate(y_test, y_pred)
            all_srcc.append(srcc)
            all_plcc.append(plcc)
            all_rmse.append(rmse)
            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)

            if verbose:
                print(f"    Fold {fold+1} "
                      f"[train:{train_mask.sum()} test:{test_mask.sum()}] "
                      f"SRCC={srcc:.4f}  PLCC={plcc:.4f}  RMSE={rmse:.4f}"
                      f"  kernel={best_kernel_name} pca={n_comp}")

        else:
            # XGBoost with optional HP tuning
            if tune and len(X_train) >= 20:
                param_grid = PARAM_GRID_SMALL if len(X_train) < 60 else PARAM_GRID
                pipe = Pipeline([
                    ('xgb', XGBRegressor(random_state=random_state, verbosity=0))
                ])
                inner_cv = KFold(n_splits=3, shuffle=True, random_state=random_state)
                search = GridSearchCV(
                    pipe, param_grid, cv=inner_cv,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1, refit=True
                )
                search.fit(X_train_s, y_train)
                best_model  = search.best_estimator_
                best_params = search.best_params_
                y_pred = best_model.predict(X_test_s)
            else:
                model = XGBRegressor(
                    n_estimators=200, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=random_state, verbosity=0
                )
                model.fit(X_train_s, y_train)
                y_pred = model.predict(X_test_s)
                best_params = {}

            best_params_per_fold.append(best_params)
            srcc, plcc, rmse = evaluate(y_test, y_pred)
            all_srcc.append(srcc)
            all_plcc.append(plcc)
            all_rmse.append(rmse)
            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)

            if verbose:
                print(f"    Fold {fold+1} "
                      f"[train:{train_mask.sum()} test:{test_mask.sum()}] "
                      f"SRCC={srcc:.4f}  PLCC={plcc:.4f}  RMSE={rmse:.4f}"
                      + (f"  best_depth={best_params.get('xgb__max_depth','')}"
                         f" lr={best_params.get('xgb__learning_rate','')}"
                         if best_params else ""))

    srcc_o, plcc_o, rmse_o = evaluate(np.array(all_y_true), np.array(all_y_pred))

    if verbose:
        print(f"    Mean   : SRCC={np.mean(all_srcc):.4f}+/-{np.std(all_srcc):.4f}  "
              f"PLCC={np.mean(all_plcc):.4f}+/-{np.std(all_plcc):.4f}  "
              f"RMSE={np.mean(all_rmse):.4f}+/-{np.std(all_rmse):.4f}")
        print(f"    Overall: SRCC={srcc_o:.4f}  PLCC={plcc_o:.4f}  RMSE={rmse_o:.4f}")

    return {
        'mean_srcc': np.mean(all_srcc), 'std_srcc': np.std(all_srcc),
        'mean_plcc': np.mean(all_plcc), 'std_plcc': np.std(all_plcc),
        'mean_rmse': np.mean(all_rmse), 'std_rmse': np.std(all_rmse),
        'overall_srcc': srcc_o, 'overall_plcc': plcc_o, 'overall_rmse': rmse_o,
        'n_samples': len(X),
        'best_params': best_params_per_fold,
    }


# ---------------------------------------------------------------------------
# 5. FEATURE IMPORTANCE
# ---------------------------------------------------------------------------

def get_feature_importance(X, y, feat_names, top_k=15, label="All"):
    scaler = RobustScaler()
    X_s = scaler.fit_transform(X)
    model = XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
    )
    model.fit(X_s, y)
    importances = model.feature_importances_
    sorted_idx  = np.argsort(importances)[::-1]

    print(f"\n  Top {top_k} features [{label}]:")
    for rank, idx in enumerate(sorted_idx[:top_k]):
        print(f"    {rank+1:>2}. {feat_names[idx]:<38} {importances[idx]:.4f}")


# ---------------------------------------------------------------------------
# 6. MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    PLY_DIR  = sys.argv[1] if len(sys.argv) > 1 else "./data/model"
    MOS_PATH = sys.argv[2] if len(sys.argv) > 2 else "./data/3DGS_MOS.xlsx"

    print("=" * 65)
    print("MVG-3DGS Pipeline (Content-Aware CV + XGBoost vs GPR)")
    print("=" * 65)

    print(f"\n[1] Loading MOS from: {MOS_PATH}")
    df = load_mos(MOS_PATH)

    print(f"\n[2] Extracting MVG features from: {PLY_DIR}")
    X, y, labels, contents, filenames, feat_names = build_feature_matrix(PLY_DIR, df)
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    print(f"\n  Unique objects: {np.unique(contents).tolist()}")

    distortion_types = [
        'All',
        'Viewpoint reduction',
        'Iteration reduction',
        'Downsampling',
        'Color distortion',
        'Gaussian noise',
    ]

    results_xgb = {}
    results_gpr = {}

    for dtype in distortion_types:
        mask = (labels != 'Reference') if dtype == 'All' else (labels == dtype)
        X_sub   = X[mask]
        y_sub   = y[mask]
        con_sub = contents[mask]
        n       = mask.sum()

        if n < 10:
            continue

        # XGBoost
        print(f"\n{'='*65}")
        print(f"[XGBoost] [{dtype}]  ({n} samples, {len(np.unique(con_sub))} objects)")
        print("-" * 65)
        do_tune = n >= 30
        results_xgb[dtype] = run_content_aware_cv(
            X_sub, y_sub, con_sub,
            model_type='xgboost', n_splits=5, tune=do_tune
        )

        # GPR
        print(f"\n[GPR]     [{dtype}]  ({n} samples, {len(np.unique(con_sub))} objects)")
        print("-" * 65)
        n_pca = min(20, n // 3)
        results_gpr[dtype] = run_content_aware_cv(
            X_sub, y_sub, con_sub,
            model_type='gpr', n_splits=5,
            n_pca_components=n_pca
        )

    # Feature importance (XGBoost only)
    print(f"\n{'='*65}")
    print("[Feature Importance - XGBoost, Full Dataset]")
    mask_all = labels != 'Reference'
    get_feature_importance(X[mask_all], y[mask_all], feat_names,
                           top_k=15, label="All distortions")

    # Comparison table
    print(f"\n{'='*65}")
    print("FINAL COMPARISON (Content-Aware CV, 5-fold)")
    print("=" * 65)
    print(f"{'Distortion Type':<25} {'N':>5}  "
          f"{'XGB_SRCC':>9} {'XGB_PLCC':>9}  "
          f"{'GPR_SRCC':>9} {'GPR_PLCC':>9}  {'Winner':>8}")
    print("-" * 85)

    for dtype in distortion_types:
        xgb = results_xgb.get(dtype)
        gpr = results_gpr.get(dtype)
        if not xgb or not gpr:
            continue
        n      = xgb['n_samples']
        winner = 'GPR' if gpr['mean_srcc'] > xgb['mean_srcc'] else 'XGBoost'
        print(f"{dtype:<25} {n:>5}  "
              f"{xgb['mean_srcc']:>9.4f} {xgb['mean_plcc']:>9.4f}  "
              f"{gpr['mean_srcc']:>9.4f} {gpr['mean_plcc']:>9.4f}  "
              f"{winner:>8}")

    print(f"\n[NOTE] Content-aware CV - split by object, no leakage.")
    print(f"       GPR uses PCA preprocessing + kernel selection per fold.")


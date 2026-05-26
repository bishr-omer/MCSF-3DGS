"""
MCASF Forensics Engine
======================
Implements the core diagnostic framework:

These are the two core forensic metrics for detecting floaters and
structural collapse in 3DGS scenes without ground-truth reference.

Author: Bishr Omer
"""

import numpy as np
from scipy.spatial import KDTree
from extract_features import load_ply, parse_attributes


# ---------------------------------------------------------------------------
# 1. DESCRIPTOR EXTRACTION (Proposal Section 3.1)
# ---------------------------------------------------------------------------

def extract_primitive_descriptors(attrs, k_neighbors=8):
    """
    For each primitive k, extract descriptor vector x_k containing:
      - Upper triangle of covariance matrix Sigma_k  (6 values)
      - Opacity alpha_k                              (1 value)
      - Displacement vector to centroid of k-NN      (3 values)
    Total: 10-dimensional descriptor per primitive.

    Args:
        attrs       : parsed attributes dict from parse_attributes()
        k_neighbors : number of spatial neighbors for centroid displacement

    Returns:
        X_desc : (N, 10) descriptor matrix
        covs   : (N, 3, 3) covariance matrices (for pairwise scoring)
        tree   : KDTree over positions (for neighbor queries)
        indices: (N, k) neighbor indices per primitive
    """
    positions   = attrs['positions'].astype(np.float64)   # (N, 3)
    covariances = attrs['covariances'].astype(np.float64) # (N, 3, 3)
    opacities   = attrs['opacities'].astype(np.float64)   # (N,)

    sig_opacity = 1.0 / (1.0 + np.exp(-opacities))       # sigmoid to [0,1]
    N = positions.shape[0]

    # Build KD-tree for spatial neighbor queries
    tree = KDTree(positions)
    # Query k+1 neighbors (includes self), then drop self
    dists, indices = tree.query(positions, k=k_neighbors + 1)
    neighbor_indices = indices[:, 1:]  # (N, k) - exclude self

    # Centroid of k-NN positions
    neighbor_positions = positions[neighbor_indices]          # (N, k, 3)
    centroids = neighbor_positions.mean(axis=1)              # (N, 3)
    displacement = positions - centroids                     # (N, 3)

    # Upper triangle of covariance matrix (6 unique values from 3x3 symmetric)
    cov_upper = np.stack([
        covariances[:, 0, 0],  # sigma_xx
        covariances[:, 0, 1],  # sigma_xy
        covariances[:, 0, 2],  # sigma_xz
        covariances[:, 1, 1],  # sigma_yy
        covariances[:, 1, 2],  # sigma_yz
        covariances[:, 2, 2],  # sigma_zz
    ], axis=1)  # (N, 6)

    # Assemble descriptor: [cov_upper | opacity | displacement]
    X_desc = np.concatenate([
        cov_upper,                          # (N, 6)
        sig_opacity[:, np.newaxis],         # (N, 1)
        displacement,                       # (N, 3)
    ], axis=1)  # (N, 10)

    return X_desc, covariances, tree, neighbor_indices


# ---------------------------------------------------------------------------
# 2. REFERENCE MVG FITTING (Equations 1-2)
# ---------------------------------------------------------------------------

def fit_reference_mvg(X_desc, attrs, opacity_threshold=0.7,
                      scale_threshold=0.5):
    """
    Fit reference MVG distribution mu_hat, Sigma_hat on high-confidence
    core primitives (high opacity, low local scale variance).

    These represent well-constrained geometry - the 'healthy' primitives
    that define what a normal covariance structure looks like.

    Args:
        X_desc            : (N, 10) descriptor matrix
        attrs             : parsed attributes
        opacity_threshold : sigmoid opacity above this = high confidence
        scale_threshold   : primitives with scale std below this percentile

    Returns:
        mu_hat    : (10,) reference mean vector
        sigma_hat : (10, 10) reference covariance matrix
        core_mask : (N,) boolean mask of core primitives used for fitting
    """
    opacities = attrs['opacities'].astype(np.float64)
    sig_opacity = 1.0 / (1.0 + np.exp(-opacities))

    scales = np.exp(attrs['scales'].astype(np.float64))  # (N, 3)
    scale_std = scales.std(axis=1)                        # per-primitive scale spread
    scale_threshold_val = np.percentile(scale_std, scale_threshold * 100)

    # Core primitives: high opacity AND low scale variance
    core_mask = (sig_opacity >= opacity_threshold) & \
                (scale_std <= scale_threshold_val)

    n_core = core_mask.sum()
    print(f"  Core primitives for MVG fitting: {n_core} / {len(X_desc)} "
          f"({100*n_core/len(X_desc):.1f}%)")

    if n_core < 20:
        # Fallback: use top 30% by opacity if strict threshold is too tight
        print(f"  [WARN] Too few core primitives, relaxing to top 30% by opacity")
        threshold = np.percentile(sig_opacity, 70)
        core_mask = sig_opacity >= threshold
        n_core = core_mask.sum()

    X_core = X_desc[core_mask]

    # Equation (1): mu_hat = mean of core descriptors
    mu_hat = X_core.mean(axis=0)

    # Equation (2): Sigma_hat = covariance of core descriptors
    sigma_hat = np.cov(X_core.T)

    # Regularize to ensure invertibility
    sigma_hat += np.eye(sigma_hat.shape[0]) * 1e-6

    return mu_hat, sigma_hat, core_mask


# ---------------------------------------------------------------------------
# 3. MAHALANOBIS ANOMALY SCORE ( Equation 2)
# ---------------------------------------------------------------------------

def compute_anomaly_scores(X_desc, mu_hat, sigma_hat):
    """
    Equation (3): A_k = (x_k - mu_hat)^T * Sigma_hat^{-1} * (x_k - mu_hat)

    Per-primitive Mahalanobis distance from the reference MVG.
    High A_k = structurally anomalous primitive (floater or collapsed region).

    Args:
        X_desc    : (N, 10) descriptor matrix
        mu_hat    : (10,) reference mean
        sigma_hat : (10, 10) reference covariance

    Returns:
        A_k : (N,) anomaly scores
    """
    sigma_inv = np.linalg.inv(sigma_hat)
    diff = X_desc - mu_hat[np.newaxis, :]          # (N, 10)
    # Vectorized Mahalanobis: diag(diff @ Sigma^-1 @ diff^T)
    A_k = np.einsum('ni,ij,nj->n', diff, sigma_inv, diff)
    return A_k


# ---------------------------------------------------------------------------
# 4. PAIRWISE COVARIANCE CONSISTENCY SCORE (Proposal Equation 4)
# ---------------------------------------------------------------------------

def compute_pairwise_consistency(covariances, neighbor_indices):
    """
    Equation (4): C_kj = || Sigma_k/||Sigma_k||_F - Sigma_j/||Sigma_j||_F ||_F

    For each primitive k and its spatial neighbors j, compute the
    Frobenius-norm difference between their normalized covariance matrices.

    High C_kj = physically implausible discontinuity between adjacent
    primitives = structural collapse signature.

    Args:
        covariances      : (N, 3, 3) covariance matrices
        neighbor_indices : (N, k) spatial neighbor indices

    Returns:
        C_k_mean : (N,) mean pairwise consistency score per primitive
        C_k_max  : (N,) max pairwise consistency score per primitive
    """
    N, k = neighbor_indices.shape

    # Normalize each covariance by its Frobenius norm
    frob_norms = np.linalg.norm(
        covariances.reshape(N, -1), axis=1, keepdims=True
    ).reshape(N, 1, 1)  # (N, 1, 1)

    frob_norms = np.maximum(frob_norms, 1e-10)
    cov_normalized = covariances / frob_norms  # (N, 3, 3)

    C_k_mean = np.zeros(N)
    C_k_max  = np.zeros(N)

    for i in range(N):
        neighbors = neighbor_indices[i]         # (k,)
        cov_i = cov_normalized[i]               # (3, 3)
        cov_neighbors = cov_normalized[neighbors]  # (k, 3, 3)

        # Frobenius norm of difference for each neighbor
        diffs = cov_neighbors - cov_i[np.newaxis, :, :]  # (k, 3, 3)
        c_ij  = np.linalg.norm(diffs.reshape(k, -1), axis=1)  # (k,)

        C_k_mean[i] = c_ij.mean()
        C_k_max[i]  = c_ij.max()

    return C_k_mean, C_k_max


# ---------------------------------------------------------------------------
# 5. FLOATER / ARTIFACT DETECTION
# ---------------------------------------------------------------------------

def detect_artifacts(A_k, C_k_mean, anomaly_percentile=95,
                     consistency_percentile=95):
    """
    Classify primitives as structurally degenerate based on:
      - High Mahalanobis anomaly score (floaters, isolated artifacts)
      - High mean pairwise consistency score (structural collapse)
      - Both (severe artifacts)

    Args:
        A_k                    : (N,) anomaly scores
        C_k_mean               : (N,) mean pairwise consistency scores
        anomaly_percentile     : threshold percentile for A_k
        consistency_percentile : threshold percentile for C_k_mean

    Returns:
        floaters          : (N,) bool - high anomaly only
        collapsed         : (N,) bool - high consistency discontinuity only
        severe            : (N,) bool - both flags
        thresh_anomaly    : float threshold used
        thresh_consistency: float threshold used
    """
    thresh_a = np.percentile(A_k, anomaly_percentile)
    thresh_c = np.percentile(C_k_mean, consistency_percentile)

    high_anomaly     = A_k > thresh_a
    high_consistency = C_k_mean > thresh_c

    floaters  = high_anomaly & ~high_consistency
    collapsed = high_consistency & ~high_anomaly
    severe    = high_anomaly & high_consistency

    return floaters, collapsed, severe, thresh_a, thresh_c


# ---------------------------------------------------------------------------
# 6. FULL PIPELINE
# ---------------------------------------------------------------------------

def run_forensics(ply_path, k_neighbors=8, anomaly_pct=95,
                  consistency_pct=95, verbose=True):
    """
    Full MCASF forensics pipeline on a single .ply file.

    Returns:
        results dict with all scores and artifact masks
    """
    if verbose:
        print(f"\nLoading: {ply_path}")

    props, data = load_ply(ply_path)
    attrs = parse_attributes(props, data)
    N = attrs['n_gaussians']

    if verbose:
        print(f"  Primitives: {N:,}")
        print(f"  Extracting descriptors (k={k_neighbors} neighbors)...")

    # Step 1: Extract descriptors
    X_desc, covariances, tree, neighbor_indices = \
        extract_primitive_descriptors(attrs, k_neighbors=k_neighbors)

    # Step 2: Fit reference MVG on core primitives
    if verbose:
        print(f"  Fitting reference MVG on core primitives...")
    mu_hat, sigma_hat, core_mask = fit_reference_mvg(X_desc, attrs)

    # Step 3: Mahalanobis anomaly scores (Eq. 3)
    if verbose:
        print(f"  Computing Mahalanobis anomaly scores (Eq. 3)...")
    A_k = compute_anomaly_scores(X_desc, mu_hat, sigma_hat)

    # Step 4: Pairwise consistency scores (Eq. 4)
    if verbose:
        print(f"  Computing pairwise consistency scores (Eq. 4)...")
    C_k_mean, C_k_max = compute_pairwise_consistency(
        covariances, neighbor_indices
    )

    # Step 5: Artifact detection
    floaters, collapsed, severe, thresh_a, thresh_c = detect_artifacts(
        A_k, C_k_mean, anomaly_pct, consistency_pct
    )

    n_floaters  = floaters.sum()
    n_collapsed = collapsed.sum()
    n_severe    = severe.sum()
    n_artifact  = (floaters | collapsed | severe).sum()

    if verbose:
        print(f"\n  === FORENSICS REPORT ===")
        print(f"  Total primitives        : {N:>8,}")
        print(f"  Core (reference) prims  : {core_mask.sum():>8,} "
              f"({100*core_mask.sum()/N:.1f}%)")
        print(f"")
        print(f"  Anomaly score (A_k):")
        print(f"    Mean                  : {A_k.mean():>10.3f}")
        print(f"    Std                   : {A_k.std():>10.3f}")
        print(f"    95th percentile       : {np.percentile(A_k,95):>10.3f}")
        print(f"    99th percentile       : {np.percentile(A_k,99):>10.3f}")
        print(f"")
        print(f"  Consistency score (C_kj):")
        print(f"    Mean                  : {C_k_mean.mean():>10.4f}")
        print(f"    Std                   : {C_k_mean.std():>10.4f}")
        print(f"    95th percentile       : {np.percentile(C_k_mean,95):>10.4f}")
        print(f"")
        print(f"  Detected artifacts (top {100-anomaly_pct}%):")
        print(f"    Floaters              : {n_floaters:>8,} "
              f"({100*n_floaters/N:.2f}%)")
        print(f"    Structural collapse   : {n_collapsed:>8,} "
              f"({100*n_collapsed/N:.2f}%)")
        print(f"    Severe (both)         : {n_severe:>8,} "
              f"({100*n_severe/N:.2f}%)")
        print(f"    Total flagged         : {n_artifact:>8,} "
              f"({100*n_artifact/N:.2f}%)")

    return {
        'n_gaussians'    : N,
        'A_k'            : A_k,
        'C_k_mean'       : C_k_mean,
        'C_k_max'        : C_k_max,
        'core_mask'      : core_mask,
        'floaters'       : floaters,
        'collapsed'      : collapsed,
        'severe'         : severe,
        'n_floaters'     : int(n_floaters),
        'n_collapsed'    : int(n_collapsed),
        'n_severe'       : int(n_severe),
        'n_artifact'     : int(n_artifact),
        'thresh_anomaly' : thresh_a,
        'thresh_consist' : thresh_c,
        'mu_hat'         : mu_hat,
        'sigma_hat'      : sigma_hat,
    }


# ---------------------------------------------------------------------------
# 7. COMPARE DISTORTED VS REFERENCE
# ---------------------------------------------------------------------------

def compare_scenes(reference_ply, distorted_ply, k_neighbors=8):
    """
    Run forensics on both a reference and distorted scene and compare.
    Shows how artifact density increases with distortion.

    Args:
        reference_ply : path to clean/reference .ply
        distorted_ply : path to distorted .ply
    """
    print("\n" + "="*60)
    print("SCENE COMPARISON: Reference vs Distorted")
    print("="*60)

    print("\n[REFERENCE]")
    ref = run_forensics(reference_ply, k_neighbors=k_neighbors)

    print("\n[DISTORTED]")
    dis = run_forensics(distorted_ply, k_neighbors=k_neighbors)

    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Metric':<35} {'Reference':>12} {'Distorted':>12} {'Change':>10}")
    print("-"*72)

    def pct_change(a, b):
        if a == 0:
            return 'N/A'
        return f"{100*(b-a)/a:+.1f}%"

    metrics = [
        ("Total primitives",
         ref['n_gaussians'], dis['n_gaussians']),
        ("Mean anomaly score A_k",
         ref['A_k'].mean(), dis['A_k'].mean()),
        ("Mean consistency C_kj",
         ref['C_k_mean'].mean(), dis['C_k_mean'].mean()),
        ("Floaters detected",
         ref['n_floaters'], dis['n_floaters']),
        ("Structural collapse",
         ref['n_collapsed'], dis['n_collapsed']),
        ("Total artifacts",
         ref['n_artifact'], dis['n_artifact']),
        ("Artifact rate (%)",
         100*ref['n_artifact']/ref['n_gaussians'],
         100*dis['n_artifact']/dis['n_gaussians']),
    ]

    for name, r_val, d_val in metrics:
        if isinstance(r_val, float):
            print(f"  {name:<33} {r_val:>12.4f} {d_val:>12.4f} "
                  f"{pct_change(r_val, d_val):>10}")
        else:
            print(f"  {name:<33} {r_val:>12,} {d_val:>12,} "
                  f"{pct_change(r_val, d_val):>10}")

    return ref, dis


# ---------------------------------------------------------------------------
# 8. MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os

    if len(sys.argv) == 3:
        # Compare mode: python forensics.py reference.ply distorted.ply
        ref_path = sys.argv[1]
        dis_path = sys.argv[2]
        compare_scenes(ref_path, dis_path)

    elif len(sys.argv) == 2:
        # Single file mode
        run_forensics(sys.argv[1])

    else:
        # Demo on uploaded test file
        test_file = "/mnt/user-data/uploads/bottle_25.ply"
        if os.path.exists(test_file):
            run_forensics(test_file)
        else:
            print("Usage:")
            print("  python forensics.py scene.ply")
            print("  python forensics.py reference.ply distorted.ply")

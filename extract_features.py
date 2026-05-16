"""
MVG Feature Extraction for 3DGS .ply files
==========================================
Extracts Multivariate Gaussian statistical features from 3D Gaussian
Splatting primitives for No-Reference Quality Assessment.

Author: Bishr Omer
Framework: MVG-based NR-IQA extended to 3DGS (MCASF)
"""

import numpy as np
import struct
from scipy.linalg import logm, norm
from scipy.stats import skew, kurtosis
import os


# ---------------------------------------------------------------------------
# 1. PLY LOADER
# ---------------------------------------------------------------------------

def load_ply(filepath):
    """
    Load a 3DGS .ply file and return attributes as a numpy array.

    Returns:
        props  : list of attribute names
        data   : (N_gaussians, N_attrs) float32 array
    """
    with open(filepath, "rb") as f:
        header_lines = []
        while True:
            line = f.readline().decode("utf-8", errors="replace").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        props = [l.split()[-1] for l in header_lines if l.startswith("property float")]
        n_vertices = int(
            [l for l in header_lines if l.startswith("element vertex")][0].split()[-1]
        )
        n_props = len(props)
        data = np.frombuffer(f.read(), dtype=np.float32).reshape(n_vertices, n_props)

    return props, data


def parse_attributes(props, data):
    """
    Extract key attribute groups from raw .ply data.

    Returns a dict with:
        positions  : (N, 3)  x, y, z
        scales     : (N, 3)  log-scale s0, s1, s2
        rotations  : (N, 4)  quaternion r0..r3
        opacities  : (N,)    pre-sigmoid opacity
        dc_color   : (N, 3)  f_dc_0..2
        sh_rest    : (N, 45) f_rest_0..44 spherical harmonics
        covariances: (N, 3, 3) reconstructed 3D covariance per Gaussian
        n_gaussians: int, total number of Gaussian primitives
    """
    idx = {p: i for i, p in enumerate(props)}

    positions  = data[:, [idx['x'],       idx['y'],       idx['z']]]
    scales     = data[:, [idx['scale_0'], idx['scale_1'], idx['scale_2']]]
    rotations  = data[:, [idx['rot_0'],   idx['rot_1'],   idx['rot_2'],   idx['rot_3']]]
    opacities  = data[:, idx['opacity']]
    dc_color   = data[:, [idx['f_dc_0'],  idx['f_dc_1'],  idx['f_dc_2']]]

    # SH rest coefficients (f_rest_0 .. f_rest_44)
    sh_cols = [idx[f'f_rest_{i}'] for i in range(45) if f'f_rest_{i}' in idx]
    sh_rest = data[:, sh_cols] if sh_cols else np.zeros((data.shape[0], 0), dtype=np.float32)

    # Reconstruct per-Gaussian 3x3 covariance from scale + rotation
    covariances = _build_covariance_matrices(scales, rotations)

    return {
        'positions':   positions,
        'scales':      scales,
        'rotations':   rotations,
        'opacities':   opacities,
        'dc_color':    dc_color,
        'sh_rest':     sh_rest,
        'covariances': covariances,
        'n_gaussians': data.shape[0],
    }


def _quat_to_rotation_matrix(q):
    """
    Convert quaternion (w, x, y, z) -> 3x3 rotation matrix.
    Vectorized over N quaternions. Input shape: (N, 4)
    """
    # Normalize
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    N = q.shape[0]
    R = np.zeros((N, 3, 3), dtype=np.float32)

    R[:, 0, 0] = 1 - 2*(y**2 + z**2)
    R[:, 0, 1] = 2*(x*y - w*z)
    R[:, 0, 2] = 2*(x*z + w*y)
    R[:, 1, 0] = 2*(x*y + w*z)
    R[:, 1, 1] = 1 - 2*(x**2 + z**2)
    R[:, 1, 2] = 2*(y*z - w*x)
    R[:, 2, 0] = 2*(x*z - w*y)
    R[:, 2, 1] = 2*(y*z + w*x)
    R[:, 2, 2] = 1 - 2*(x**2 + y**2)

    return R


def _build_covariance_matrices(scales, rotations):
    """
    Reconstruct 3x3 covariance matrix per Gaussian.
    Sigma = R * S * S^T * R^T  where S = diag(exp(scale))
    Input:
        scales    : (N, 3) log-scale
        rotations : (N, 4) quaternion
    Output:
        covs : (N, 3, 3)
    """
    S = np.exp(scales)                          # actual scale values (N, 3)
    R = _quat_to_rotation_matrix(rotations)     # (N, 3, 3)

    # Build diagonal scale matrix per Gaussian and compute R S S^T R^T
    # Using einsum for efficiency
    RS = R * S[:, np.newaxis, :]                # (N, 3, 3) each col scaled
    covs = np.einsum('nij,nkj->nik', RS, RS)    # (N, 3, 3)

    return covs


# ---------------------------------------------------------------------------
# 2. MVG FEATURE EXTRACTION
# ---------------------------------------------------------------------------

def extract_mvg_features(attrs):
    """
    Extract a fixed-length MVG feature vector from parsed 3DGS attributes.

    Feature groups:
        A. Scale distribution statistics         (18 features)
        B. Opacity distribution statistics       (6 features)
        C. Covariance eigenvalue statistics      (18 features)
        D. Covariance anisotropy features        (6 features)
        E. Spatial density features              (6 features)
        F. Color distribution statistics         (9 features)
        G. Scene-level MVG descriptor            (6 features)
        H. Gaussian count & density              (4 features)  [NEW - Downsampling]
        I. SH coefficient statistics             (12 features) [NEW - Color distortion]
        J. Positional noise indicators           (8 features)  [NEW - Gaussian noise]
        K. Nearest-neighbor distance stats       (5 features)  [NEW - Gaussian noise]
    Total: 98 features
    """
    features = []
    feat_names = []

    scales      = attrs['scales']        # (N, 3) log-scale
    opacities   = attrs['opacities']     # (N,)
    covariances = attrs['covariances']   # (N, 3, 3)
    positions   = attrs['positions']     # (N, 3)
    dc_color    = attrs['dc_color']      # (N, 3)
    sh_rest     = attrs['sh_rest']       # (N, 45)
    n_gaussians = attrs['n_gaussians']   # int

    # Sigmoid opacity for real [0,1] values
    sig_opacity = 1.0 / (1.0 + np.exp(-opacities.astype(np.float64)))
    # Exp scales for actual sizes
    real_scales = np.exp(scales.astype(np.float64))
    pos = positions.astype(np.float64)

    # -- A. Scale distribution statistics (6 stats x 3 axes = 18) ----------
    for axis in range(3):
        s = real_scales[:, axis]
        features += [s.mean(), s.std(), float(skew(s)), float(kurtosis(s)),
                     np.percentile(s, 25), np.percentile(s, 75)]
        feat_names += [f'scale_{axis}_{stat}' for stat in
                       ['mean','std','skew','kurt','q25','q75']]

    # -- B. Opacity distribution statistics (6) -----------------------------
    features += [
        sig_opacity.mean(),
        sig_opacity.std(),
        float(skew(sig_opacity)),
        float(kurtosis(sig_opacity)),
        np.percentile(sig_opacity, 10),
        np.percentile(sig_opacity, 90),
    ]
    feat_names += ['opacity_mean','opacity_std','opacity_skew','opacity_kurt',
                   'opacity_p10','opacity_p90']

    # -- C. Covariance eigenvalue statistics (6 stats x 3 eigenvals = 18) --
    eigvals = np.linalg.eigvalsh(covariances.astype(np.float64))  # (N, 3)
    eigvals = np.abs(eigvals)

    for ev_idx in range(3):
        ev = eigvals[:, ev_idx]
        features += [ev.mean(), ev.std(), float(skew(ev)), float(kurtosis(ev)),
                     np.percentile(ev, 25), np.percentile(ev, 75)]
        feat_names += [f'eigval_{ev_idx}_{stat}' for stat in
                       ['mean','std','skew','kurt','q25','q75']]

    # -- D. Covariance anisotropy features (6) ------------------------------
    ev_max = eigvals[:, 2] + 1e-10
    ev_min = eigvals[:, 0] + 1e-10
    anisotropy = ev_max / ev_min

    traces = np.trace(covariances.astype(np.float64), axis1=1, axis2=2)
    dets   = np.linalg.det(covariances.astype(np.float64))

    features += [
        anisotropy.mean(),
        anisotropy.std(),
        float(skew(anisotropy)),
        traces.mean(),
        np.log(np.abs(dets) + 1e-10).mean(),
        np.log(np.abs(dets) + 1e-10).std(),
    ]
    feat_names += ['anisotropy_mean','anisotropy_std','anisotropy_skew',
                   'trace_mean','logdet_mean','logdet_std']

    # -- E. Spatial density features (6) ------------------------------------
    scene_center = pos.mean(axis=0)
    dists_from_center = np.linalg.norm(pos - scene_center, axis=1)

    features += [
        dists_from_center.mean(),
        dists_from_center.std(),
        np.percentile(dists_from_center, 90),
        pos[:, 0].std(),
        pos[:, 1].std(),
        pos[:, 2].std(),
    ]
    feat_names += ['spatial_dist_mean','spatial_dist_std','spatial_dist_p90',
                   'spread_x','spread_y','spread_z']

    # -- F. Color distribution statistics (3 stats x 3 channels = 9) -------
    for ch in range(3):
        c = dc_color[:, ch].astype(np.float64)
        features += [c.mean(), c.std(), float(skew(c))]
        feat_names += [f'color_{ch}_{stat}' for stat in ['mean','std','skew']]

    # -- G. Scene-level MVG descriptor (6) ----------------------------------
    joint = np.column_stack([real_scales, sig_opacity[:, np.newaxis]])
    scene_mean = joint.mean(axis=0)
    scene_cov  = np.cov(joint.T)

    features += list(scene_mean[:3])
    features += [scene_cov[0,0], scene_cov[1,1], scene_cov[2,2]]
    feat_names += ['scene_mvg_mean_s0','scene_mvg_mean_s1','scene_mvg_mean_s2',
                   'scene_mvg_cov_s0','scene_mvg_cov_s1','scene_mvg_cov_s2']

    # -- H. Gaussian count & density (4) [Downsampling] ---------------------
    # Bounding box volume for density
    bbox_min = pos.min(axis=0)
    bbox_max = pos.max(axis=0)
    bbox_dims = bbox_max - bbox_min + 1e-8
    bbox_vol  = float(np.prod(bbox_dims))

    features += [
        float(n_gaussians),                        # raw count
        np.log(float(n_gaussians) + 1),            # log count (more linear)
        float(n_gaussians) / bbox_vol,             # point density per unit vol
        np.log(float(n_gaussians) / bbox_vol + 1), # log density
    ]
    feat_names += ['n_gaussians','log_n_gaussians',
                   'point_density','log_point_density']

    # -- I. SH coefficient statistics (12) [Color distortion] ---------------
    # SH coefficients encode view-dependent color; perturbation changes their
    # distribution significantly
    if sh_rest.shape[1] > 0:
        sh = sh_rest.astype(np.float64)
        sh_flat = sh.flatten()
        # Per-band statistics (3 bands of 15 coefficients each)
        for band in range(3):
            band_sh = sh[:, band*15:(band+1)*15].flatten()
            features += [band_sh.mean(), band_sh.std(),
                         float(skew(band_sh)), float(kurtosis(band_sh))]
            feat_names += [f'sh_band{band}_{s}' for s in ['mean','std','skew','kurt']]
    else:
        features += [0.0] * 12
        feat_names += [f'sh_band{b}_{s}' for b in range(3)
                       for s in ['mean','std','skew','kurt']]

    # -- J. Positional noise indicators (8) [Gaussian noise] ----------------
    # Gaussian noise adds jitter to positions; this shows up as:
    # 1. Increased local variance (neighbors further apart)
    # 2. Higher kurtosis in position distributions
    # 3. Disrupted smoothness of the scale field

    for axis in range(3):
        p = pos[:, axis]
        features += [float(kurtosis(p)), float(skew(p))]
        feat_names += [f'pos_{axis}_kurt', f'pos_{axis}_skew']

    # Scale field smoothness: std of scale differences between sorted positions
    sorted_idx = np.argsort(pos[:, 0])
    scale_diffs = np.diff(real_scales[sorted_idx, :], axis=0)
    features += [scale_diffs.std()]
    feat_names += ['scale_field_roughness']

    # Opacity field roughness
    opacity_diffs = np.diff(sig_opacity[sorted_idx])
    features += [opacity_diffs.std()]
    feat_names += ['opacity_field_roughness']

    # -- K. Nearest-neighbor distance stats (5) [Gaussian noise] ------------
    # Sample 2000 points for efficiency
    sample_size = min(2000, n_gaussians)
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(n_gaussians, size=sample_size, replace=False)
    pos_sample = pos[sample_idx]

    # Pairwise distances within sample - use efficient batch approach
    # Compute distances to 5 nearest neighbors via chunked approach
    chunk = min(200, sample_size)
    nn_dists = []
    for i in range(0, sample_size, chunk):
        batch = pos_sample[i:i+chunk]
        diffs = pos_sample[np.newaxis, :, :] - batch[:, np.newaxis, :]
        dists = np.sqrt((diffs**2).sum(axis=2))
        # Exclude self (dist=0): get 2nd smallest
        dists.sort(axis=1)
        nn_dists.append(dists[:, 1])  # nearest neighbor (not self)

    nn_dists = np.concatenate(nn_dists)
    features += [
        nn_dists.mean(),
        nn_dists.std(),
        float(skew(nn_dists)),
        np.percentile(nn_dists, 10),
        np.percentile(nn_dists, 90),
    ]
    feat_names += ['nn_dist_mean','nn_dist_std','nn_dist_skew',
                   'nn_dist_p10','nn_dist_p90']

    return np.array(features, dtype=np.float64), feat_names


# ---------------------------------------------------------------------------
# 3. MAIN PIPELINE FUNCTION
# ---------------------------------------------------------------------------

def process_ply_file(filepath):
    """
    Full pipeline: load .ply -> parse attributes -> extract MVG features.

    Returns:
        features  : (69,) numpy array
        feat_names: list of 69 feature names
    """
    props, data = load_ply(filepath)
    attrs = parse_attributes(props, data)
    features, feat_names = extract_mvg_features(attrs)
    return features, feat_names


def process_dataset(ply_dir, mos_file=None):
    """
    Process all .ply files in a directory.

    Args:
        ply_dir  : path to folder containing .ply files
        mos_file : optional path to MOS scores file (xlsx or csv)

    Returns:
        X         : (N_files, 69) feature matrix
        filenames : list of filenames
        feat_names: list of feature names
    """
    ply_files = sorted([f for f in os.listdir(ply_dir) if f.endswith('.ply')])
    print(f"Found {len(ply_files)} .ply files in {ply_dir}")

    X = []
    filenames = []
    feat_names = None

    for i, fname in enumerate(ply_files):
        fpath = os.path.join(ply_dir, fname)
        try:
            feats, names = process_ply_file(fpath)
            X.append(feats)
            filenames.append(fname)
            if feat_names is None:
                feat_names = names
            print(f"  [{i+1}/{len(ply_files)}] {fname} -> {feats.shape[0]} features")
        except Exception as e:
            print(f"  [ERROR] {fname}: {e}")

    X = np.array(X)
    return X, filenames, feat_names


# ---------------------------------------------------------------------------
# 4. QUICK TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    test_file = sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/uploads/bottle_25.ply"

    print(f"Testing on: {test_file}")
    features, feat_names = process_ply_file(test_file)

    print(f"\nFeature vector length: {len(features)}")
    print(f"\nSample features:")
    for name, val in zip(feat_names[:15], features[:15]):
        print(f"  {name:35s}: {val:.6f}")
    print(f"  ... ({len(features) - 15} more)")
    print(f"\nFeature vector stats:")
    print(f"  min={features.min():.4f}, max={features.max():.4f}, "
          f"mean={features.mean():.4f}, std={features.std():.4f}")
    print(f"\nNaN count: {np.isnan(features).sum()}")
    print(f"Inf count: {np.isinf(features).sum()}")

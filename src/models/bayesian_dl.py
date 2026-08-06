"""
Bayesian Deep Learning — MC Dropout, Variational BNN, Deep Ensembles
=====================================================================
Implements Section A4 (BDL) with additions:
  - Time-aware TimeSeriesSplit cross-validation with purge buffer
  - SHAP attributions for regime predictions (Section D7)
  - Full epistemic/aleatoric uncertainty decomposition
  - Calibration diagnostics (ECE, reliability diagram)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

# ── TensorFlow ────────────────────────────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    import tensorflow_probability as tfp
    tfd  = tfp.distributions
    tfpl = tfp.layers
    TF_AVAILABLE = True
except ImportError:
    tf = tfp = tfd = tfpl = None
    TF_AVAILABLE = False
    print("[bayesian_dl] tensorflow/tensorflow-probability not available")

# ── SHAP ─────────────────────────────────────────────────────────────────────
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    shap = None
    SHAP_AVAILABLE = False
    print("[bayesian_dl] shap not available — SHAP attributions will be skipped")


# =============================================================================
# Section A4.2 — MC Dropout Regime Classifier
# =============================================================================

def build_mc_dropout_regime_classifier(
    input_dim: int,
    n_regimes: int = 5,
    dropout_rate: float = 0.3,
    hidden_units: tuple = (128, 64, 32),
) -> "tf.keras.Model":
    """
    Regime classifier with permanently-enabled dropout for MC inference.

    dropout active at *inference* time (training=True in forward pass)
    approximates Bayesian weight uncertainty (Gal & Ghahramani, 2016).

    Parameters
    ----------
    input_dim    : number of input features
    n_regimes    : number of output classes
    dropout_rate : dropout probability
    hidden_units : sizes of hidden Dense layers

    Returns
    -------
    tf.keras.Model (compiled, ready for training)
    """
    if not TF_AVAILABLE:
        raise ImportError("tensorflow required: pip install tensorflow>=2.15")

    inputs = layers.Input(shape=(input_dim,), name="features")
    x = inputs
    for i, units in enumerate(hidden_units):
        x = layers.Dense(units, activation="relu", name=f"dense_{i}")(x)
        x = layers.Dropout(dropout_rate, name=f"dropout_{i}")(x, training=True)

    outputs = layers.Dense(n_regimes, activation="softmax", name="regime_probs")(x)

    model = Model(inputs, outputs, name="mc_dropout_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def mc_predict(model, X, n_samples: int = 200) -> tuple:
    """
    Generate predictive distribution via repeated stochastic passes.

    Parameters
    ----------
    model     : MC-dropout model
    X         : (N, D) feature matrix
    n_samples : number of MC forward passes

    Returns
    -------
    mean      : (N, K) mean regime probabilities
    epistemic : (N, K) std across samples (model uncertainty)
    aleatoric : (N, K) mean per-sample Bernoulli variance (data noise)
    all_preds : (n_samples, N, K) full predictive distribution
    """
    if not TF_AVAILABLE:
        raise ImportError("tensorflow required")

    X_tf = tf.constant(X, dtype=tf.float32)
    all_preds = np.stack([
        model(X_tf, training=True).numpy() for _ in range(n_samples)
    ])
    mean      = all_preds.mean(axis=0)
    epistemic = all_preds.std(axis=0)
    aleatoric = (all_preds * (1 - all_preds)).mean(axis=0)
    return mean, epistemic, aleatoric, all_preds


# =============================================================================
# Section A4.3 — Variational BNN (DenseFlipout)
# =============================================================================

def build_variational_regime_classifier(
    input_dim: int,
    n_regimes: int = 5,
    train_size: int = 2000,
) -> "tf.keras.Model":
    """
    Mean-field variational inference BNN using DenseFlipout layers.

    Optimises the ELBO: E[log p(y|x,w)] - KL(q(w)||p(w))
    KL weight = 1/N_train to match standard VB convention.
    """
    if not TF_AVAILABLE:
        raise ImportError("tensorflow and tensorflow-probability required")

    kl_weight = 1.0 / max(train_size, 1)

    def kl_fn(q, p, _):
        return kl_weight * tfd.kl_divergence(q, p)

    inputs = tf.keras.Input(shape=(input_dim,), name="features")
    x = tfpl.DenseFlipout(128, activation="relu",
                          kernel_divergence_fn=kl_fn, name="vb_dense_0")(inputs)
    x = tfpl.DenseFlipout(64,  activation="relu",
                          kernel_divergence_fn=kl_fn, name="vb_dense_1")(x)
    logits = tfpl.DenseFlipout(n_regimes,
                               kernel_divergence_fn=kl_fn, name="vb_logits")(x)
    outputs = tfpl.OneHotCategorical(n_regimes, name="regime_dist")(logits)

    model = tf.keras.Model(inputs, outputs, name="vi_bnn_classifier")
    nll = lambda y, rv_y: -rv_y.log_prob(y)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=nll, metrics=["accuracy"])
    return model


# =============================================================================
# Section A4.4 — Deep Ensembles (M independent models)
# =============================================================================

def train_deep_ensemble(
    build_fn,
    X_train: np.ndarray,
    y_train: np.ndarray,
    M: int = 10,
    epochs: int = 80,
    batch_size: int = 64,
) -> list:
    """
    Train M independent regime classifiers with different random seeds.

    Parameters
    ----------
    build_fn  : callable() → tf.keras.Model
    X_train   : (N, D) training features
    y_train   : (N,)  integer regime labels
    M         : ensemble size (default 10 per spec)
    epochs    : training epochs per member

    Returns
    -------
    list of M fitted models
    """
    if not TF_AVAILABLE:
        raise ImportError("tensorflow required")

    ensemble = []
    for m in range(M):
        tf.random.set_seed(m * 17 + 3)
        np.random.seed(m * 17 + 3)
        model = build_fn()
        model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            validation_split=0.1,
        )
        ensemble.append(model)

    return ensemble


def ensemble_predict(ensemble: list, X: np.ndarray) -> tuple:
    """
    Generate predictive distribution from a deep ensemble.

    Returns
    -------
    mean      : (N, K) mean regime probabilities
    epistemic : (N, K) std across members (model disagreement)
    aleatoric : (N, K) mean Bernoulli variance within members (data noise)
    """
    if not TF_AVAILABLE:
        raise ImportError("tensorflow required")

    preds = np.stack([m.predict(X, verbose=0) for m in ensemble])  # (M, N, K)
    mean      = preds.mean(axis=0)
    epistemic = preds.std(axis=0)
    aleatoric = (preds * (1 - preds)).mean(axis=0)
    return mean, epistemic, aleatoric


# =============================================================================
# Section A4.6 — Time-Aware Cross-Validation with Purge Buffer
# =============================================================================

def time_aware_cv(
    X: np.ndarray,
    y: np.ndarray,
    build_fn,
    n_splits: int = 5,
    purge_days: int = 21,
    epochs: int = 80,
) -> pd.DataFrame:
    """
    Walk-forward time-series cross-validation with purge buffer.

    The purge buffer prevents information leakage caused by overlapping
    return windows between train and test splits.

    Parameters
    ----------
    X          : (N, D) feature matrix (chronologically ordered)
    y          : (N,)  integer labels
    build_fn   : callable() → model
    n_splits   : number of CV folds
    purge_days : gap between train end and test start
    epochs     : training epochs per fold

    Returns
    -------
    pd.DataFrame — per-fold accuracy, epistemic/aleatoric stats
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rows = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        # Apply purge buffer
        purged_test_idx = test_idx[test_idx > train_idx[-1] + purge_days]
        if len(purged_test_idx) < 10:
            continue

        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[purged_test_idx], y[purged_test_idx]

        if not TF_AVAILABLE:
            rows.append({"fold": fold, "accuracy": np.nan,
                         "epi_mean": np.nan, "ale_mean": np.nan})
            continue

        ensemble = train_deep_ensemble(build_fn, X_tr, y_tr, M=5, epochs=epochs)
        mean, epi, ale = ensemble_predict(ensemble, X_te)
        acc = float((mean.argmax(1) == y_te).mean())

        rows.append({
            "fold": fold,
            "n_train": len(train_idx),
            "n_test": len(purged_test_idx),
            "accuracy": round(acc, 4),
            "epi_mean": round(float(epi.mean()), 4),
            "ale_mean": round(float(ale.mean()), 4),
            "epistemic_aleatoric_ratio": round(float(epi.mean() / (ale.mean() + 1e-9)), 3),
        })

    return pd.DataFrame(rows)


# =============================================================================
# SHAP Feature Attributions (Section D7)
# =============================================================================

def compute_shap_attributions(
    model,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    feature_names: list,
    n_mc_passes: int = 50,
) -> pd.DataFrame:
    """
    Compute SHAP values for a Bayesian regime classifier.

    Uses DeepExplainer on the mean prediction (averaged over MC passes)
    to explain feature contributions on selected dates.

    Parameters
    ----------
    model         : MC-dropout TF model
    X_background  : (n_bg, D) background data (calibration set)
    X_explain     : (N, D)   data to explain
    feature_names : list of D feature name strings
    n_mc_passes   : MC draws for mean prediction

    Returns
    -------
    pd.DataFrame (N, D) — mean SHAP value per feature per sample
    """
    if not SHAP_AVAILABLE:
        print("[bayesian_dl] shap not installed — returning zero attributions")
        return pd.DataFrame(
            np.zeros((len(X_explain), len(feature_names))),
            columns=feature_names,
        )

    if not TF_AVAILABLE:
        raise ImportError("tensorflow required for SHAP attributions")

    # Build a deterministic wrapper: average over MC passes for SHAP
    def mean_predict(x):
        preds = np.stack([
            model(tf.constant(x, dtype=tf.float32), training=True).numpy()
            for _ in range(n_mc_passes)
        ])
        return preds.mean(0)

    explainer = shap.KernelExplainer(mean_predict, X_background)
    shap_values = explainer.shap_values(X_explain, nsamples=100, silent=True)

    # Average SHAP across regime classes (list of K arrays each (N, D))
    if isinstance(shap_values, list):
        mean_shap = np.mean(np.abs(np.stack(shap_values, axis=0)), axis=0)
    else:
        mean_shap = np.abs(shap_values)

    return pd.DataFrame(mean_shap, columns=feature_names)


def top_shap_features(
    shap_df: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """Return the most influential features by mean absolute SHAP."""
    return (
        shap_df.abs().mean().sort_values(ascending=False)
        .head(top_n)
        .reset_index()
        .rename(columns={"index": "feature", 0: "mean_abs_shap"})
    )


# =============================================================================
# Calibration Diagnostics (ECE + Reliability Diagram)
# =============================================================================

def expected_calibration_error(
    probs_max: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error (ECE) — measures alignment between
    predicted confidence and empirical accuracy.

    Parameters
    ----------
    probs_max : (N,) max regime probability per sample (model confidence)
    correct   : (N,) binary array — 1 if predicted class is true class
    n_bins    : number of confidence bins

    Returns
    -------
    float — ECE (0 = perfect calibration)
    """
    bins = np.linspace(0, 1, n_bins + 1)
    n_total = len(probs_max)
    ece = 0.0

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs_max > lo) & (probs_max <= hi)
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        acc  = float(correct[mask].mean())
        conf = float(probs_max[mask].mean())
        ece += (n_bin / n_total) * abs(acc - conf)

    return float(ece)


def reliability_diagram_data(
    probs_max: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Compute binned accuracy vs confidence for a reliability diagram.

    Returns
    -------
    pd.DataFrame with: bin_mid, accuracy, confidence, count, gap
    """
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs_max > lo) & (probs_max <= hi)
        n = mask.sum()
        if n == 0:
            rows.append({"bin_mid": (lo + hi) / 2, "accuracy": 0,
                         "confidence": (lo + hi) / 2, "count": 0, "gap": 0})
        else:
            acc  = float(correct[mask].mean())
            conf = float(probs_max[mask].mean())
            rows.append({
                "bin_mid":    round((lo + hi) / 2, 2),
                "accuracy":   round(acc, 4),
                "confidence": round(conf, 4),
                "count":      int(n),
                "gap":        round(abs(acc - conf), 4),
            })

    return pd.DataFrame(rows)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.data.synthetic_data import generate_synthetic_market_data
    from src.data.feature_engineering import engineer_regime_features

    print("Generating data...")
    df, true_regimes = generate_synthetic_market_data(seed=42)
    features = engineer_regime_features(df)

    aligned_regimes = true_regimes[len(true_regimes) - len(features):]
    X = features.values.astype(np.float32)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = aligned_regimes

    input_dim = X.shape[1]

    if TF_AVAILABLE:
        build_fn = lambda: build_mc_dropout_regime_classifier(input_dim, n_regimes=5)

        print(f"\nBuilding MC-Dropout classifier (input_dim={input_dim})...")
        model = build_fn()
        print(model.summary())

        print("\nFitting on first 1000 samples...")
        model.fit(X[:1000], y[:1000], epochs=10, batch_size=64, verbose=1)

        print("\nMC Prediction (200 passes, first 20 samples)...")
        mean, epi, ale, _ = mc_predict(model, X[:20], n_samples=200)
        print(f"  Mean probs (sample 0):      {np.round(mean[0], 4)}")
        print(f"  Epistemic std (sample 0):   {np.round(epi[0], 4)}")
        print(f"  Aleatoric std (sample 0):   {np.round(ale[0], 4)}")

        probs_max = mean.max(axis=1)
        correct   = (mean.argmax(1) == y[:20]).astype(float)
        ece = expected_calibration_error(probs_max, correct)
        print(f"\n  ECE (20-sample demo): {ece:.4f}")

        if SHAP_AVAILABLE:
            print("\n  Computing SHAP attributions (demo)...")
            feat_names = features.columns.tolist()
            shap_df = compute_shap_attributions(
                model, X[:50], X[:10], feat_names, n_mc_passes=10
            )
            top = top_shap_features(shap_df)
            print(f"  Top 5 features:\n{top.head(5).to_string()}")
    else:
        print("tensorflow not available — skipping live demo")

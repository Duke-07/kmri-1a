# =============================================================================
# R Conformal Prediction — Full Implementation
# =============================================================================
# Implements all conformal variants from Section A6 in R:
#   - Split-conformal classifier (Section A6.2)
#   - Adaptive Prediction Sets / APS (Section A6.3)
#   - Mondrian class-conditional conformal (Section A6.5)
#   - Adaptive Conformal Inference / ACI (Section A6.5)
#   - Brier Score, Ranked Probability Score, ECE
#   - Rolling coverage stability tracking
# =============================================================================

library(tidyverse)

# =============================================================================
# Utility
# =============================================================================

#' Empirical quantile with 'higher' interpolation (finite-sample correction).
quantile_higher <- function(x, q) {
  quantile(x, q, type = 1)  # type=1 is the 'higher' (inverse-ECDF) method
}

# =============================================================================
# Section A6.2 — Split-Conformal Classifier
# =============================================================================

#' Split-conformal prediction sets with marginal coverage >= 1-alpha.
#'
#' @param cal_probs  matrix (n_cal, K) calibration probabilities
#' @param y_cal     integer vector (1-indexed) true labels for calibration
#' @param test_probs matrix (n_test, K) test probabilities
#' @param alpha     miscoverage level (default 0.10 = 90% coverage)
#' @return list(sets, q_hat, coverage, set_sizes)
split_conformal <- function(cal_probs, y_cal, test_probs, alpha = 0.1) {
  n_cal  <- nrow(cal_probs)
  K      <- ncol(cal_probs)
  
  # Non-conformity scores: 1 - p(true class)
  scores <- numeric(n_cal)
  for (i in 1:n_cal) {
    scores[i] <- 1 - cal_probs[i, y_cal[i]]
  }
  
  # Conformal quantile
  q_level <- min(ceiling((n_cal + 1) * (1 - alpha)) / n_cal, 1.0)
  q_hat   <- quantile_higher(scores, q_level)
  
  # Prediction sets
  sets     <- test_probs >= (1 - q_hat)
  coverage <- mean(scores <= q_hat)
  
  list(
    sets      = sets,
    q_hat     = as.numeric(q_hat),
    coverage  = coverage,
    set_sizes = rowSums(sets)
  )
}


# =============================================================================
# Section A6.3 — Adaptive Prediction Sets (APS)
# =============================================================================

#' Adaptive Prediction Sets — more efficient for calibrated models.
#' Score = cumulative probability up to and including the true class rank.
adaptive_prediction_sets <- function(cal_probs, y_cal, test_probs, alpha = 0.1) {
  n_cal <- nrow(cal_probs)
  K     <- ncol(cal_probs)
  
  scores <- numeric(n_cal)
  for (i in 1:n_cal) {
    sorted_idx   <- order(cal_probs[i, ], decreasing = TRUE)
    sorted_probs <- cal_probs[i, sorted_idx]
    cumsums      <- cumsum(sorted_probs)
    true_rank    <- which(sorted_idx == y_cal[i])
    scores[i]    <- cumsums[true_rank]
  }
  
  q_level <- min(ceiling((n_cal + 1) * (1 - alpha)) / n_cal, 1.0)
  q_hat   <- quantile_higher(scores, q_level)
  
  n_test    <- nrow(test_probs)
  pred_sets <- matrix(FALSE, nrow = n_test, ncol = K)
  for (i in 1:n_test) {
    sorted_idx   <- order(test_probs[i, ], decreasing = TRUE)
    sorted_probs <- test_probs[i, sorted_idx]
    cumsums      <- cumsum(sorted_probs)
    in_set       <- cumsums <= q_hat
    in_set[1]    <- TRUE  # always include top class
    pred_sets[i, sorted_idx[in_set]] <- TRUE
  }
  
  list(
    sets      = pred_sets,
    q_hat     = as.numeric(q_hat),
    set_sizes = rowSums(pred_sets)
  )
}


# =============================================================================
# Section A6.5 — Mondrian Conformal (Class-Conditional)
# =============================================================================

#' Separate threshold per regime class — conditional coverage per class.
mondrian_conformal <- function(cal_probs, y_cal, test_probs, alpha = 0.1, K = 5) {
  q_hats <- numeric(K)
  for (k in 1:K) {
    mask_k   <- y_cal == k
    n_k      <- sum(mask_k)
    if (n_k < 5) {
      q_hats[k] <- 1.0
      next
    }
    scores_k <- 1 - cal_probs[mask_k, k]
    q_level  <- min(ceiling((n_k + 1) * (1 - alpha)) / n_k, 1.0)
    q_hats[k] <- quantile_higher(scores_k, q_level)
  }
  
  pred_sets <- matrix(FALSE, nrow = nrow(test_probs), ncol = K)
  for (k in 1:K) {
    pred_sets[, k] <- test_probs[, k] >= (1 - q_hats[k])
  }
  
  list(
    sets      = pred_sets,
    q_hats    = q_hats,
    set_sizes = rowSums(pred_sets)
  )
}


# =============================================================================
# Section A6.5 — Adaptive Conformal Inference (ACI)
# =============================================================================

#' Online ACI: adapt alpha after each realised outcome.
#' Maintains rolling coverage under distribution shift.
#'
#' @param scores_stream  list of (nonconformity_score, q_hat, covered) tuples
#' @param alpha_target   target miscoverage level
#' @param gamma          adaptation step size
adaptive_conformal_inference <- function(scores_stream, alpha_target = 0.1, gamma = 0.01) {
  alpha_t <- alpha_target
  rows    <- list()
  
  for (step in seq_along(scores_stream)) {
    item     <- scores_stream[[step]]
    score    <- item$score
    q_hat    <- item$q_hat
    covered  <- item$covered
    err_t    <- if (covered) 0 else 1
    alpha_t  <- alpha_t + gamma * (alpha_target - err_t)
    alpha_t  <- min(max(alpha_t, 1e-3), 1 - 1e-3)
    rows[[step]] <- data.frame(step = step, alpha_t = alpha_t,
                               covered = covered, q_hat = q_hat)
  }
  
  df <- do.call(rbind, rows)
  # Rolling 50-step coverage
  if (nrow(df) >= 10) {
    df$rolling_coverage_50 <- zoo::rollmean(df$covered, k = min(50, nrow(df)),
                                             fill = NA, align = "right")
  }
  df
}


# =============================================================================
# Section A6.6 — Calibration Diagnostics
# =============================================================================

#' Expected Calibration Error (ECE).
#' Measures alignment between predicted confidence and empirical accuracy.
ece_r <- function(probs_max, correct, n_bins = 10) {
  breaks <- seq(0, 1, length.out = n_bins + 1)
  n      <- length(probs_max)
  ece    <- 0.0
  
  for (i in 1:n_bins) {
    lo   <- breaks[i]
    hi   <- breaks[i + 1]
    mask <- probs_max > lo & probs_max <= hi
    if (sum(mask) == 0) next
    acc  <- mean(correct[mask])
    conf <- mean(probs_max[mask])
    ece  <- ece + (sum(mask) / n) * abs(acc - conf)
  }
  ece
}


#' Reliability diagram data.
reliability_diagram <- function(probs_max, correct, n_bins = 10) {
  breaks <- seq(0, 1, length.out = n_bins + 1)
  rows   <- list()
  
  for (i in 1:n_bins) {
    lo   <- breaks[i]; hi <- breaks[i + 1]
    mask <- probs_max > lo & probs_max <= hi
    n_bin <- sum(mask)
    bin_mid <- (lo + hi) / 2
    if (n_bin == 0) {
      rows[[i]] <- data.frame(bin_mid = bin_mid, accuracy = 0,
                               confidence = bin_mid, count = 0, gap = 0)
    } else {
      acc  <- mean(correct[mask])
      conf <- mean(probs_max[mask])
      rows[[i]] <- data.frame(bin_mid = round(bin_mid, 2),
                               accuracy = round(acc, 4),
                               confidence = round(conf, 4),
                               count = n_bin,
                               gap = round(abs(acc - conf), 4))
    }
  }
  do.call(rbind, rows)
}


#' Multiclass Brier Score.
brier_score_r <- function(probs, y_true, K = 5) {
  onehot <- model.matrix(~ factor(y_true, levels = 1:K) - 1)
  mean(rowSums((probs - onehot)^2))
}


#' Ranked Probability Score (proper scoring rule for ordinal outcomes).
rps_r <- function(probs, y_true, K = 5) {
  onehot   <- model.matrix(~ factor(y_true, levels = 1:K) - 1)
  cdf_probs <- t(apply(probs, 1, cumsum))
  cdf_true  <- t(apply(onehot, 1, cumsum))
  mean(rowSums((cdf_probs - cdf_true)^2)) / (K - 1)
}


#' Rolling conformal coverage stability.
rolling_coverage <- function(scores, q_hat, window = 252) {
  covered <- as.numeric(scores <= q_hat)
  if (!requireNamespace("zoo", quietly = TRUE)) {
    return(data.frame(t = seq_along(covered), covered = covered))
  }
  rolling <- zoo::rollmean(covered, k = min(window, length(covered)),
                            fill = NA, align = "right")
  data.frame(t = seq_along(covered), covered = covered, rolling_coverage = rolling)
}


# =============================================================================
# Python vs R Reconciliation
# =============================================================================

#' Compute numerical reconciliation between Python and R conformal outputs.
#' @param python_sets  boolean matrix (T, K) from Python
#' @param r_sets       boolean matrix (T, K) from R
reconcile_conformal <- function(python_sets, r_sets) {
  agreement <- colMeans(python_sets == r_sets)
  set_size_diff <- mean(rowSums(python_sets)) - mean(rowSums(r_sets))
  
  list(
    per_regime_agreement = round(agreement, 4),
    mean_agreement       = round(mean(agreement), 4),
    set_size_diff        = round(set_size_diff, 3),
    sets_identical       = all(python_sets == r_sets)
  )
}


# =============================================================================
# Main demonstration
# =============================================================================

main_conformal <- function() {
  message("\n=== R Conformal Prediction Module ===")
  message("CIN: U62012MH2023PTC410415\n")
  
  set.seed(42)
  K      <- 5
  n_cal  <- 300
  n_test <- 100
  
  # Mock probabilities
  cal_probs  <- t(apply(matrix(rexp(n_cal * K), n_cal, K), 1, function(x) x / sum(x)))
  test_probs <- t(apply(matrix(rexp(n_test * K), n_test, K), 1, function(x) x / sum(x)))
  y_cal      <- sample(1:K, n_cal, replace = TRUE)
  
  # Split-conformal
  sc <- split_conformal(cal_probs, y_cal, test_probs, alpha = 0.10)
  message(sprintf("Split-Conformal: q_hat=%.4f | cal_coverage=%.3f | avg_set_size=%.2f",
                  sc$q_hat, sc$coverage, mean(sc$set_sizes)))
  
  # APS
  aps <- adaptive_prediction_sets(cal_probs, y_cal, test_probs, alpha = 0.10)
  message(sprintf("APS:             q_hat=%.4f | avg_set_size=%.2f",
                  aps$q_hat, mean(aps$set_sizes)))
  
  # Mondrian
  mond <- mondrian_conformal(cal_probs, y_cal, test_probs, alpha = 0.10, K = K)
  message(sprintf("Mondrian:        q_hats: %s",
                  paste(round(mond$q_hats, 3), collapse = ", ")))
  
  # Calibration metrics
  probs_max <- apply(cal_probs, 1, max)
  correct   <- as.numeric(apply(cal_probs, 1, which.max) == y_cal)
  
  ece_val <- ece_r(probs_max, correct)
  bs_val  <- brier_score_r(cal_probs, y_cal, K = K)
  rps_val <- rps_r(cal_probs, y_cal, K = K)
  
  message(sprintf("\nCalibration: ECE=%.4f | Brier=%.4f | RPS=%.4f",
                  ece_val, bs_val, rps_val))
  
  rel <- reliability_diagram(probs_max, correct)
  message("\nReliability Diagram:")
  print(rel)
  
  message("\n=== R Conformal Module Complete ===")
  message("CIN: U62012MH2023PTC410415")
}

if (!interactive()) {
  tryCatch(main_conformal(), error = function(e) {
    message("Error in main_conformal: ", e$message)
  })
}

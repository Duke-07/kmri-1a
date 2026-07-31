# Conformal prediction wrappers in R
# Requires: conformalInference (if available) or manual implementation

split_conformal_classifier <- function(cal_probs, y_cal, test_probs, alpha = 0.1) {
  # cal_probs: matrix of calibration probabilities
  # y_cal: integer labels (1-indexed in R)
  
  n_cal <- nrow(cal_probs)
  
  # Compute non-conformity scores (1 - P(y_true))
  cal_scores <- numeric(n_cal)
  for(i in 1:n_cal) {
    cal_scores[i] <- 1 - cal_probs[i, y_cal[i]]
  }
  
  # Quantile calculation
  q_level <- ceiling((n_cal + 1) * (1 - alpha)) / n_cal
  q_hat <- quantile(cal_scores, q_level, type=1)
  
  # Prediction sets
  pred_sets <- test_probs >= (1 - q_hat)
  
  return(list(sets = pred_sets, q_hat = q_hat))
}

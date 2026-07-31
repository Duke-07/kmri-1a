# conformal prediction wrappers in r
# requires: conformalinference (if available) or manual implementation

split_conformal_classifier <- function(cal_probs, y_cal, test_probs, alpha = 0.1) {
  # cal_probs: matrix of calibration probabilities
  # y_cal: integer labels (1-indexed in r)
  
  n_cal <- nrow(cal_probs)
  
  # compute non-conformity scores (1 - p(y_true))
  cal_scores <- numeric(n_cal)
  for(i in 1:n_cal) {
    cal_scores[i] <- 1 - cal_probs[i, y_cal[i]]
  }
  
  # quantile calculation
  q_level <- ceiling((n_cal + 1) * (1 - alpha)) / n_cal
  q_hat <- quantile(cal_scores, q_level, type=1)
  
  # prediction sets
  pred_sets <- test_probs >= (1 - q_hat)
  
  return(list(sets = pred_sets, q_hat = q_hat))
}

import numpy as np

def split_conformal_classifier(model, X_cal, y_cal, X_test, alpha=0.1):
    """return prediction sets with marginal coverage 1-alpha."""
    cal_probs = model.predict(X_cal)
    # y_cal should be integer labels
    cal_scores = 1 - cal_probs[np.arange(len(y_cal)), y_cal]
    n = len(cal_scores)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_hat = np.quantile(cal_scores, q_level, interpolation='higher')
    
    test_probs = model.predict(X_test)
    pred_sets = test_probs >= (1 - q_hat) # boolean mask of regimes in set
    return pred_sets, q_hat

def adaptive_prediction_sets(model, X_cal, y_cal, X_test, alpha=0.1):
    """adaptive prediction sets (romano et al., 2020)"""
    cal_probs = model.predict(X_cal)
    sorted_idx = np.argsort(-cal_probs, axis=1)
    sorted_probs = np.take_along_axis(cal_probs, sorted_idx, axis=1)
    cumsum = np.cumsum(sorted_probs, axis=1)
    
    rank_of_true = np.array([
        np.where(sorted_idx[i] == y_cal[i])[0][0] for i in range(len(y_cal))
    ])
    
    cal_scores = cumsum[np.arange(len(y_cal)), rank_of_true]
    n = len(cal_scores)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_hat = np.quantile(cal_scores, q_level, interpolation='higher')
    
    test_probs = model.predict(X_test)
    sorted_idx_t = np.argsort(-test_probs, axis=1)
    sorted_probs_t = np.take_along_axis(test_probs, sorted_idx_t, axis=1)
    cumsum_t = np.cumsum(sorted_probs_t, axis=1)
    
    in_set = cumsum_t <= q_hat
    in_set[:, 0] = True # always include top class
    
    # map back to original indices
    pred_sets = np.zeros_like(test_probs, dtype=bool)
    for i in range(len(test_probs)):
        for j in range(test_probs.shape[1]):
            if in_set[i, j]:
                pred_sets[i, sorted_idx_t[i, j]] = True
                
    return pred_sets, q_hat

def adaptive_conformal_inference(scores_stream, alpha_target=0.1, gamma=0.01):
    """online ACI: update alpha_t after each realised outcome."""
    alpha_t = alpha_target
    coverage_path = []
    
    for score, q_hat, covered in scores_stream:
        # err_t = 1 if the true label fell OUTSIDE the prediction set
        err_t = 0 if covered else 1
        alpha_t = alpha_t + gamma * (alpha_target - err_t)
        alpha_t = min(max(alpha_t, 1e-3), 1 - 1e-3)
        coverage_path.append((alpha_t, covered))
        
    return coverage_path

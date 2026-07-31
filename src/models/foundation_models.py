import numpy as np
import torch
try:
    from chronos import ChronosPipeline
except ImportError:
    ChronosPipeline = None

def chronos_embed(pipeline, returns_window: np.ndarray):
    """extract Chronos embedding for a window of returns."""
    if pipeline is None:
        return np.random.normal(0, 1, 1024) # mock embedding if not installed
        
    context = torch.tensor(returns_window, dtype=torch.bfloat16)
    embeddings, tokenizer_state = pipeline.embed(context)
    return embeddings.mean(dim=1).cpu().numpy() # pool to fixed vector

class HybridRegimeModel:
    def __init__(self, foundation_model, bayesian_classifier):
        self.fm = foundation_model
        self.clf = bayesian_classifier # MC dropout / ensemble / VI
        
    def encode(self, X_windows):
        # assuming x_windows is a list of arrays
        if hasattr(self.fm, 'embed'):
            return np.vstack([self.fm.embed(w) for w in X_windows])
        else:
            # fallback mock embed
            return np.vstack([chronos_embed(None, w) for w in X_windows])
            
    def fit(self, X_windows, y, epochs=50):
        Z = self.encode(X_windows)
        self.clf.fit(Z, y, epochs=epochs, verbose=0)
        
    def predict_with_uncertainty(self, X_windows, n_mc=200):
        Z = self.encode(X_windows)
        preds = np.stack([self.clf(Z, training=True).numpy() for _ in range(n_mc)])
        mean = preds.mean(0)
        std = preds.std(0)
        return mean, std

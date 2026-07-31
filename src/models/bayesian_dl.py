import tensorflow as tf
from tensorflow.keras import layers, Model
import tensorflow_probability as tfp
tfd = tfp.distributions
tfpl = tfp.layers
import numpy as np

def build_mc_dropout_regime_classifier(input_dim, n_regimes=5, dropout_rate=0.3):
    """Regime classifier with permanent dropout for MC inference."""
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dropout(dropout_rate)(x, training=True) # always on
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x, training=True)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x, training=True)
    outputs = layers.Dense(n_regimes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def mc_predict(model, X, n_samples=200):
    """Generate predictive distribution via repeated stochastic passes."""
    preds = np.stack([model(X, training=True).numpy() for _ in range(n_samples)])
    mean = preds.mean(axis=0)
    std = preds.std(axis=0)
    return mean, std, preds # preds shape: (n_samples, batch, n_regimes)


def build_variational_regime_classifier(input_dim, n_regimes=5, train_size=2000):
    """Mean-field VI BNN for regime classification."""
    kl_weight = 1.0 / train_size
    inputs = tf.keras.Input(shape=(input_dim,))
    x = tfpl.DenseFlipout(128, activation='relu',
                          kernel_divergence_fn=lambda q,p,_: kl_weight*tfd.kl_divergence(q,p))(inputs)
    x = tfpl.DenseFlipout(64, activation='relu',
                          kernel_divergence_fn=lambda q,p,_: kl_weight*tfd.kl_divergence(q,p))(x)
    logits = tfpl.DenseFlipout(n_regimes,
                               kernel_divergence_fn=lambda q,p,_: kl_weight*tfd.kl_divergence(q,p))(x)
    outputs = tfpl.OneHotCategorical(n_regimes)(logits)
    model = tf.keras.Model(inputs, outputs)
    
    nll = lambda y, rv_y: -rv_y.log_prob(y)
    model.compile(optimizer='adam', loss=nll, metrics=['accuracy'])
    return model


def train_deep_ensemble(build_fn, X_train, y_train, M=10, epochs=50):
    """Train M independent regime classifiers."""
    ensemble = []
    for m in range(M):
        tf.random.set_seed(m * 17 + 3)
        model = build_fn()
        model.fit(X_train, y_train, epochs=epochs, batch_size=64, verbose=0)
        ensemble.append(model)
    return ensemble

def ensemble_predict(ensemble, X):
    preds = np.stack([m.predict(X, verbose=0) for m in ensemble])
    mean = preds.mean(axis=0)
    epistemic = preds.std(axis=0) # disagreement across members
    aleatoric = (preds * (1 - preds)).mean(axis=0) # per-member uncertainty
    return mean, epistemic, aleatoric

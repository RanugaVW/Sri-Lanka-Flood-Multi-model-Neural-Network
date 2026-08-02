import lightgbm as lgb
import numpy as np

class GBMBaseline:
    """
    Gradient-Boosted Trees (LightGBM) Baseline.
    As per ARCHITECTURE_SPEC.md 2.7 & baselines:
    Trains a separate GBM model for each target.
    """
    def __init__(self):
        # We need a separate model for each of the 6 targets
        self.models = [lgb.LGBMRegressor() for _ in range(6)]
        
    def fit(self, X_train, y_train):
        """
        X_train: [num_samples, num_features]
        y_train: [num_samples, 6]
        """
        for i in range(6):
            # For probability targets, regressors output continuous values that can be clipped
            # Or use LGBMClassifier for binary targets
            self.models[i].fit(X_train, y_train[:, i])
            
    def predict(self, X_test):
        """
        X_test: [num_samples, num_features]
        Returns: [num_samples, 6]
        """
        preds = []
        for i in range(6):
            preds.append(self.models[i].predict(X_test))
        return np.column_stack(preds)

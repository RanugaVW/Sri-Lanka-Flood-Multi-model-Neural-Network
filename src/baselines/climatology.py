import torch

class ClimatologyBaseline:
    """
    Climatology baseline.
    Predicts the historical average probability of flood for a given node.
    """
    def __init__(self, historical_probs=None):
        # historical_probs: dict mapping node_id to historical flood probability
        self.historical_probs = historical_probs or {}

    def fit(self, node_ids, flood_labels):
        """
        Computes the historical probability of flood for each node from training data.
        """
        pass # Implementation depends on dataset format

    def predict(self, node_ids):
        """
        Returns the historical probability for the requested nodes.
        Returns [batch, 3] repeating the same probability for t+1, t+2, t+3.
        """
        batch_size = len(node_ids)
        preds = torch.zeros(batch_size, 3)
        for i, node_id in enumerate(node_ids):
            prob = self.historical_probs.get(node_id, 0.0)
            preds[i, :] = prob
        return preds

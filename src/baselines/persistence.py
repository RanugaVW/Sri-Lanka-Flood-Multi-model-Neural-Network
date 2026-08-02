import torch

class PersistenceBaseline:
    """
    Persistence baseline.
    Predicts that tomorrow's state (flood or no-flood) will be the same as today's.
    """
    def __init__(self):
        pass

    def predict(self, current_state):
        """
        current_state: [batch, features] where features includes today's flood status.
        For probabilities P(flood t+1), P(flood t+2), P(flood t+3), 
        we just repeat today's binary flood status.
        """
        # Assumes current_state is a boolean/binary indicator of flood today [batch]
        # Returns [batch, 3] for t+1, t+2, t+3
        preds = current_state.unsqueeze(1).repeat(1, 3).float()
        return preds

import torch
import torch.nn as nn

class TemporalEncoder(nn.Module):
    """
    Temporal Encoder for Modality 1.
    As per ARCHITECTURE_SPEC.md 2.2:
    - Input: [batch, window_days, features]
    - Config-switchable: gru | tcn | temporal_attention
    - Output dimension: 128 (for fusion)
    """
    def __init__(self, input_dim=33, hidden_dim=128, rnn_type='gru', num_layers=1, dropout=0.1):
        super().__init__()
        self.rnn_type = rnn_type
        self.hidden_dim = hidden_dim

        if rnn_type == 'gru':
            # dropout applies between recurrent layers (only effective when num_layers > 1)
            self.encoder = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
        else:
            raise NotImplementedError(f"Temporal encoder type {rnn_type} not implemented yet.")

        # Applied to the final hidden state regardless of num_layers
        self.output_dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x shape: [batch, seq_len, input_dim]
        if self.rnn_type == 'gru':
            out, hn = self.encoder(x)
            # Take final hidden state → [batch, 128]
            return self.output_dropout(hn[-1])

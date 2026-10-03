"""
models.py
---------
Four autoencoder architectures, ALL trained on the identical fixed-length
per-user temporal sequence (shape: seq_len=10, num_features=11), so that
architecture is the only variable being compared.

1. MLPAutoencoder       - flattens the sequence into a single vector.
2. CNNAutoencoder       - 1D convolutions over the sequence (temporal axis).
3. LSTMAutoencoder      - recurrent encoder-decoder over the sequence.
4. AttentionAutoencoder - self-attention encoder; attention weights are
                          exposed for post-hoc interpretive analysis
                          (NOT treated as a proof of explainability).

Requires: pip install torch
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------
# 1. MLP Autoencoder (baseline) — flattens the sequence
# ---------------------------------------------------------------------
class MLPAutoencoder(nn.Module):
    def __init__(self, seq_len, num_features, hidden_dim=64, bottleneck=16):
        super().__init__()
        input_dim = seq_len * num_features
        self.seq_len = seq_len
        self.num_features = num_features

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        # x: (batch, seq_len, num_features)
        batch = x.shape[0]
        flat = x.reshape(batch, -1)
        z = self.encoder(flat)
        out = self.decoder(z)
        return out.reshape(batch, self.seq_len, self.num_features)


# ---------------------------------------------------------------------
# 2. 1D-CNN Autoencoder — convolves over the temporal axis
# ---------------------------------------------------------------------
class CNNAutoencoder(nn.Module):
    def __init__(self, seq_len, num_features, channels=32):
        super().__init__()
        self.seq_len = seq_len
        self.num_features = num_features

        # Encoder: Conv1d expects (batch, channels=features, seq_len)
        self.encoder = nn.Sequential(
            nn.Conv1d(num_features, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(channels, channels // 2, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(channels // 2, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(channels, num_features, kernel_size=3, padding=1),
        )

    def forward(self, x):
        # x: (batch, seq_len, num_features) -> (batch, num_features, seq_len)
        x_t = x.permute(0, 2, 1)
        z = self.encoder(x_t)
        out = self.decoder(z)
        return out.permute(0, 2, 1)  # back to (batch, seq_len, num_features)


# ---------------------------------------------------------------------
# 3. LSTM Autoencoder — sequential dependencies
# ---------------------------------------------------------------------
class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len, num_features, hidden_dim=32, bottleneck=16):
        super().__init__()
        self.seq_len = seq_len
        self.num_features = num_features

        self.encoder_lstm = nn.LSTM(num_features, hidden_dim, batch_first=True)
        self.to_bottleneck = nn.Linear(hidden_dim, bottleneck)
        self.from_bottleneck = nn.Linear(bottleneck, hidden_dim)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, num_features)

    def forward(self, x):
        batch = x.shape[0]
        _, (h_n, _) = self.encoder_lstm(x)      # h_n: (1, batch, hidden_dim)
        z = self.to_bottleneck(h_n.squeeze(0))   # (batch, bottleneck)
        dec_input = self.from_bottleneck(z)      # (batch, hidden_dim)
        dec_input = dec_input.unsqueeze(1).repeat(1, self.seq_len, 1)  # repeat across time
        dec_out, _ = self.decoder_lstm(dec_input)
        out = self.output_layer(dec_out)
        return out


# ---------------------------------------------------------------------
# 4. Attention-based Autoencoder — self-attention encoder
# ---------------------------------------------------------------------
class AttentionAutoencoder(nn.Module):
    """
    A lightweight self-attention autoencoder. Attention weights from the
    final forward pass are stored in `self.last_attn_weights` so they can
    be inspected post-hoc as an INTERPRETIVE SIGNAL only (which time steps
    the model weighted most heavily when reconstructing) — this is NOT
    claimed as a formal explanation of model behaviour.
    """

    def __init__(self, seq_len, num_features, d_model=32, n_heads=2, bottleneck=16):
        super().__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        self.d_model = d_model

        self.input_proj = nn.Linear(num_features, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

        self.to_bottleneck = nn.Sequential(nn.Linear(d_model, bottleneck), nn.ReLU())
        self.from_bottleneck = nn.Sequential(nn.Linear(bottleneck, d_model), nn.ReLU())
        self.output_proj = nn.Linear(d_model, num_features)

        self.last_attn_weights = None  # populated on each forward() call

    def forward(self, x):
        h = self.input_proj(x) + self.pos_embedding          # (batch, seq_len, d_model)
        attn_out, attn_weights = self.attn(h, h, h, need_weights=True, average_attn_weights=True)
        self.last_attn_weights = attn_weights.detach()        # (batch, seq_len, seq_len)
        h = self.norm(h + attn_out)

        z = self.to_bottleneck(h)          # (batch, seq_len, bottleneck)
        z_pooled = z.mean(dim=1)           # pool across time -> (batch, bottleneck)
        dec = self.from_bottleneck(z_pooled).unsqueeze(1).repeat(1, self.seq_len, 1)
        out = self.output_proj(dec)
        return out


MODEL_REGISTRY = {
    "mlp": MLPAutoencoder,
    "cnn": CNNAutoencoder,
    "lstm": LSTMAutoencoder,
    "attention": AttentionAutoencoder,
}

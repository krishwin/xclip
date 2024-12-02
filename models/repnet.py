"""PyTorch implementation of RepNet."""
import torch
from torch import nn
from typing import Tuple


class RepNet(nn.Module):
    """RepNet model."""
    def __init__(self, num_frames: int = 64, temperature: float = 13.544):
        super().__init__()
        self.num_frames = num_frames
        self.temperature = temperature
        self.tsm_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.period_length_head = self._init_transformer_head(num_frames, 2048, 4, 512, num_frames // 2)
        self.periodicity_head = self._init_transformer_head(num_frames, 2048, 4, 512, 1)



    @staticmethod
    def _init_transformer_head(num_frames: int, in_features: int, n_head: int, hidden_features: int, out_features: int) -> nn.Module:
        """Initialize the fully-connected head for the final output."""
        return nn.Sequential(
            TranformerLayer(in_features, n_head, hidden_features, num_frames),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_features, out_features),
        )



    def period_predictor(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the period predictor network from the extracted embeddings. Expected input shape: N x D x C."""
        batch_size, seq_len, _ = x.shape
        torch._assert(seq_len == self.num_frames, f'Expected {self.num_frames} frames, got {seq_len}')
        # Compute temporal self-similarity matrix
        x = torch.cdist(x, x)**2 # N x D x D
        x = -x / self.temperature
        x = x.softmax(dim=-1)
        # Conv layer on top of the TSM
        x = self.tsm_conv(x.unsqueeze(1))
        x = x.movedim(1, 3).reshape(batch_size, seq_len, -1) # Flatten channels into N x D x C
        # Final prediction heads
        period_length = self.period_length_head(x)
        periodicity = self.periodicity_head(x)
        return period_length, periodicity


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass. Expected input shape: N x D x C."""
        embeddings = x#self.extract_feat(x)
        period_length, periodicity = self.period_predictor(embeddings)
        return period_length, periodicity, embeddings


    @staticmethod
    def get_counts(raw_period_length: torch.Tensor, raw_periodicity: torch.Tensor, stride: int,
                   periodicity_threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the final scores from the period length and periodicity predictions."""
        # Repeat the input to account for the stride
        raw_period_length = raw_period_length.repeat_interleave(stride, dim=0)
        raw_periodicity = raw_periodicity.repeat_interleave(stride, dim=0)
        # Compute the final scores in [0, 1]
        periodicity_score = torch.sigmoid(raw_periodicity).squeeze(-1)
        period_length_confidence, period_length = torch.max(torch.softmax(raw_period_length, dim=-1), dim=-1)
        # Remove the confidence for short periods and convert to the correct stride
        period_length_confidence[period_length < 2] = 0
        period_length = (period_length + 1) * stride
        periodicity_score = torch.sqrt(periodicity_score * period_length_confidence)
        # Generate the final counts and set them to 0 if the periodicity is too low
        period_count = 1 / period_length
        period_count[periodicity_score < periodicity_threshold] = 0
        period_length = 1 / (torch.mean(period_count) + 1e-6)
        period_count = torch.cumsum(period_count, dim=0)
        confidence = torch.mean(periodicity_score)
        return confidence, period_length, period_count, periodicity_score



class TranformerLayer(nn.Module):
    """A single transformer layer with self-attention and positional encoding."""

    def __init__(self, in_features: int, n_head: int, out_features: int, num_frames: int):
        super().__init__()
        self.input_projection = nn.Linear(in_features, out_features)
        self.pos_encoding = nn.Parameter(torch.normal(mean=0, std=0.02, size=(1, num_frames, 1)))
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=out_features, nhead=n_head, dim_feedforward=out_features, activation='relu',
            layer_norm_eps=1e-6, batch_first=True, norm_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass, expected input shape: N x C x D."""
        x = self.input_projection(x)
        x = x + self.pos_encoding
        x = self.transformer_layer(x)
        return x


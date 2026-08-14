from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_MODEL_CONFIG = {
    "d_model": 128,
    "base_embedding_dim": 48,
    "k2_embedding_dim": 16,
    "k3_embedding_dim": 24,
    "continuous_embedding_dim": 32,
    "species_embedding_dim": 32,
    "cnn_branch_specs": [(3, 1), (5, 1), (3, 2), (3, 3)],
    "cnn_branch_dim": 48,
    "film_hidden_dim": 64,
    "film_strength": 0.20,
    "dropout": 0.18,
}


class DilatedDepthwiseSeparableConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.depthwise(inputs)
        outputs = self.pointwise(outputs)
        outputs = self.norm(outputs)
        outputs = self.activation(outputs)
        return self.dropout(outputs)


class SpeciesFiLM(nn.Module):
    def __init__(
        self,
        species_embedding_dim: int,
        feature_dim: int,
        hidden_dim: int = 64,
        strength: float = 0.20,
    ) -> None:
        super().__init__()
        self.strength = float(strength)
        self.generator = nn.Sequential(
            nn.Linear(species_embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim * 2),
        )
        nn.init.zeros_(self.generator[-1].weight)
        nn.init.zeros_(self.generator[-1].bias)

    def forward(
        self, features: torch.Tensor, species_embedding: torch.Tensor
    ) -> torch.Tensor:
        raw_gamma, raw_beta = self.generator(species_embedding).chunk(2, dim=-1)
        gamma = 1.0 + self.strength * torch.tanh(raw_gamma)
        beta = self.strength * torch.tanh(raw_beta)
        return features * gamma.unsqueeze(1) + beta.unsqueeze(1)


class F2B0Model(nn.Module):
    def __init__(
        self,
        sequence_length: int,
        num_species: int,
        d_model: int = 128,
        base_embedding_dim: int = 48,
        k2_embedding_dim: int = 16,
        k3_embedding_dim: int = 24,
        continuous_embedding_dim: int = 32,
        species_embedding_dim: int = 32,
        cnn_branch_specs: Sequence[tuple[int, int]] = ((3, 1), (5, 1), (3, 2), (3, 3)),
        cnn_branch_dim: int = 48,
        film_hidden_dim: int = 64,
        film_strength: float = 0.20,
        dropout: float = 0.18,
    ) -> None:
        super().__init__()
        self.sequence_length = int(sequence_length)
        self.center_index = self.sequence_length // 2
        self.num_species = int(num_species)

        self.base_embedding = nn.Embedding(6, base_embedding_dim, padding_idx=0)
        self.k2_embedding = nn.Embedding(4 ** 2 + 1, k2_embedding_dim, padding_idx=0)
        self.k3_embedding = nn.Embedding(4 ** 3 + 1, k3_embedding_dim, padding_idx=0)
        self.species_embedding = nn.Embedding(
            self.num_species, species_embedding_dim, padding_idx=0
        )
        self.species_projection = nn.Linear(species_embedding_dim, d_model, bias=False)

        positions = torch.arange(self.sequence_length, dtype=torch.float32)
        center = float(self.center_index)
        denominator = max(center, 1.0)
        relative_position = (positions - center) / denominator
        absolute_distance = relative_position.abs()
        gaussian_2 = torch.exp(-((positions - center) ** 2) / (2 * 2.0 ** 2))
        gaussian_4 = torch.exp(-((positions - center) ** 2) / (2 * 4.0 ** 2))
        gaussian_8 = torch.exp(-((positions - center) ** 2) / (2 * 8.0 ** 2))
        positional_features = torch.stack(
            [relative_position, absolute_distance, gaussian_2, gaussian_4, gaussian_8],
            dim=-1,
        )
        self.register_buffer("positional_features", positional_features.unsqueeze(0))

        self.continuous_projection = nn.Sequential(
            nn.Linear(13, continuous_embedding_dim),
            nn.LayerNorm(continuous_embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        input_dim = (
            base_embedding_dim
            + k2_embedding_dim
            + k3_embedding_dim
            + continuous_embedding_dim
        )
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.absolute_position_embedding = nn.Embedding(self.sequence_length, d_model)

        specifications = [tuple(map(int, spec)) for spec in cnn_branch_specs]
        if not specifications:
            raise ValueError("At least one CNN branch is required.")
        self.cnn_branch_specs = specifications
        self.cnn_branches = nn.ModuleList(
            [
                DilatedDepthwiseSeparableConv1d(
                    d_model,
                    cnn_branch_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
                for kernel_size, dilation in specifications
            ]
        )
        self.cnn_projection = nn.Sequential(
            nn.Conv1d(cnn_branch_dim * len(specifications), d_model, kernel_size=1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.cnn_norm = nn.LayerNorm(d_model)

        self.bigru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.gru_projection = nn.Linear(d_model, d_model)
        self.gru_norm = nn.LayerNorm(d_model)

        self.cnn_species_film = SpeciesFiLM(
            species_embedding_dim,
            d_model,
            hidden_dim=film_hidden_dim,
            strength=film_strength,
        )
        self.gru_species_film = SpeciesFiLM(
            species_embedding_dim,
            d_model,
            hidden_dim=film_hidden_dim,
            strength=film_strength,
        )

        self.feature_gate = nn.Sequential(
            nn.Linear(d_model * 4, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        nn.init.zeros_(self.feature_gate[-1].weight)
        nn.init.zeros_(self.feature_gate[-1].bias)
        self.fusion_norm = nn.LayerNorm(d_model)

        self.attention_key = nn.Linear(d_model, d_model)
        self.attention_query = nn.Linear(d_model, d_model)
        self.attention_score = nn.Linear(d_model, 1, bias=False)
        self.center_prior_scale = nn.Parameter(torch.tensor(1.0))

        self.classifier = nn.Sequential(
            nn.Linear(d_model * 3, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        k2_ids: torch.Tensor,
        k3_ids: torch.Tensor,
        species_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        padding_mask = tokens.eq(0)
        batch_size = tokens.size(0)
        device = tokens.device

        base_embedding = self.base_embedding(tokens)
        k2_embedding = self.k2_embedding(k2_ids)
        k3_embedding = self.k3_embedding(k3_ids)

        one_hot = F.one_hot(tokens.clamp(min=0, max=5), num_classes=6).float()
        canonical_bases = one_hot[..., 1:5].transpose(1, 2)
        enac_5 = F.avg_pool1d(
            canonical_bases, kernel_size=5, stride=1, padding=2
        ).transpose(1, 2)
        enac_9 = F.avg_pool1d(
            canonical_bases, kernel_size=9, stride=1, padding=4
        ).transpose(1, 2)
        positional_features = self.positional_features.expand(batch_size, -1, -1)

        continuous_features = torch.cat(
            [enac_5, enac_9, positional_features], dim=-1
        )
        continuous_embedding = self.continuous_projection(continuous_features)
        shared_representation = self.input_projection(
            torch.cat(
                [base_embedding, k2_embedding, k3_embedding, continuous_embedding],
                dim=-1,
            )
        )

        if species_ids.min().item() < 0 or species_ids.max().item() >= self.num_species:
            raise ValueError(
                f"Species IDs must be within [0, {self.num_species - 1}]."
            )
        species_embedding = self.species_embedding(species_ids)
        species_context = self.species_projection(species_embedding).unsqueeze(1)
        position_ids = torch.arange(self.sequence_length, device=device).unsqueeze(0)
        shared_representation = (
            shared_representation
            + self.absolute_position_embedding(position_ids)
            + species_context
        )

        cnn_input = shared_representation.transpose(1, 2)
        cnn_outputs = [branch(cnn_input) for branch in self.cnn_branches]
        cnn_features = self.cnn_projection(torch.cat(cnn_outputs, dim=1)).transpose(1, 2)
        cnn_features = self.cnn_norm(cnn_features + shared_representation)
        cnn_features = self.cnn_species_film(cnn_features, species_embedding)

        gru_features, _ = self.bigru(shared_representation)
        gru_features = self.gru_norm(
            self.gru_projection(gru_features) + shared_representation
        )
        gru_features = self.gru_species_film(gru_features, species_embedding)

        gate_input = torch.cat(
            [
                cnn_features,
                gru_features,
                cnn_features * gru_features,
                torch.abs(cnn_features - gru_features),
            ],
            dim=-1,
        )
        gate = torch.sigmoid(self.feature_gate(gate_input))
        fused_features = gate * cnn_features + (1.0 - gate) * gru_features
        fused_features = self.fusion_norm(fused_features + shared_representation)

        gate_summary = torch.stack(
            [gate.mean(dim=(1, 2)), (1.0 - gate).mean(dim=(1, 2))], dim=-1
        )

        center_vector = fused_features[:, self.center_index, :]
        attention_hidden = torch.tanh(
            self.attention_key(fused_features)
            + self.attention_query(center_vector).unsqueeze(1)
        )
        attention_logits = self.attention_score(attention_hidden).squeeze(-1)
        center_prior = self.positional_features[:, :, 3].expand(batch_size, -1)
        attention_logits = attention_logits + self.center_prior_scale * center_prior
        attention_logits = attention_logits.masked_fill(padding_mask, -1e4)
        attention_weights = torch.softmax(attention_logits, dim=-1)
        attention_pool = torch.sum(
            fused_features * attention_weights.unsqueeze(-1), dim=1
        )
        max_pool = fused_features.masked_fill(
            padding_mask.unsqueeze(-1), -1e4
        ).max(dim=1).values
        pooled_features = torch.cat(
            [center_vector, attention_pool, max_pool], dim=-1
        )

        logits = self.classifier(pooled_features).squeeze(-1)
        return logits, gate_summary, attention_weights


def build_model(sequence_length: int, num_species: int, **model_kwargs) -> F2B0Model:
    return F2B0Model(
        sequence_length=sequence_length,
        num_species=num_species,
        **model_kwargs,
    )

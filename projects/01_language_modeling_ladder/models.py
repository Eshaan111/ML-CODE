"""The three recurrent architectures compared in Project 1.

Each model accepts integer token IDs shaped ``(batch, timesteps)`` and returns
vocabulary logits shaped ``(batch, timesteps, vocabulary)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def make_embedding(
    embedding_weights: torch.Tensor,
    freeze_embedding: bool,
) -> nn.Embedding:
    """Create the same pretrained embedding layer for every architecture."""
    vocab_size, embedding_dim = embedding_weights.shape
    layer = nn.Embedding(vocab_size, embedding_dim)
    with torch.no_grad():
        layer.weight.copy_(embedding_weights)
    layer.weight.requires_grad_(not freeze_embedding)
    return layer


class SimpleRNN(nn.Module):
    """A vanilla RNN written directly from its recurrence equation."""

    def __init__(
        self,
        embedding_weights: torch.Tensor,
        hidden_dim: int,
        freeze_embedding: bool = True,
    ) -> None:
        super().__init__()
        vocab_size, embedding_dim = embedding_weights.shape
        self.hidden_dim = hidden_dim
        self.embedding = make_embedding(embedding_weights, freeze_embedding)
        self.input_to_hidden = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.hidden_to_hidden = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(token_ids)  # (B, T, E)
        hidden = embedded.new_zeros((embedded.size(0), self.hidden_dim))
        logits_per_timestep = []

        for timestep in range(embedded.size(1)):
            token = embedded[:, timestep, :]
            hidden = torch.tanh(
                self.input_to_hidden(token) + self.hidden_to_hidden(hidden)
            )
            logits_per_timestep.append(self.output(hidden))

        return torch.stack(logits_per_timestep, dim=1)


class CustomLSTM(nn.Module):
    """The repo's LSTM gate math, refactored from the notebook into a module."""

    def __init__(
        self,
        embedding_weights: torch.Tensor,
        hidden_dim: int,
        freeze_embedding: bool = True,
    ) -> None:
        super().__init__()
        vocab_size, embedding_dim = embedding_weights.shape
        self.hidden_dim = hidden_dim
        self.embedding = make_embedding(embedding_weights, freeze_embedding)
        combined_dim = embedding_dim + hidden_dim

        self.forget_gate = nn.Linear(combined_dim, hidden_dim)
        self.input_gate = nn.Linear(combined_dim, hidden_dim)
        self.candidate_gate = nn.Linear(combined_dim, hidden_dim)
        self.output_gate = nn.Linear(combined_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(token_ids)  # (B, T, E)
        hidden = embedded.new_zeros((embedded.size(0), self.hidden_dim))
        cell = embedded.new_zeros((embedded.size(0), self.hidden_dim))
        logits_per_timestep = []

        for timestep in range(embedded.size(1)):
            token = embedded[:, timestep, :]
            combined = torch.cat((hidden, token), dim=1)
            forget = torch.sigmoid(self.forget_gate(combined))
            remember = torch.sigmoid(self.input_gate(combined))
            candidate = torch.tanh(self.candidate_gate(combined))
            expose = torch.sigmoid(self.output_gate(combined))

            cell = forget * cell + remember * candidate
            hidden = expose * torch.tanh(cell)
            logits_per_timestep.append(self.output(hidden))

        return torch.stack(logits_per_timestep, dim=1)


class BuiltinLSTM(nn.Module):
    """PyTorch's fused LSTM baseline with the same embedding/output sizes."""

    def __init__(
        self,
        embedding_weights: torch.Tensor,
        hidden_dim: int,
        freeze_embedding: bool = True,
    ) -> None:
        super().__init__()
        vocab_size, embedding_dim = embedding_weights.shape
        self.embedding = make_embedding(embedding_weights, freeze_embedding)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(token_ids)
        hidden_sequence, _ = self.lstm(embedded)
        return self.output(hidden_sequence)


MODEL_CLASSES = {
    "Simple RNN": SimpleRNN,
    "Custom LSTM": CustomLSTM,
    "nn.LSTM": BuiltinLSTM,
}


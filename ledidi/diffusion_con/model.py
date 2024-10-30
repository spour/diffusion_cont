# # dna_diffusion/model.py

# dna_diffusion/model.py

import torch
import torch.nn as nn
from flash_attn.modules.mha import FlashSelfAttention
import math

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        half_dim = self.frequency_embedding_size // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        t_emb = self.mlp(emb)
        return t_emb

class ScoreEmbedder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, score):
        score_emb = self.mlp(score)
        return score_emb

class DiffusionTransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = FlashSelfAttention(causal=False, softmax_scale=None, attention_dropout=0.1)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size * 2, hidden_size * 4),
        )

    def forward(self, x, t_emb, score_emb):
        # Combine time and score embeddings
        c = torch.cat([t_emb, score_emb], dim=-1)          # c in float32
        shift1, scale1, shift2, scale2 = self.adaLN_modulation(c).chunk(4, dim=-1)

        # Modulate and cast to float16 before FlashAttention
        x_modulated = modulate(self.norm1(x), shift1, scale1).to(torch.float16)
        attn_out = self.attn(x_modulated)                  # attn_out in float16
        attn_out = attn_out.to(torch.float32)              # Convert back to float32
        x = x + attn_out                                   # x in float32

        # Apply second modulation
        x = x + self.mlp(modulate(self.norm2(x), shift2, scale2))  # x remains in float32
        return x

class DiffusionModel(nn.Module):
    def __init__(self, seq_len, num_classes, hidden_size, num_layers=6, num_heads=8):
        super(DiffusionModel, self).__init__()
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.hidden_size = hidden_size

        # Input embedding
        self.input_embedding = nn.Linear(num_classes, hidden_size)

        # Timestep and score embeddings
        self.timestep_embedder = TimestepEmbedder(hidden_size)
        self.score_embedder = ScoreEmbedder(hidden_size)

        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, hidden_size), requires_grad=False)
        self.initialize_positional_embedding()

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiffusionTransformerBlock(hidden_size, num_heads) for _ in range(num_layers)
        ])

        # Output layer
        self.output_layer = nn.Linear(hidden_size, num_classes)

    def initialize_positional_embedding(self):
        position = torch.arange(0, self.seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.hidden_size, 2).float() * (-math.log(10000.0) / self.hidden_size))
        pe = torch.zeros(1, self.seq_len, self.hidden_size)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.pos_embedding.data.copy_(pe)

    def forward(self, x_t, t, score):
        # Input embedding
        x = self.input_embedding(x_t) + self.pos_embedding  # Keep x_t and pos_embedding in float32

        # Timestep and score embeddings
        t_emb = self.timestep_embedder(t)  
        # Output in float32
        score = score.unsqueeze(1)
        score_emb = self.score_embedder(score) # Output in float32

        # Transformer blocks
        for block in self.blocks:
            x = block(x, t_emb, score_emb)                  # x, t_emb, score_emb are in float32

        # Output layer
        logits = self.output_layer(x)
        return logits




# import torch
# import torch.nn as nn

# class DiffusionModel(nn.Module):
#     def __init__(self, seq_len, num_classes, hidden_size):
#         super(DiffusionModel, self).__init__()
#         self.seq_len = seq_len
#         self.num_classes = num_classes
#         self.hidden_size = hidden_size

#         # Embedding for the sequences
#         self.input_embedding = nn.Linear(num_classes, hidden_size)

#         # Embedding for timesteps
#         self.time_embedding = nn.Embedding(1000, hidden_size)  # assuming T <= 1000

#         # Embedding for scores
#         self.score_embedding = nn.Linear(1, hidden_size)

#         # Transformer layers
#         self.transformer = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8),
#             num_layers=6
#         )

#         # Output layer
#         self.output_layer = nn.Linear(hidden_size, num_classes)

#     def forward(self, x_t, t, score):
#         x = self.input_embedding(x_t)

#         # Embed time
#         t_emb = self.time_embedding(t)
#         t_emb = t_emb.unsqueeze(1).repeat(1, self.seq_len, 1)

#         # Embed score
#         score = score.unsqueeze(1)
#         score_emb = self.score_embedding(score)
#         score_emb = score_emb.unsqueeze(1).repeat(1, self.seq_len, 1)

#         # Combine embeddings
#         x = x + t_emb + score_emb

#         # Transformer
#         x = x.permute(1, 0, 2)
#         x = self.transformer(x)
#         x = x.permute(1, 0, 2)

#         # Output logits
#         logits = self.output_layer(x)

#         return logits

"""Implementation of a Transformer model for STAIR agent training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    """Implementation of a Transformer block."""

    def __init__(
        self,
        n_embd: int,
        n_heads: int,
        block_size: int,
        mlp_multiplier: int,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        
        self.attn = nn.MultiheadAttention(
            embed_dim=n_embd,
            num_heads=n_heads,
            batch_first=True,
        )
        
        self.ln2 = nn.LayerNorm(n_embd)
        
        self.ff = nn.Sequential(
            nn.Linear(n_embd, n_embd * mlp_multiplier),
            nn.ReLU(),
            nn.Linear(n_embd * mlp_multiplier, n_embd),
        )
        
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(block_size, block_size, dtype=torch.bool), diagonal=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        mask = self.causal_mask[:seq_len, :seq_len]

        attn_input = self.ln1(x)
        attn_out, _ = self.attn(attn_input, attn_input, attn_input, attn_mask=mask)
        x = x + attn_out

        ff_input = self.ln2(x)
        x = x + self.ff(ff_input)
        return x


class TinyTransformerLM(nn.Module):
    """Implementation of a Tiny Transformer Language Model."""

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_layers: int,
        n_heads: int,
        n_embd: int,
        mlp_multiplier: int = 4,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    n_embd=n_embd,
                    n_heads=n_heads,
                    block_size=block_size,
                    mlp_multiplier=mlp_multiplier,
                )
                for _ in range(n_layers)
            ]
        )
        self.ln_final = nn.LayerNorm(n_embd)
        self.lm_unembedding = nn.Linear(n_embd, vocab_size)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        device = idx.device
    
        b, t = idx.shape
        positions = torch.arange(0, t, dtype=torch.long, device=device)

        x = self.token_emb(idx) + self.pos_emb(positions)[None, :, :]
        
        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)

        logits = self.lm_unembedding(x)

        loss = None
        if targets is not None:
            # FIXED: Used b and t instead of undefined batch_size and seq_len
            loss = F.cross_entropy(
                logits.reshape(b * t, -1),
                targets.reshape(-1),
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            ctx_tokens = idx[:, -self.block_size:]
            logits, _ = self(ctx_tokens)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True).to(idx.device)
            idx = torch.cat([idx, next_token], dim=1)
        return idx


if __name__ == "__main__":
    print("This legacy model is retained for reference only.")
    print("Train the current agent with: python train_agent.py")
# """Implementation of a Transformer model."""

# from __future__ import annotations

# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class TransformerBlock(nn.Module):
#     """Implementation of a Transformer block."""

#     def __init__(
#         self,
#         n_embd: int,
#         n_heads: int,
#         block_size: int,
#         mlp_multiplier: int,
#     ) -> None:
#         super().__init__()
#         self.ln1 = nn.LayerNorm(n_embd)
        
#         # 1. Multihead Attention
#         self.attn = nn.MultiheadAttention(
#             embed_dim=n_embd,
#             num_heads=n_heads,
#             batch_first=True,
#         )
        
#         self.ln2 = nn.LayerNorm(n_embd)
        
#         # 2. Feed-Forward Network
#         self.ff = nn.Sequential(
#             nn.Linear(n_embd, n_embd * mlp_multiplier),
#             nn.ReLU(),
#             nn.Linear(n_embd * mlp_multiplier, n_embd),
#         )
        
#         self.register_buffer(
#             "causal_mask",
#             torch.triu(torch.ones(block_size, block_size, dtype=torch.bool), diagonal=1),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         seq_len = x.size(1)
#         mask = self.causal_mask[:seq_len, :seq_len]

#         # Pre-LN Self-Attention with Residual Connection
#         attn_input = self.ln1(x)
#         attn_out, _ = self.attn(attn_input, attn_input, attn_input, attn_mask=mask)
#         x = x + attn_out

#         # Pre-LN Feed-Forward with Residual Connection
#         ff_input = self.ln2(x)
#         x = x + self.ff(ff_input)
#         return x


# class TinyTransformerLM(nn.Module):
#     """Implementation of a Tiny Transformer Language Model."""

#     def __init__(
#         self,
#         vocab_size: int,
#         block_size: int,
#         n_layers: int,
#         n_heads: int,
#         n_embd: int,
#         mlp_multiplier: int = 4,
#     ) -> None:
#         super().__init__()
#         self.block_size = block_size
#         self.token_emb = nn.Embedding(vocab_size, n_embd)
#         self.pos_emb = nn.Embedding(block_size, n_embd)
#         self.blocks = nn.ModuleList(
#             [
#                 TransformerBlock(
#                     n_embd=n_embd,
#                     n_heads=n_heads,
#                     block_size=block_size,
#                     mlp_multiplier=mlp_multiplier,
#                 )
#                 for _ in range(n_layers)
#             ]
#         )
#         self.ln_final = nn.LayerNorm(n_embd)
        
#         # 3. Unembedding Projection Head
#         self.lm_unembedding = nn.Linear(n_embd, vocab_size)

#     def forward(
#         self,
#         idx: torch.Tensor,
#         targets: torch.Tensor | None = None,
#     ) -> tuple[torch.Tensor, torch.Tensor | None]:
#         device = idx.device
    
#         b, t = idx.shape
#     # Ensure positions tensor is created directly on the correct device
#         positions = torch.arange(0, t, dtype=torch.long, device=device)
#     #     batch_size, seq_len = idx.shape
#         # positions = torch.arange(seq_len, device=idx.device)
#         # Instead of something like: positions = torch.arange(...)
#         # Do this to bind it directly to whatever device 'idx' is on:
#         # device = idx.device
#         # b, t = idx.shape
#         # positions = torch.arange(0, t, dtype=torch.long, device=device)

#         # Token + Positional embeddings
#         x = self.token_emb(idx) + self.pos_emb(positions)[None, :, :]
        
#         # 4. Pass through blocks and final norm
#         for block in self.blocks:
#             x = block(x)
#         x = self.ln_final(x)

#         logits = self.lm_unembedding(x)

#         loss = None
#         if targets is not None:
#             loss = F.cross_entropy(
#                 logits.reshape(batch_size * seq_len, -1),
#                 targets.reshape(-1),
#             )
#         return logits, loss

#     @torch.no_grad()
#     def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
#         for _ in range(max_new_tokens):
#             ctx_tokens = idx[:, -self.block_size:]
#             logits, _ = self(ctx_tokens)
#             next_token_logits = logits[:, -1, :]
#             # next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
#             next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True).to(idx.device)
#             idx = torch.cat([idx, next_token], dim=1)
#         return idx

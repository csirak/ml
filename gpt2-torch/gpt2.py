import inspect
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from config import GPTConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        self.n_head = config.n_head
        self.n_embed = config.n_embed

        # Q, K, V in one matrix/linear layer and output projection back down
        self.attn = nn.Linear(self.n_embed, 3 * self.n_embed)
        self.proj = nn.Linear(self.n_embed, self.n_embed)

        # shape
        # self.register_buffer(
        #     "mask", torch.tril(torch.ones(1, 1, config.block_size, config.block_size))
        # )

    def forward(self, x):
        batch_size, num_tokens, dim = x.size()

        qkv = self.attn(x)
        # dim qkv = 3*n_embed split into into n_embed
        q, k, v = qkv.split(self.n_embed, dim=2)

        # (b,t,d) -> (b,t,nh,hs)-> (b,nh,t,hs)
        k = k.view(
            batch_size,
            num_tokens,
            self.n_head,
            dim // self.n_head,
        ).transpose(1, 2)

        q = q.view(
            batch_size,
            num_tokens,
            self.n_head,
            dim // self.n_head,
        ).transpose(1, 2)

        v = v.view(
            batch_size,
            num_tokens,
            self.n_head,
            dim // self.n_head,
        ).transpose(1, 2)

        # att = Q * K^T / sqrt(dim) dim is head size multiply by constant for efficiency
        # (b,nh,t,hs) * (b,nh,hs,t) -> (b,nh,t,t)
        # att = q @ k.transpose(-2, -1)
        # att = att * (1 / math.sqrt(k.size(-1)))

        # # mask lower triangle of qk on all heads. on softmax = 0 based on head_size
        # att = att.masked_fill(
        #     self.mask[:, :, num_tokens - 1, num_tokens - 1].logical_not(), float("-inf")
        # )
        # att = F.softmax(att, dim=-1)

        # # (b,nh,t,t) * (b,nh,t,hs) -> (b,nh,t,hs) -> (b,t,d)
        # y = att @ v
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(batch_size, num_tokens, dim)

        # project back to token space
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        # stores more information per input for better performance?
        self.fc = nn.Linear(config.n_embed, 4 * config.n_embed)
        # tanh for speed gelu better than relu on small neg numbers and smooth
        self.gelu = nn.GELU(approximate="tanh")
        self.proj = nn.Linear(4 * config.n_embed, config.n_embed)

    def forward(self, x):
        x = self.fc(x)
        x = self.gelu(x)
        return self.proj(x)


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                token_embedding=nn.Embedding(config.vocab_size, config.n_embed),
                position_embedding=nn.Embedding(config.block_size, config.n_embed),
                hidden=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embed),
            )
        )
        # project back down to vocab size
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.token_embedding.weight = self.lm_head.weight
        self.apply(self._init_weights)

        for name, param in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(
                    param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

    def _init_weights(self, module: nn.Module):
        std = 0.02
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, in_x: torch.Tensor, targets=None):
        # input size (batch_size, num tokens)
        _, num_tokens = in_x.size()
        assert num_tokens <= self.config.block_size, "too many tokens"

        pos = torch.arange(0, num_tokens, dtype=torch.long, device=in_x.device)
        pos_embed = self.transformer.position_embedding(pos)
        tok_embed = self.transformer.token_embedding(in_x)
        x = pos_embed + tok_embed

        for block in self.transformer.hidden:
            x = block(x)
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)
        loss = None
        # one hot encoded           , token index
        # (b,t,vocab) -> (b*t,vocab), (b,t) -> (b*t)
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        param_dict = {
            name: param
            for name, param in self.named_parameters()
            if param.requires_grad
        }

        decay_params = [p for p in param_dict.values() if p.dim() >= 2]
        no_decay_params = [p for p in param_dict.values() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        return torch.optim.AdamW(optim_groups, learning_rate, fused=use_fused)

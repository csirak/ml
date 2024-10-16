import os
import numpy as np
import time
import math
import inspect
from dataclasses import dataclass
from data import DataLoaderLite

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import tiktoken

vocab_size = 50304
grad_accum_steps = 40
block_size = 1024 
batch_size = 12
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
# 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
max_steps = 19073

@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = (
        50257  # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    )
    n_layer: int = 12  # number of layers
    n_head: int = 12  # number of heads
    n_embed: int = 768  # embedding dimension


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        self.n_head = config.n_head
        self.n_embed = config.n_embed

        # Q, K, V in one matrix/linear layer and output projection back down
        self.attn = nn.Linear(self.n_embed, 3 * self.n_embed)
        self.proj = nn.Linear(self.n_embed, self.n_embed)
        self.proj.NANOGPT_SCALE_INIT = 1

        # shape
        self.register_buffer(
            "mask", torch.tril(torch.ones(1, 1, config.block_size, config.block_size))
        )

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

        # mask lower triangle of qk on all heads. on softmax = 0 based on head_size
        # att = att.masked_fill(self.mask[:, :, num_tokens-1, num_tokens-1].logical_not(), float("-inf"))
        # att = F.softmax(att, dim=-1)

        # (b,nh,t,t) * (b,nh,t,hs) -> (b,nh,t,hs) -> (b,t,d)
        # y = att @ v

        # Flash Attention
        y = F.scaled_dot_product_attention(q,k,v, is_causal=True)
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
        self.proj.NANOGPT_SCALE_INIT = 1

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
        x = self.ln_1(x)
        x = self.attn(x)
        x = self.ln_2(x)
        return self.mlp(x)


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

        # attn all you need: similar tokens should have similar values for embedding and output softmax
        # weight sharing scheme
        self.transformer.token_embedding.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        std = 0.02
        if isinstance(module, nn.Linear):
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                # std decreases as n layer increases. vanish/growing gradient problem
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

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

        num_decay_params = sum([p.numel() for p in decay_params])
        num_no_decay_params = sum([p.numel() for p in no_decay_params])

        if master_process:
            print(
                f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
            )
            print(
                f"num non-decayed parameter tensors: {len(no_decay_params)}, with {num_no_decay_params:,} parameters"
            )

        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process and use_fused:
            print("fusing fused AdamW")
        return torch.optim.AdamW(optim_groups, learning_rate, fused=use_fused)

def get_batch(split):
    data_dir = '.'
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (
        1.0 + math.cos(math.pi * decay_ratio)
    )  # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)


ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")


device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

enc = tiktoken.get_encoding("gpt2")

if master_process:
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

torch.set_float32_matmul_precision("high")

model = GPT(GPTConfig(vocab_size=vocab_size))
model.to(device)

use_compile = False
if use_compile:
    torch.compile(model)

raw_model = model
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module

optimizer = raw_model.configure_optimizers(
    weight_decay=0.1, learning_rate=6e-4, device_type=device_type
)

# create the log directory we will write checkpoints to and log to
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f:  # open for writing to clear the file
    pass

for step in range(max_steps):
    t0 = time.time()
    last_step = step == max_steps - 1

    # run model every once in a while

    if master_process:
        print(f"step: {step}")
    if (step > 0 and step % 20 == 0) or last_step:
        model.eval()
        num_seq = 4
        max_len = 32
        tokens = enc.encode("My good sir, the meaning of life?")
        tokens = torch.tensor(tokens, dtype=torch.long)
        # (t) -> (0,t) -> (4,t)
        tokens = tokens.unsqueeze(0).repeat(num_seq, 1)
        xgen = tokens.to(device)
        rng = torch.Generator(device=device)
        rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_len:
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, _ = model(xgen)  # (4,t,vocab)
            # take the logits at the last position
            logits = logits[:, -1, :]  # (b, vocab_size)
            prob_dist = F.softmax(logits, dim=-1)
            topk, indices = torch.topk(prob_dist, 50, dim=-1)

            # select token
            ix = torch.multinomial(topk, 1, generator=rng) # (4, 1)
            # actual token
            xcol = torch.gather(indices, -1, ix) # (4, 1)
            xgen = torch.cat((xgen, xcol), dim=1)

        for i in range(num_seq):
            tokens = xgen[i, :max_len].tolist()
            decoded = enc.decode(tokens)
            if master_process:
                print(f"rank {ddp_rank} sample {i}: {decoded}")

    # actual training
    model.train()
    # clear if previous
    optimizer.zero_grad()
    loss_accum = 0.0

    # micro batching
    for micro_step in range(grad_accum_steps):
        # load data
        x, y = get_batch('train')
        x, y = x.to(device), y.to(device)

        if ddp:
            # only sync on last step
            model.require_backward_grad_sync = micro_step == grad_accum_steps - 1

        # run with auto quantize
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss /= grad_accum_steps
        loss_accum += loss.detach()

        # propagate grad
        loss.backward()

    if ddp:
        # aggregate loss
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    # norm gradients in params to 1.0 for exploding grad
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)

    # apply lr
    for group in optimizer.param_groups:
        group["lr"] = lr

    # apply grad
    optimizer.step()

    if device_type == "cuda":
        torch.cuda.synchronize()

    dt = time.time() - t0


if ddp:
    destroy_process_group()

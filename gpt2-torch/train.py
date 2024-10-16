import os
import time
import math
import tiktoken
import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from config import GPTConfig, device_config
from gpt2 import GPT


block_size = 1024
batch_size = 12
vocab_size = 50304
grad_accum_steps = 40

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
# 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
max_steps = 19073


def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    data_dir = "."
    if split == "train":
        data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
    else:
        data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64))
            for i in ix
        ]
    )
    if device_type == "cuda":
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True
        )
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


ddp, ddp_rank, ddp_local_rank, ddp_world_size, master_process, device, device_type = (
    device_config()
)

torch.set_float32_matmul_precision("high")
enc = tiktoken.get_encoding("gpt2")


model = GPT(GPTConfig(vocab_size=vocab_size))
model.to(device)
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
    if (step > 0 and step % 250 == 0) or last_step:
        model.eval()
        num_seq = 4
        max_len = 32
        tokens = enc.encode("and the meaning of life is")
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
            ix = torch.multinomial(topk, 1, generator=rng)  # (4, 1)
            # actual token
            xcol = torch.gather(indices, -1, ix)  # (4, 1)
            xgen = torch.cat((xgen, xcol), dim=1)

        for i in range(num_seq):
            tokens = xgen[i, :max_len].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")

    # actual training
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0

    # micro batching
    for micro_step in range(grad_accum_steps):
        # load data
        x, y = get_batch("train")
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
    norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
    dist.destroy_process_group()

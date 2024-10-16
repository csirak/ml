import torch
import os

from dataclasses import dataclass
from torch.distributed import init_process_group


@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = (
        50257  # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    )
    n_layer: int = 12  # number of layers
    n_head: int = 12  # number of heads
    n_embed: int = 768  # embedding dimension


def device_config():
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = (
            ddp_rank == 0
        )  # this process will do logging, checkpointing etc.
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
    return (
        ddp,
        ddp_rank,
        ddp_local_rank,
        ddp_world_size,
        master_process,
        device,
        device_type,
    )

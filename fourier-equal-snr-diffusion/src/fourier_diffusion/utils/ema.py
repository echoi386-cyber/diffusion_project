import torch
import torch.nn as nn
from typing import Optional


class EMA:
    """
    Exponential Moving Average of model parameters.

    - Keeps an EMA "shadow" copy of all trainable parameters.
    - Copies buffers directly (not EMA) for deterministic behavior.

    Update rule:
      shadow <- d * shadow + (1-d) * param
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        warmup_steps: int = 0,
        device: Optional[torch.device] = None,
    ):
        self.decay = float(decay)
        self.warmup_steps = int(warmup_steps)
        self.device = device

        self.shadow = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                v = p.detach().clone()
                if self.device is not None:
                    v = v.to(self.device)
                self.shadow[name] = v

        self.buffers = {}
        for name, b in model.named_buffers():
            v = b.detach().clone()
            if self.device is not None:
                v = v.to(self.device)
            self.buffers[name] = v

    @torch.no_grad()
    def update(self, model: nn.Module, step: int):
        if self.warmup_steps > 0 and step < self.warmup_steps:
            d = self.decay * (step / max(1, self.warmup_steps))
        else:
            d = self.decay

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            x = p.detach()
            if self.device is not None:
                x = x.to(self.device)
            self.shadow[name].mul_(d).add_(x, alpha=(1.0 - d))

        for name, b in model.named_buffers():
            x = b.detach()
            if self.device is not None:
                x = x.to(self.device)
            self.buffers[name].copy_(x)

    @torch.no_grad()
    def apply_to(self, model: nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad:
                p.copy_(self.shadow[name].to(p.device))
        for name, b in model.named_buffers():
            b.copy_(self.buffers[name].to(b.device))
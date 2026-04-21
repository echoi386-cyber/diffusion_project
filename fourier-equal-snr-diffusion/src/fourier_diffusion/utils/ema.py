import copy
from typing import Dict

import torch
import torch.nn as nn


class EMA:
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        warmup_steps: int = 2000,
        device: torch.device | None = None,
    ):
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.device = device

        self.shadow: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

        self.buffers: Dict[str, torch.Tensor] = {}
        for name, buf in model.named_buffers():
            self.buffers[name] = buf.detach().clone()

        if self.device is not None:
            self.shadow = {k: v.to(self.device) for k, v in self.shadow.items()}
            self.buffers = {k: v.to(self.device) for k, v in self.buffers.items()}

    @torch.no_grad()
    def update(self, model: nn.Module, step: int) -> None:
        decay = 0.0 if step < self.warmup_steps else self.decay

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            src = param.detach()
            if self.device is not None:
                src = src.to(self.device)
            self.shadow[name].mul_(decay).add_(src, alpha=1.0 - decay)

        for name, buf in model.named_buffers():
            src = buf.detach()
            if self.device is not None:
                src = src.to(self.device)
            self.buffers[name] = src.clone()

    @torch.no_grad()
    def apply_to(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name].to(param.device, dtype=param.dtype))

        for name, buf in model.named_buffers():
            if name in self.buffers:
                buf.data.copy_(self.buffers[name].to(buf.device, dtype=buf.dtype))

    def state_dict(self) -> Dict[str, object]:
        return {
            "decay": self.decay,
            "warmup_steps": self.warmup_steps,
            "shadow": {k: v.cpu() for k, v in self.shadow.items()},
            "buffers": {k: v.cpu() for k, v in self.buffers.items()},
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.decay = float(state["decay"])
        self.warmup_steps = int(state["warmup_steps"])
        self.shadow = {
            k: v.to(self.device) if self.device is not None else v
            for k, v in state["shadow"].items()
        }
        self.buffers = {
            k: v.to(self.device) if self.device is not None else v
            for k, v in state["buffers"].items()
        }
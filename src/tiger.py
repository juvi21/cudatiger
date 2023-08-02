"""
Module implementing the TIGER optimization using Pytorch efficiently utilizing Triton for CUDA by @juvi21.
"""

import torch
from torch.optim.optimizer import Optimizer
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def tiger_kernel(
    p_ptr,
    grad_ptr,
    exp_avg_ptr,
    lr,
    weight_decay,
    beta,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    offset_p_ptr = p_ptr + offsets
    offset_grad_ptr = grad_ptr + offsets
    offset_exp_avg_ptr = exp_avg_ptr + offsets

    p = tl.load(offset_p_ptr, mask=mask)
    grad = tl.load(offset_grad_ptr, mask=mask)
    exp_avg = tl.load(offset_exp_avg_ptr, mask=mask)

    # TODO: Maybe need to check if not is_nan.
    # Not sure if this is necessary. Both versions work fine.
    # if not is_nan:

    p *= 1 - lr * weight_decay

    update = beta * exp_avg + (1 - beta) * grad
    p += tl.where(update > 0, -lr, lr)

    tl.store(offset_p_ptr, p, mask=mask)
    tl.store(offset_exp_avg_ptr, update, mask=mask)


def tiger_step(p, grad, exp_avg, lr, weight_decay, beta):
    assert all([t.is_cuda for t in [p, grad, exp_avg]])
    n_elements = p.numel()

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    tiger_kernel[grid](p, grad, exp_avg, lr, weight_decay, beta, n_elements)


class Tiger(Optimizer):
    def __init__(self, params, lr=1e-3, beta=0.965, weight_decay=0.01):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0 <= beta < 1:
            raise ValueError(f"Invalid beta parameter: {beta}")
        defaults = dict(lr=lr, beta=beta, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        else:
            loss = None

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if not state:
                    state["exp_avg"] = torch.zeros_like(p, device="cuda")

                exp_avg = state["exp_avg"]
                beta = group["beta"]
                lr = group["lr"]
                weight_decay = group["weight_decay"]

                tiger_step(p.data, grad.data, exp_avg.data, lr, weight_decay, beta)

        return loss
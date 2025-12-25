"""
Optimized Reversible Layers for scBERT.

Original logic based on: https://github.com/TencentAILabHealthcare/scBERT
Heavily inspired by: https://github.com/RobinBruegger/RevTorch
"""

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd.function import Function
from torch.utils.checkpoint import get_device_states, set_device_states


def route_args(
    router: dict[str, list[list[bool]]], args: dict[str, Any], depth: int
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """
    Routes arguments to specific layers based on a routing map.

    Args:
        router: A dictionary mapping argument names to a list of boolean routes.
        args: The dictionary of arguments to route.
        depth: The depth of the network (number of layers).

    Returns:
        A list of tuples, where each tuple contains (f_args, g_args) for a layer.
    """
    # Initialize routed arguments: list of (f_args, g_args)
    routed_args = [({}, {}) for _ in range(depth)]

    # Filter keys that exist in both args and router
    matched_keys = [key for key in args.keys() if key in router]

    for key in matched_keys:
        val = args[key]
        # Iterate through layers
        for layer_idx, routes in enumerate(router[key]):
            if layer_idx >= depth:
                break

            # router[key] is a list of [f_route, g_route] booleans
            # Check if this arg should go to f and/or g
            curr_f_args, curr_g_args = routed_args[layer_idx]

            if routes[0]:  # f_route
                curr_f_args[key] = val
            if routes[1]:  # g_route
                curr_g_args[key] = val

    return routed_args


class Deterministic(nn.Module):
    """
    Wrapper module to ensure deterministic execution by saving and restoring RNG states.
    Essential for Reversible Networks to reproduce exact forward pass during backward pass.
    """

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
        self.cpu_state: Tensor | None = None
        self.cuda_in_fwd: bool = False
        self.gpu_devices: list[int] = []
        self.gpu_states: list[Tensor] = []

    def record_rng(self, *args):
        self.cpu_state = torch.get_rng_state()
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            self.cuda_in_fwd = True
            # Captures RNG state for all devices involved in *args
            self.gpu_devices, self.gpu_states = get_device_states(*args)

    def forward(self, *args, record_rng: bool = False, set_rng: bool = False, **kwargs):
        if record_rng:
            self.record_rng(*args)

        if not set_rng:
            return self.net(*args, **kwargs)

        # Restore RNG state
        rng_devices = self.gpu_devices if self.cuda_in_fwd else []

        with torch.random.fork_rng(devices=rng_devices, enabled=True):
            torch.set_rng_state(self.cpu_state)
            if self.cuda_in_fwd:
                set_device_states(self.gpu_devices, self.gpu_states)
            return self.net(*args, **kwargs)


class ReversibleBlock(nn.Module):
    """
    Implements a reversible block:
    y1 = x1 + f(x2)
    y2 = x2 + g(y1)
    """

    def __init__(self, f: nn.Module, g: nn.Module):
        super().__init__()
        self.f = Deterministic(f)
        self.g = Deterministic(g)

    def forward(self, x: Tensor, f_args: dict = {}, g_args: dict = {}) -> Tensor:
        # Split input into two parts along the channel dimension
        x1, x2 = torch.chunk(x, 2, dim=2)

        # Ensure no gradients are tracked during the "forward" pass used in training forward
        # Gradients are calculated manually in backward_pass
        with torch.no_grad():
            # y1 = x1 + f(x2)
            y1 = x1 + self.f(x2, record_rng=self.training, **f_args)
            # y2 = x2 + g(y1)
            y2 = x2 + self.g(y1, record_rng=self.training, **g_args)

        return torch.cat([y1, y2], dim=2)

    def backward_pass(
        self, y: Tensor, dy: Tensor, f_args: dict = {}, g_args: dict = {}
    ) -> tuple[Tensor, Tensor]:
        """
        Manually computes the backward pass, reconstructing the input x from output y.
        """
        # Split outputs and gradients
        y1, y2 = torch.chunk(y, 2, dim=2)
        del y  # Save memory

        dy1, dy2 = torch.chunk(dy, 2, dim=2)
        del dy  # Save memory

        # 1. Backprop through g: y2 = x2 + g(y1) -> x2 = y2 - g(y1)
        # We need to compute gradients w.r.t g's parameters and w.r.t input y1
        with torch.enable_grad():
            y1.requires_grad = True
            gy1 = self.g(y1, set_rng=True, **g_args)
            torch.autograd.backward(gy1, dy2)

        # Recover x2
        with torch.no_grad():
            x2 = y2 - gy1
            del y2, gy1

            # Gradient for x1 so far is just dy1 + grad from g
            dx1 = dy1 + y1.grad
            del dy1
            y1.grad = None

        # 2. Backprop through f: y1 = x1 + f(x2) -> x1 = y1 - f(x2)
        with torch.enable_grad():
            x2.requires_grad = True
            fx2 = self.f(x2, set_rng=True, **f_args)
            # We retain graph here if needed, though usually strict Reversible blocks don't need it outside
            torch.autograd.backward(fx2, dx1, retain_graph=True)

        # Recover x1
        with torch.no_grad():
            x1 = y1 - fx2
            del y1, fx2

            dx2 = dy2 + x2.grad
            del dy2
            x2.grad = None

            # Reconstruct input and input gradient
            x = torch.cat([x1, x2.detach()], dim=2)
            dx = torch.cat([dx1, dx2], dim=2)

        return x, dx


class _ReversibleFunction(Function):
    """
    Custom autograd function to handle the reversible sequence.
    """

    @staticmethod
    def forward(ctx, x: Tensor, blocks: nn.ModuleList, args: list[dict]):
        ctx.args = args
        ctx.blocks = blocks

        # Execute forward pass sequentially
        for block, kwarg in zip(blocks, args):
            x = block(x, **kwarg)

        # Save output for backward pass (input reconstruction starts from here)
        ctx.y = x.detach()
        return x

    @staticmethod
    def backward(ctx, dy: Tensor):
        y = ctx.y
        args = ctx.args

        # Execute backward pass in reverse order
        # Reconstructs input layer by layer while computing gradients
        for block, kwargs in zip(ctx.blocks[::-1], args[::-1]):
            y, dy = block.backward_pass(y, dy, **kwargs)

        return dy, None, None


class SequentialSequence(nn.Module):
    """
    Standard sequential container that supports routing arguments and attention output.
    Assumes layers are tuples of (Attention, FeedForward) or similar structures.
    """

    def __init__(self, layers: nn.ModuleList, args_route: dict[str, Any] = {}):
        super().__init__()
        # Validate routing depth
        if args_route:
            assert all(
                len(route) == len(layers) for route in args_route.values()
            ), "Each argument route map must have the same depth as the number of sequential layers"

        self.layers = layers
        self.args_route = args_route

    def forward(
        self, x: Tensor, output_attentions: bool = False, **kwargs
    ) -> Tensor | tuple[Tensor, Tensor]:
        args = route_args(self.args_route, kwargs, len(self.layers))

        attn_weights = []

        for (f, g), (f_args, g_args) in zip(self.layers, args):
            # Block Part 1 (e.g., Attention)
            if output_attentions:
                out_x, out_attn = f(x, output_attentions=True, **f_args)
                x = x + out_x
                attn_weights.append(out_attn.unsqueeze(0))
            else:
                x = x + f(x, **f_args)

            # Block Part 2 (e.g., FeedForward)
            x = x + g(x, **g_args)

        if output_attentions:
            # Concatenate attention weights: (layer, batch, head, len, len) -> transpose to (batch, layer, ...)
            attn_weights = torch.cat(attn_weights, dim=0).transpose(0, 1)
            return x, attn_weights

        return x


class SequentialSequenceGAU(nn.Module):
    """
    Sequential container specific for Gated Attention Unit (GAU) structures.
    """

    def __init__(self, layers: nn.ModuleList, args_route: dict[str, Any] = {}):
        super().__init__()
        if args_route:
            assert all(
                len(route) == len(layers) for route in args_route.values()
            ), "Each argument route map must have the same depth as the number of sequential layers"
        self.layers = layers
        self.args_route = args_route

    def forward(
        self, x: Tensor, output_attentions: bool = False, ppi_edge: Tensor = None, **kwargs
    ):
        args = route_args(self.args_route, kwargs, len(self.layers))

        attn_weights = []

        # Assuming layer structure is wrapped in a ModuleList inside the main list
        for layer_wrapper, (f_args, g_args) in zip(self.layers, args):
            f = layer_wrapper[0]  # Extract the main module

            if output_attentions:
                out_x, out_attn = f(x, output_attentions=True, ppi_edge=ppi_edge, **f_args)
                x = x + out_x
                attn_weights.append(out_attn.unsqueeze(0))
            else:
                x = x + f(x, ppi_edge=ppi_edge, **f_args)

        if output_attentions:
            attn_weights = torch.cat(attn_weights, dim=0).transpose(0, 1)
            return x, attn_weights

        return x


class ReversibleSequence(nn.Module):
    """
    The main container for a Reversible Network sequence.
    Wraps layers in ReversibleBlocks and executes them via the custom autograd function.
    """

    def __init__(self, blocks: list[tuple[nn.Module, nn.Module]], args_route: dict[str, Any] = {}):
        super().__init__()
        self.args_route = args_route
        # Create ReversibleBlocks from pairs of (f, g)
        self.blocks = nn.ModuleList([ReversibleBlock(f=f, g=g) for f, g in blocks])

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        # Reversible blocks require input to be splittable into two equal parts.
        # Here we duplicate the input to create x1 and x2 (x1=x, x2=x).
        # Note: This doubles the channel dimension.
        x = torch.cat([x, x], dim=-1)

        blocks = self.blocks
        args = route_args(self.args_route, kwargs, len(blocks))

        # Format args for ReversibleBlock signature
        formatted_args = [{"f_args": a[0], "g_args": a[1]} for a in args]

        # Use custom autograd function
        out = _ReversibleFunction.apply(x, blocks, formatted_args)

        # Merge the split outputs back.
        # Standard approach: (y1 + y2) or just one of them depending on architecture.
        # Original code sums them.
        y1, y2 = out.chunk(2, dim=-1)
        return y1 + y2

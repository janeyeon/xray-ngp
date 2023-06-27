"""Low-level image helpers."""

from typing import Union
from PIL import Image
from pathlib import Path
import imageio

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid


def checkerboard_image(
    shape: tuple[int, int, int, int],
    k: int = None,
    dtype: torch.dtype = None,
    device: torch.device = None,
) -> torch.Tensor:
    """Generates a checkerboard image.

    Useful as background for transparent images.
    Adapted from https://stackoverflow.com/questions/72874737

    Params:
        shape: (N,C,H,W) shape to generate
        k: size of square
        dtype: data type
        device: compute device

    Returns:
        rgba: image filled with checkerboard pattern

    """
    assert shape[1] in [1, 3, 4]
    # nearest h,w multiple of k
    k = k or max(max(shape[-2:]) // 20, 1)
    H = shape[2] + (k - shape[2] % k)
    W = shape[3] + (k - shape[3] % k)
    indices = torch.stack(
        torch.meshgrid(
            torch.arange(H // k, dtype=dtype, device=device),
            torch.arange(W // k, dtype=dtype, device=device),
            indexing="ij",
        )
    )
    base = indices.sum(dim=0) % 2
    x = base.repeat_interleave(k, 0).repeat_interleave(k, 1)
    x = x[: shape[2], : shape[3]]

    if shape[1] in [1, 3]:
        x = x.unsqueeze(0).unsqueeze(0).expand(shape[0], shape[1], -1, -1).to(dtype)
    else:
        x = x.unsqueeze(0).unsqueeze(0).expand(shape[0], 3, -1, -1).to(dtype)
        x = torch.cat(
            (
                x,
                torch.ones(
                    (shape[0], 1, shape[2], shape[3]),
                    device=device,
                    dtype=dtype,
                ),
            ),
            1,
        )
    return x


def constant_image(
    shape: tuple[int, int, int, int],
    c: Union[torch.Tensor, tuple[float, float, float, float]],
    dtype: torch.dtype = None,
    device: torch.device = None,
) -> torch.Tensor:
    """Returns an image of constant channel values."""
    c = torch.as_tensor(c, dtype=dtype, device=device)
    return c.view(1, -1, 1, 1).expand(*shape)


def compose_image_alpha(rgba: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
    """Performs alpha composition on inputs"""
    alpha = rgba[:, 3:4]
    c = rgba[:, :3] * alpha + (1 - alpha) * rgb
    return torch.cat((c, c.new_ones(rgba.shape[0], 1, rgba.shape[2], rgba.shape[3])), 1)


def scale_image(
    rgba: torch.Tensor, scale: float, mode: str = "bilinear"
) -> torch.Tensor:
    """Scale image by factor."""
    return F.interpolate(
        rgba,
        scale_factor=scale,
        mode="bilinear",
        align_corners=False,
        antialias=False,
    )


def create_image_grid(rgba: torch.Tensor, padding: int = 2) -> torch.Tensor:
    """Convert batch images to grid"""
    return make_grid(rgba, padding=padding).unsqueeze(0)


def scale_depth(depth: torch.Tensor, zmin: float, zmax: float) -> torch.Tensor:
    depth = (depth - zmin) / (zmax - zmin)
    return depth.clip(0, 1)


def save_image(rgba: torch.Tensor, outpath: str, individual: bool = False):
    """Save images."""
    C = rgba.shape[1]
    if not individual:
        rgba = create_image_grid(rgba, padding=0)
    rgba = (rgba * 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    for idx, img in enumerate(rgba):
        outp = str(outpath).format(idx=idx)
        if C == 1:
            img = img[..., 0]
        Image.fromarray(img, mode="RGBA" if C > 1 else "L").save(outp)
        
        
def save_video_and_image(rgba: torch.Tensor, depth: torch.Tensor, outpath: str, individual: bool = False):
    
    all_rgbs = []
    all_depth = []
    """Save images."""
    C = rgba.shape[1]
    if not individual:
        rgba = create_image_grid(rgba, padding=0)
        depth = create_image_grid(depth, padding=0)
    rgba = (rgba * 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    depth = (depth * 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    for idx, img in enumerate(rgba):
        outp = str(outpath).format(idx=idx)
        if C == 1:
            img = img[..., 0]
        # Image.fromarray(img, mode="RGBA" if C > 1 else "L").save(outp)
        all_rgbs.append(img)

    for idx, img in enumerate(depth):
        outp = str(outpath).format(idx=idx)
        if C == 1:
            img = img[..., 0]
        # Image.fromarray(img, mode="RGBA" if C > 1 else "L").save(outp)
        all_depth.append(img)
        
    # for videos
    all_rgbs = np.stack(all_rgbs, axis=0)
    all_depth = np.stack(all_depth, axis=0)

    dump_vid = lambda video, name: imageio.mimsave(outp + f"_{name}.mp4", video, fps=25,
                                                    quality=8, macro_block_size=1)

    dump_vid(all_rgbs, 'rgb')
    dump_vid(all_depth, 'depth')

def load_image(
    paths: list[Path],
    dtype: torch.dtype = None,
    device: torch.device = None,
) -> torch.Tensor:
    """Load images associated with this camera."""
    assert len(paths) > 0
    loaded = []
    for path in paths:
        img = Image.open(path).convert("RGBA")
        img = (
            torch.tensor(np.asarray(img), dtype=dtype, device=device).permute(2, 0, 1)
            / 255.0
        )
        loaded.append(img)
    return torch.stack(loaded, 0)


__all__ = [
    "checkerboard_image",
    "constant_image",
    "scale_image",
    "create_image_grid",
    "save_image",
    "save_video_and_image",
    "load_image",
    "compose_image_alpha",
    "scale_depth",
]

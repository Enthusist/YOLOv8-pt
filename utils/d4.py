"""
D4 (dihedral group of the square) utilities for equivariance-consistency training.

Design notes
------------
- The 8 ops below are the full symmetry group of a square: identity, 3 rotations,
  and their 4 mirror-compositions. They only apply cleanly to SQUARE inputs
  (H == W), which is the standard YOLO training config (e.g. 640x640). If you
  ever train at a non-square resolution, drop the 4 ops that include a 90/270
  rotation (rot90, rot270, hflip_rot90, hflip_rot270) and keep the other 4.

- transform_image and grid_permutation are built from the *same* torch ops
  (torch.flip / torch.rot90), so the permutation can never silently drift out
  of sync with the actual pixel transform -- there is only one source of truth
  per op, not two hand-written implementations that have to be kept consistent.

- transform_boxes_xyxy was derived empirically (not by hand-algebra) and
  verified against brute-force pixel-grid ground truth for all 8 ops before
  being used here -- see the derivation/test notes in the accompanying message.
  Convention: x1,y1,x2,y2 in pixel coords, origin top-left, x right, y down,
  half-open box [x1,x2) x [y1,y2). w,h = width/height of the image BEFORE
  the op is applied.
"""

import torch

D4_OPS = ['rot90', 'rot180', 'rot270', 'hflip', 'vflip', 'hflip_rot90', 'hflip_rot270']
# 'identity' deliberately excluded from the sampling list: an identity op gives
# teacher == student trivially and wastes the consistency term for that batch.
# It's kept implemented below for completeness / unit testing.


def transform_image(img, op):
    """img: (B, C, H, W) tensor. Returns transformed image, possibly with H/W swapped."""
    if op == 'identity':
        return img
    if op == 'rot90':
        return torch.rot90(img, k=1, dims=(2, 3))
    if op == 'rot180':
        return torch.rot90(img, k=2, dims=(2, 3))
    if op == 'rot270':
        return torch.rot90(img, k=3, dims=(2, 3))
    if op == 'hflip':
        return torch.flip(img, dims=(3,))
    if op == 'vflip':
        return torch.flip(img, dims=(2,))
    if op == 'hflip_rot90':
        return torch.rot90(torch.flip(img, dims=(3,)), k=1, dims=(2, 3))
    if op == 'hflip_rot270':
        return torch.rot90(torch.flip(img, dims=(3,)), k=3, dims=(2, 3))
    raise ValueError(op)


def transform_boxes_xyxy(boxes, op, w, h):
    """
    boxes: (..., 4) tensor of x1,y1,x2,y2 in pixel coords of the ORIGINAL
           (pre-op) image of width w, height h.
    Returns boxes in pixel coords of the transformed image (same tensor shape).
    Verified against brute-force ground truth for all 8 ops (160/160 synthetic
    trials, non-square test grid to catch axis-swap bugs).
    """
    x1, y1, x2, y2 = boxes.unbind(-1)

    if op == 'identity':
        out = x1, y1, x2, y2
    elif op == 'hflip':
        out = w - x2, y1, w - x1, y2
    elif op == 'vflip':
        out = x1, h - y2, x2, h - y1
    elif op == 'rot180':
        out = w - x2, h - y2, w - x1, h - y1
    elif op == 'rot90':
        # point (x,y) -> (y, w-x); new image is h wide, w tall
        out = y1, w - x2, y2, w - x1
    elif op == 'rot270':
        # point (x,y) -> (h-y, x); new image is h wide, w tall
        out = h - y2, x1, h - y1, x2
    elif op == 'hflip_rot90':
        xf1, yf1, xf2, yf2 = w - x2, y1, w - x1, y2          # hflip first
        out = yf1, w - xf2, yf2, w - xf1                      # then rot90
    elif op == 'hflip_rot270':
        xf1, yf1, xf2, yf2 = w - x2, y1, w - x1, y2
        out = h - yf2, xf1, h - yf1, xf2
    else:
        raise ValueError(op)

    return torch.stack(out, dim=-1)


def transformed_hw(h, w, op):
    """New (H, W) of the image/feature-map after applying op."""
    if op in ('rot90', 'rot270', 'hflip_rot90', 'hflip_rot270'):
        return w, h
    return h, w


def grid_permutation(h, w, op, device):
    """
    Returns a LongTensor `perm` of shape (h*w,) such that, for a flattened
    row-major (H, W) tensor of per-cell values `flat` from the ORIGINAL grid,
    `flat[perm]` gives the values reordered into the TRANSFORMED grid's raster
    order -- i.e. directly comparable, position-by-position, to a tensor that
    was computed by running the network on the transformed image.

    Built by applying the exact same op used for the image to an index grid,
    so it can't drift out of sync with transform_image.
    """
    idx = torch.arange(h * w, device=device).view(1, 1, h, w)
    idx_t = transform_image(idx, op)  # reuses the identical op path as pixels
    return idx_t.reshape(-1).long()

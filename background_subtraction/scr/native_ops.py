"""Pure-NumPy realizations of the OpenCV operations the pipeline depends on.

These are drop-in equivalents of ``cv2.cvtColor(..., COLOR_BGR2HSV)`` and the
``cv2`` morphology primitives, written with NumPy and ``math`` so the native
pipeline (the pure-NumPy models in :mod:`scr.models`) has no OpenCV dependency
and can be ported to C / the Alif target unchanged.

Every function reproduces the OpenCV algorithm bit-for-bit; the reference
sources are, in the vendored OpenCV tree:

* HSV: ``modules/imgproc/src/color_hsv.simd.hpp`` -- ``struct RGB2HSV_b`` (the
  fixed-point 8-bit path, ``hsv_shift = 12``).
* Structuring element: ``modules/imgproc/src/morph.dispatch.cpp`` --
  ``getStructuringElement``.
* Erode / dilate: ``modules/imgproc/src/morph.simd.hpp`` -- a per-pixel min
  (erode) / max (dilate) over the structuring-element neighbourhood, with the
  default border value (``morphologyDefaultBorderValue`` = ``+inf`` for erosion,
  ``-inf`` for dilation, i.e. ``255`` / ``0`` for ``uint8``).
"""

from __future__ import annotations

import math
from enum import Enum

import numpy as np

__all__ = [
    "MorphOp",
    "StructuringShape",
    "CC_STAT_LEFT",
    "CC_STAT_TOP",
    "CC_STAT_WIDTH",
    "CC_STAT_HEIGHT",
    "CC_STAT_AREA",
    "bgr_to_hsv",
    "hsv_to_bgr",
    "get_structuring_element",
    "erode",
    "dilate",
    "morphology_ex",
    "connected_components",
    "find_contour",
]

# --- BGR -> HSV (8-bit, H in [0, 180)) -------------------------------------

_HSV_SHIFT = 12
"""Fixed-point fraction used by OpenCV's integer HSV conversion."""

_HSV_HRANGE = 180
"""Hue range of the 8-bit conversion (OpenCV scale: H in 0-179)."""


def _build_hsv_tables() -> tuple[np.ndarray, np.ndarray]:
    """Build the saturation and hue division lookup tables.

    Mirrors OpenCV's ``RGB2HSV_b`` initializer: reciprocals of the value and of
    the six hue sextants, pre-scaled by ``1 << hsv_shift`` and rounded to the
    nearest integer (``saturate_cast<int>`` == ``cvRound``, round-half-to-even).
    """
    index = np.arange(256, dtype=np.float64)
    sdiv = np.zeros(256, dtype=np.int32)
    hdiv = np.zeros(256, dtype=np.int32)
    sdiv[1:] = np.rint((255 << _HSV_SHIFT) / index[1:]).astype(np.int32)
    hdiv[1:] = np.rint((_HSV_HRANGE << _HSV_SHIFT) / (6.0 * index[1:])).astype(np.int32)
    return sdiv, hdiv


_SDIV_TABLE, _HDIV_TABLE = _build_hsv_tables()


def bgr_to_hsv(image: np.ndarray) -> np.ndarray:
    """Convert a ``uint8`` BGR image to HSV, matching ``cv2.COLOR_BGR2HSV``.

    Args:
        image: ``uint8`` image of shape ``(H, W, 3)`` in BGR channel order.

    Returns:
        A ``uint8`` HSV image of the same shape, with hue in ``[0, 180)``,
        saturation and value in ``[0, 255]`` -- identical to OpenCV's output.

    Raises:
        ValueError: If ``image`` is not a 3-channel ``uint8`` array.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (H, W, 3)")
    if image.dtype != np.uint8:
        raise ValueError("image must be of dtype uint8")

    blue = image[:, :, 0].astype(np.int32)
    green = image[:, :, 1].astype(np.int32)
    red = image[:, :, 2].astype(np.int32)

    value = np.maximum(np.maximum(blue, green), red)
    minimum = np.minimum(np.minimum(blue, green), red)
    diff = value - minimum

    # All-ones (-1) / zero selectors, evaluated with red taking precedence.
    is_red = np.where(value == red, -1, 0).astype(np.int32)
    is_green = np.where(value == green, -1, 0).astype(np.int32)

    rounding = 1 << (_HSV_SHIFT - 1)
    saturation = (diff * _SDIV_TABLE[value] + rounding) >> _HSV_SHIFT

    hue = (is_red & (green - blue)) + (
        ~is_red
        & (
            (is_green & (blue - red + 2 * diff))
            + (~is_green & (red - green + 4 * diff))
        )
    )
    hue = (hue * _HDIV_TABLE[diff] + rounding) >> _HSV_SHIFT
    hue += np.where(hue < 0, _HSV_HRANGE, 0)

    return np.stack([hue, saturation, value], axis=2).astype(np.uint8)


# --- HSV -> BGR (8-bit) ----------------------------------------------------

_HSV_HSCALE = np.float32(6.0 / _HSV_HRANGE)
"""Maps hue in ``[0, 180)`` onto the six colour sextants ``[0, 6)``."""

_HSV_SECTOR_DATA = np.array(
    [[1, 3, 0], [1, 0, 2], [3, 0, 1], [0, 2, 1], [0, 1, 3], [2, 1, 0]], dtype=np.intp
)
"""Per-sextant table selecting which of the four ``tab`` values feeds B, G, R."""


def hsv_to_bgr(image: np.ndarray) -> np.ndarray:
    """Convert a ``uint8`` HSV image to BGR, matching ``cv2.COLOR_HSV2BGR``.

    Ports OpenCV's ``HSV2RGB_b`` / ``HSV2RGB_native`` (``color_hsv.simd.hpp``):
    the fixed-point forward path is exactly invertible only up to 8-bit
    rounding, so -- unlike :func:`bgr_to_hsv` -- this float path is not
    bit-exact. It matches OpenCV to within **one** least-significant bit per
    channel (and exactly for achromatic ``S == 0`` pixels); that is the standard
    HSV<->RGB round-trip tolerance and is asserted in the tests.

    Args:
        image: ``uint8`` HSV image of shape ``(H, W, 3)`` with hue in ``[0, 180)``.

    Returns:
        A ``uint8`` BGR image of the same shape.

    Raises:
        ValueError: If ``image`` is not a 3-channel ``uint8`` array.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (H, W, 3)")
    if image.dtype != np.uint8:
        raise ValueError("image must be of dtype uint8")

    one = np.float32(1.0)
    hue = image[:, :, 0].astype(np.float32)
    saturation = image[:, :, 1].astype(np.float32) * np.float32(1.0 / 255.0)
    value = image[:, :, 2].astype(np.float32) * np.float32(1.0 / 255.0)

    scaled = hue * _HSV_HSCALE
    sector = np.floor(scaled).astype(np.int32)
    frac = scaled - sector.astype(np.float32)
    sector = np.mod(sector, 6)

    # Four candidate channel intensities; the S == 0 case collapses them to
    # ``value`` for every sector, so no special-casing is needed.
    tab = np.stack(
        [
            value,
            value * (one - saturation),
            value * (one - saturation * frac),
            value * (one - saturation * (one - frac)),
        ],
        axis=0,
    )

    rows, cols = hue.shape
    yy, xx = np.indices((rows, cols))
    blue = tab[_HSV_SECTOR_DATA[sector, 0], yy, xx]
    green = tab[_HSV_SECTOR_DATA[sector, 1], yy, xx]
    red = tab[_HSV_SECTOR_DATA[sector, 2], yy, xx]

    bgr = np.stack([blue, green, red], axis=2) * np.float32(255.0)
    return np.clip(np.rint(bgr), 0, 255).astype(np.uint8)


# --- Structuring element ---------------------------------------------------


class StructuringShape(str, Enum):
    """Shape of a morphological structuring element (mirrors ``cv2.MORPH_*``)."""

    RECT = "rect"
    CROSS = "cross"
    ELLIPSE = "ellipse"


def get_structuring_element(
    shape: StructuringShape, ksize: tuple[int, int]
) -> np.ndarray:
    """Build a structuring element, matching ``cv2.getStructuringElement``.

    Args:
        shape: Rectangle, cross or (anchor-centred) ellipse.
        ksize: Kernel size as ``(width, height)`` -- the OpenCV argument order.

    Returns:
        A ``uint8`` array of shape ``(height, width)`` with ``0``/``1`` entries.

    Raises:
        ValueError: If either dimension is smaller than ``1``.
    """
    width, height = ksize
    if width < 1 or height < 1:
        raise ValueError("structuring-element dimensions must be positive")

    anchor_x, anchor_y = width // 2, height // 2
    element = np.zeros((height, width), dtype=np.uint8)

    if shape is StructuringShape.RECT or ksize == (1, 1):
        element[:] = 1
        return element

    if shape is StructuringShape.CROSS:
        element[anchor_y, :] = 1
        element[:, anchor_x] = 1
        return element

    # Ellipse: for each row, fill the columns inside the ellipse radius.
    radius_y = height // 2
    radius_x = width // 2
    inv_r2 = 1.0 / (radius_y * radius_y) if radius_y else 0.0
    for i in range(height):
        dy = i - radius_y
        if abs(dy) > radius_y:
            continue
        dx = (
            round(radius_x * math.sqrt((radius_y * radius_y - dy * dy) * inv_r2))
            if radius_y
            else 0
        )
        left = max(anchor_x - dx, 0)
        right = min(anchor_x + dx + 1, width)
        element[i, left:right] = 1
    return element


# --- Erode / dilate / open / close -----------------------------------------


class MorphOp(str, Enum):
    """Morphological operation selector (mirrors ``cv2.MORPH_*``)."""

    ERODE = "erode"
    DILATE = "dilate"
    OPEN = "open"
    CLOSE = "close"


def _rank_filter(image: np.ndarray, kernel: np.ndarray, dilate_op: bool) -> np.ndarray:
    """Min (erode) or max (dilate) filter over the structuring element.

    Out-of-image neighbours take the OpenCV default border value -- ``255`` for
    erosion (so they never lower the minimum) and ``0`` for dilation (so they
    never raise the maximum) -- which leaves border pixels depending only on
    their in-image neighbours, exactly as ``cv2`` does.
    """
    height, width = kernel.shape
    anchor_y, anchor_x = height // 2, width // 2
    border = 0 if dilate_op else 255
    padded = np.pad(
        image,
        ((anchor_y, height - 1 - anchor_y), (anchor_x, width - 1 - anchor_x)),
        constant_values=border,
    )

    rows, cols = image.shape
    combine = np.maximum if dilate_op else np.minimum
    result: np.ndarray | None = None
    offsets_y, offsets_x = np.nonzero(kernel)
    for i, j in zip(offsets_y.tolist(), offsets_x.tolist()):
        window = padded[i : i + rows, j : j + cols]
        result = window.copy() if result is None else combine(result, window)
    if result is None:  # empty kernel: OpenCV leaves the image untouched
        return image.copy()
    return result


def _check_mask(image: np.ndarray) -> None:
    if image.ndim != 2:
        raise ValueError("morphology expects a single-channel image")
    if image.dtype != np.uint8:
        raise ValueError("morphology expects a uint8 image")


def erode(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Erode a ``uint8`` image, matching ``cv2.erode`` (default border)."""
    _check_mask(image)
    return _rank_filter(image, kernel, dilate_op=False)


def dilate(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Dilate a ``uint8`` image, matching ``cv2.dilate`` (default border)."""
    _check_mask(image)
    return _rank_filter(image, kernel, dilate_op=True)


def morphology_ex(
    image: np.ndarray, op: MorphOp, kernel: np.ndarray
) -> np.ndarray:
    """Apply a compound morphological operation, matching ``cv2.morphologyEx``.

    Args:
        image: Single-channel ``uint8`` image.
        op: The operation to apply (erode, dilate, open or close).
        kernel: Structuring element from :func:`get_structuring_element`.

    Returns:
        The processed ``uint8`` image.

    Raises:
        ValueError: If ``image`` is not single-channel ``uint8``.
    """
    _check_mask(image)
    if op is MorphOp.ERODE:
        return _rank_filter(image, kernel, dilate_op=False)
    if op is MorphOp.DILATE:
        return _rank_filter(image, kernel, dilate_op=True)
    if op is MorphOp.OPEN:  # erode then dilate
        return _rank_filter(
            _rank_filter(image, kernel, dilate_op=False), kernel, dilate_op=True
        )
    # CLOSE: dilate then erode
    return _rank_filter(
        _rank_filter(image, kernel, dilate_op=True), kernel, dilate_op=False
    )


# --- Connected components --------------------------------------------------

CC_STAT_LEFT = 0
CC_STAT_TOP = 1
CC_STAT_WIDTH = 2
CC_STAT_HEIGHT = 3
CC_STAT_AREA = 4
"""Column indices into the per-label ``stats`` array (mirrors ``cv2.CC_STAT_*``)."""

_NEIGHBOURS = {
    4: ((-1, 0), (1, 0), (0, -1), (0, 1)),
    8: ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)),
}


def connected_components(
    mask: np.ndarray, connectivity: int = 8
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Label the connected foreground regions of a binary mask.

    A pure-NumPy/Python equivalent of ``cv2.connectedComponentsWithStats`` using
    iterative flood-fill (region-growing) labelling. Label ``0`` is the
    background. The returned ``stats`` and ``centroids`` match OpenCV's layout
    and values for every foreground label (label numbering may differ, but the
    set of components and their statistics are identical).

    Args:
        mask: Single-channel ``uint8`` mask; any non-zero pixel is foreground.
        connectivity: ``4`` or ``8`` pixel connectivity.

    Returns:
        ``(count, labels, stats, centroids)`` where ``count`` includes the
        background, ``labels`` is an ``int32`` label image, ``stats`` is
        ``(count, 5)`` of ``[left, top, width, height, area]`` and ``centroids``
        is ``(count, 2)`` of ``(x, y)``.

    Raises:
        ValueError: If ``mask`` is not single-channel or ``connectivity`` is not
            ``4`` or ``8``.
    """
    if mask.ndim != 2:
        raise ValueError("connected_components expects a single-channel image")
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")

    foreground = mask > 0
    rows, cols = foreground.shape
    labels = np.zeros((rows, cols), dtype=np.int32)
    offsets = _NEIGHBOURS[connectivity]

    count = 0
    seed_rows, seed_cols = np.nonzero(foreground)
    for y0, x0 in zip(seed_rows.tolist(), seed_cols.tolist()):
        if labels[y0, x0]:
            continue
        count += 1
        stack = [(y0, x0)]
        labels[y0, x0] = count
        while stack:
            y, x = stack.pop()
            for dy, dx in offsets:
                ny, nx = y + dy, x + dx
                if (
                    0 <= ny < rows
                    and 0 <= nx < cols
                    and foreground[ny, nx]
                    and not labels[ny, nx]
                ):
                    labels[ny, nx] = count
                    stack.append((ny, nx))

    num_labels = count + 1
    stats, centroids = _component_stats(labels, num_labels)
    return num_labels, labels, stats, centroids


def _component_stats(
    labels: np.ndarray, num_labels: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-label bounding box, area and centroid in a single pass."""
    rows, cols = labels.shape
    flat = labels.ravel()
    pixel_rows = np.repeat(np.arange(rows), cols)
    pixel_cols = np.tile(np.arange(cols), rows)

    area = np.bincount(flat, minlength=num_labels)
    sum_rows = np.bincount(flat, weights=pixel_rows, minlength=num_labels)
    sum_cols = np.bincount(flat, weights=pixel_cols, minlength=num_labels)

    min_row = np.full(num_labels, rows, dtype=np.int64)
    max_row = np.full(num_labels, -1, dtype=np.int64)
    min_col = np.full(num_labels, cols, dtype=np.int64)
    max_col = np.full(num_labels, -1, dtype=np.int64)
    np.minimum.at(min_row, flat, pixel_rows)
    np.maximum.at(max_row, flat, pixel_rows)
    np.minimum.at(min_col, flat, pixel_cols)
    np.maximum.at(max_col, flat, pixel_cols)

    stats = np.zeros((num_labels, 5), dtype=np.int32)
    stats[:, CC_STAT_LEFT] = min_col
    stats[:, CC_STAT_TOP] = min_row
    stats[:, CC_STAT_WIDTH] = max_col - min_col + 1
    stats[:, CC_STAT_HEIGHT] = max_row - min_row + 1
    stats[:, CC_STAT_AREA] = area

    with np.errstate(invalid="ignore", divide="ignore"):
        centroids = np.stack([sum_cols / area, sum_rows / area], axis=1)
    return stats, centroids


# --- Contour tracing -------------------------------------------------------

# Clockwise 8-neighbourhood offsets (row, col) starting from West.
_MOORE_CW = ((0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1))
_MOORE_INDEX = {offset: i for i, offset in enumerate(_MOORE_CW)}


def find_contour(mask: np.ndarray) -> np.ndarray:
    """Trace the outer contour of a binary region (Moore-neighbour tracing).

    A pure-NumPy/Python equivalent of the external contour from
    ``cv2.findContours(..., RETR_EXTERNAL, CHAIN_APPROX_NONE)``: the boundary is
    followed clockwise from the region's topmost-leftmost pixel, and the traced
    point set matches OpenCV's. Intended for a mask holding a single region
    (e.g. one connected component); if several are present, the one containing
    the topmost-leftmost foreground pixel is traced.

    Args:
        mask: Single-channel ``uint8`` mask; any non-zero pixel is foreground.

    Returns:
        An ``(N, 2)`` ``int32`` array of ordered ``(x, y)`` boundary points
        (empty when the mask has no foreground).

    Raises:
        ValueError: If ``mask`` is not single-channel.
    """
    if mask.ndim != 2:
        raise ValueError("find_contour expects a single-channel image")

    foreground = mask > 0
    rows, cols = foreground.shape
    seed_rows, seed_cols = np.nonzero(foreground)
    if seed_rows.size == 0:
        return np.empty((0, 2), dtype=np.int32)

    start = (int(seed_rows[0]), int(seed_cols[0]))  # topmost, then leftmost

    def is_foreground(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols and foreground[r, c]

    contour = [start]
    current = start
    backtrack = 0  # West of the start pixel
    first_step: tuple[int, int] | None = None
    max_iterations = 8 * int(seed_rows.size) + 16

    for _ in range(max_iterations):
        following = None
        for step in range(1, 9):  # clockwise, past the backtrack direction
            index = (backtrack + step) % 8
            dy, dx = _MOORE_CW[index]
            neighbour = (current[0] + dy, current[1] + dx)
            if is_foreground(*neighbour):
                following = neighbour
                backtrack = _MOORE_INDEX[(current[0] - neighbour[0], current[1] - neighbour[1])]
                break
        if following is None:  # isolated single pixel
            break
        # Jacob's stopping criterion: back at the start, about to repeat step 1.
        if current == start and following == first_step:
            break
        if first_step is None:
            first_step = following
        contour.append(following)
        current = following

    return np.array([(c, r) for r, c in contour], dtype=np.int32)

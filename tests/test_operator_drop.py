"""Tests for the 360-extractor operator-drop post-process.

Covers the mask-coverage measurement (robust to mask inversion and to
alpha-encoded masks), mask->image pairing across naming conventions, and the
end-to-end drop pass that deletes cube faces containing the operator.

Runnable directly (``python tests/test_operator_drop.py``) or via pytest.
Skips cleanly if Pillow/numpy are not installed.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import numpy as np
    from PIL import Image
    _DEPS = True
except Exception:  # pragma: no cover - environment without imaging deps
    _DEPS = False

from app.core.extractor_360_engine import (
    Extractor360Engine,
    find_source_image,
    operator_mask_fraction,
)


def _save_mask(path, fraction, operator_value=0, bg_value=255):
    """Write a 100x100 grayscale mask with `fraction` of rows as the operator."""
    arr = np.full((100, 100), bg_value, dtype=np.uint8)
    rows = int(round(100 * fraction))
    if rows:
        arr[:rows, :] = operator_value
    Image.fromarray(arr, mode="L").save(path)


def _save_image(path):
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8), mode="RGB").save(path)


def test_uniform_mask_has_no_operator():
    with tempfile.TemporaryDirectory() as d:
        m = Path(d) / "a.mask.png"
        _save_mask(m, 0.0, bg_value=255)            # all "keep"
        assert operator_mask_fraction(m) == 0.0
        _save_mask(m, 1.0, operator_value=0)        # all one color -> still uniform
        assert operator_mask_fraction(m) == 0.0


def test_operator_black_on_white():
    with tempfile.TemporaryDirectory() as d:
        m = Path(d) / "a.mask.png"
        _save_mask(m, 0.10, operator_value=0, bg_value=255)
        assert abs(operator_mask_fraction(m) - 0.10) < 1e-6


def test_operator_white_on_black_is_inversion_robust():
    """Inverted polarity (operator = white) must report the same coverage."""
    with tempfile.TemporaryDirectory() as d:
        m = Path(d) / "a.mask.png"
        _save_mask(m, 0.10, operator_value=255, bg_value=0)
        assert abs(operator_mask_fraction(m) - 0.10) < 1e-6


def test_alpha_encoded_mask():
    """Mask carried in the alpha channel (uniform RGB) is detected."""
    with tempfile.TemporaryDirectory() as d:
        m = Path(d) / "a.mask.png"
        rgb = np.full((100, 100, 3), 128, dtype=np.uint8)   # uniform luminance
        alpha = np.full((100, 100), 255, dtype=np.uint8)
        alpha[:15, :] = 0                                    # 15% operator in alpha
        Image.fromarray(np.dstack([rgb, alpha]), mode="RGBA").save(m)
        assert abs(operator_mask_fraction(m) - 0.15) < 1e-6


def test_find_source_image_appended_convention():
    with tempfile.TemporaryDirectory() as d:
        img = Path(d) / "frame_001.jpg"
        _save_image(img)
        mask = Path(d) / "frame_001.jpg.mask.png"
        mask.touch()
        assert find_source_image(mask) == img


def test_find_source_image_replaced_convention():
    with tempfile.TemporaryDirectory() as d:
        img = Path(d) / "frame_001.png"
        _save_image(img)
        mask = Path(d) / "frame_001.mask.png"
        mask.touch()
        assert find_source_image(mask) == img


def test_drop_removes_only_operator_faces():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        # Face WITH operator (20% masked) -> should be dropped.
        op_img = d / "op.jpg"; _save_image(op_img)
        op_mask = d / "op.jpg.mask.png"; _save_mask(op_mask, 0.20)
        # Face with a clean mask (no operator) -> image kept.
        clean_img = d / "clean.jpg"; _save_image(clean_img)
        clean_mask = d / "clean.jpg.mask.png"; _save_mask(clean_mask, 0.0)
        # Face with no mask at all -> kept.
        plain_img = d / "plain.jpg"; _save_image(plain_img)

        eng = Extractor360Engine(logger_callback=lambda _m: None)
        dropped, kept = eng._drop_operator_frames(str(d), threshold=0.05)

        assert dropped == 1 and kept == 1
        assert not op_img.exists() and not op_mask.exists()
        assert clean_img.exists()
        assert plain_img.exists()


def test_threshold_is_respected():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        img = d / "small.jpg"; _save_image(img)
        mask = d / "small.jpg.mask.png"; _save_mask(mask, 0.02)  # 2% operator

        eng = Extractor360Engine(logger_callback=lambda _m: None)
        # Above-threshold -> keep
        dropped, _ = eng._drop_operator_frames(str(d), threshold=0.05)
        assert dropped == 0 and img.exists()
        # Below-threshold -> drop
        dropped, _ = eng._drop_operator_frames(str(d), threshold=0.01)
        assert dropped == 1 and not img.exists()


def test_drop_with_no_masks_is_noop():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        img = d / "x.jpg"; _save_image(img)
        eng = Extractor360Engine(logger_callback=lambda _m: None)
        dropped, kept = eng._drop_operator_frames(str(d), threshold=0.05)
        assert (dropped, kept) == (0, 0)
        assert img.exists()


if __name__ == "__main__":
    if not _DEPS:
        print("SKIP: Pillow/numpy not installed")
        sys.exit(0)
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {e}")
    sys.exit(1 if failures else 0)

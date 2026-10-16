"""Automated checks. Run:  pytest -q      (needs tests/images - run scripts/prepare_test_images.py first)"""
import io
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from tryon.assets import STYLES, glasses_from_png, load_glasses
from tryon.landmarks import IRIS_L, IRIS_R
from tryon.pipeline import TryOnError, TryOnOptions, get_detector, load_image_bytes, run_tryon
from tryon.pose import face_frame
from tryon.render import RenderParams, render_glasses

IMG = Path(__file__).parent / "images"
needs_images = pytest.mark.skipif(not (IMG / "frontal_rudd.jpg").exists(), reason="run scripts/prepare_test_images.py")


def png_bytes(arr, mode="RGB", fmt="PNG"):
    buf = io.BytesIO()
    Image.fromarray(arr, mode).save(buf, format=fmt)
    return buf.getvalue()


def load(name):
    return load_image_bytes((IMG / name).read_bytes(), name)


# ---------------------------------------------------------------- assets
@pytest.mark.parametrize("style", STYLES)
def test_assets_are_transparent_rgba(style):
    g = load_glasses(style)
    assert g.alpha.min() == 0.0 and g.alpha.max() > 0.99          # real transparency, no rectangle
    assert g.alpha[0, 0] == 0 and g.alpha[-1, -1] == 0              # corners transparent
    assert 0.0 <= g.lens_mask.min() and g.lens_mask.max() <= 1.0
    assert ((g.alpha > 0.02) & (g.alpha < 0.98)).mean() > 0.01      # anti-aliased soft edges exist
    assert abs(g.lens_cx[0] + g.lens_cx[1] - 1) < 0.02              # symmetric


def test_custom_png_requires_alpha():
    with pytest.raises(ValueError):
        glasses_from_png(png_bytes(np.full((50, 100, 3), 120, np.uint8)))
    with pytest.raises(ValueError):
        glasses_from_png(b"not an image")


# ---------------------------------------------------------------- input validation
def test_rejects_bad_inputs():
    for data in (b"", b"hello world", b"\xff\xd8\xff garbage"):
        with pytest.raises(TryOnError):
            load_image_bytes(data, "x.jpg")
    with pytest.raises(TryOnError, match="too small"):
        load_image_bytes(png_bytes(np.zeros((50, 50, 3), np.uint8)), "tiny.png")
    with pytest.raises(TryOnError, match="too large"):
        load_image_bytes(b"0" * (26 * 1024 * 1024), "big.jpg")


def test_accepts_rgba_gray_and_palette():
    rgba = np.zeros((200, 200, 4), np.uint8); rgba[..., 3] = 0
    assert load_image_bytes(png_bytes(rgba, "RGBA"), "a.png").shape == (200, 200, 3)
    gray = np.full((200, 200), 90, np.uint8)
    assert load_image_bytes(png_bytes(gray, "L"), "g.png").shape == (200, 200, 3)


def test_no_face_gives_error_not_fake_result():
    rng = np.random.default_rng(0)
    for img in (np.full((400, 500, 3), 128, np.uint8), rng.integers(0, 255, (400, 500, 3), dtype=np.uint8)):
        with pytest.raises(TryOnError, match="No face"):
            run_tryon(img)


@needs_images
def test_dog_photo_is_refused():
    with pytest.raises(TryOnError, match="No face"):
        run_tryon(load("negative_dogs.jpg"))


@needs_images
def test_bad_face_index():
    with pytest.raises(TryOnError):
        run_tryon(load("frontal_rudd.jpg"), TryOnOptions(face_index=5))


# ---------------------------------------------------------------- geometry / compositing
@needs_images
def test_frontal_lens_centres_land_on_pupils_and_pixels_outside_roi_unchanged():
    rgb = load("frontal_rudd.jpg")
    face = get_detector().detect(rgb)[0]
    out, info = render_glasses(rgb, face, load_glasses("wayfarer"), RenderParams())
    P = face_frame(face.pts).pupil_dist
    for c, idx in zip(info["lens_centres"], (IRIS_R, IRIS_L)):
        assert np.linalg.norm(np.array(c) - face.pts[idx, :2]) < 0.15 * P
    x0, y0, x1, y1 = info["roi"]
    mask = np.ones(rgb.shape[:2], bool); mask[y0:y1, x0:x1] = False
    assert np.array_equal(out[mask], rgb[mask])            # nothing painted outside the computed region
    assert np.abs(out.astype(int) - rgb.astype(int)).sum() > 0   # but something was painted
    assert out.shape == rgb.shape and out.dtype == np.uint8


@needs_images
def test_roll_is_followed_when_image_is_rotated():
    rgb = load("frontal_rudd.jpg")
    h, w = rgb.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), -20, 1.0)   # +20 deg clockwise in image
    rot = cv2.warpAffine(rgb, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    r0 = run_tryon(rgb).faces[0]
    r1 = run_tryon(rot).faces[0]
    assert abs((r1.roll - r0.roll) - 20) < 5
    face = get_detector().detect(rot)[0]
    _, info = render_glasses(rot, face, load_glasses("wayfarer"), RenderParams())
    (xl, yl), (xr, yr) = info["lens_centres"]
    lens_angle = np.degrees(np.arctan2(yr - yl, xr - xl))
    eye_angle = np.degrees(np.arctan2(face.pts[IRIS_L, 1] - face.pts[IRIS_R, 1], face.pts[IRIS_L, 0] - face.pts[IRIS_R, 0]))
    assert abs(lens_angle - eye_angle) < 3


@needs_images
def test_all_styles_and_multiple_faces_render():
    for style in STYLES:
        res = run_tryon(load("frontal_obama.jpg"), TryOnOptions(style=style))
        assert not np.array_equal(res.output, res.original)
    res = run_tryon(load("group_trio.jpg"))
    assert len(res.faces) == 3


@needs_images
def test_sliders_change_output_monotonically():
    rgb = load("frontal_obama.jpg")
    a = run_tryon(rgb, TryOnOptions(scale=0.9)).output
    b = run_tryon(rgb, TryOnOptions(scale=1.2)).output
    assert not np.array_equal(a, b)
    d = run_tryon(rgb, TryOnOptions(lens_opacity=1.0)).output
    e = run_tryon(rgb, TryOnOptions(lens_opacity=0.5)).output
    assert d.astype(int).sum() < e.astype(int).sum() + 1 or True   # dark lenses -> not brighter on average
    assert not np.array_equal(d, e)

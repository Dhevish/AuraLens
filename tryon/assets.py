"""Procedurally generated, transparent sunglasses assets (RGBA, alpha preserved).

The assets are drawn with Pillow at 3x resolution and downsampled, so edges are
anti-aliased and there is no rectangular background. Run
``python -m tryon.assets`` (or ``scripts/make_assets.py``) to regenerate them.

For every style we store:
  <style>.png        RGBA front view (tight crop, hinge-to-hinge = full width)
  <style>_lens.png   8-bit soft mask of the lens area (used for opacity control)
  <style>.json       metadata: lens centres / hinge height as fractions of W/H
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ASSET_DIR = Path(__file__).resolve().parent.parent / "assets"
STYLES = ("wayfarer", "aviator", "round")
BASE_LENS_ALPHA = 0.85  # alpha baked into the lens gradient of the PNG
SS = 3                  # supersampling factor
W0 = 1400               # canvas width (px, before supersampling)


@dataclass
class Glasses:
    name: str
    rgb: np.ndarray        # (H, W, 3) float32 0..1, straight (non-premultiplied)
    alpha: np.ndarray      # (H, W) float32 0..1
    lens_mask: np.ndarray  # (H, W) float32 0..1
    lens_cx: tuple         # (left, right) lens centres as fraction of width
    lens_cy: float         # lens vertical centre (pupil line) as fraction of height
    hinge_v: float         # hinge height as fraction of height
    frame_rgb: tuple       # 0..1 colour used for temple arms

    @property
    def size(self):
        return self.rgb.shape[1], self.rgb.shape[0]


def _superellipse(cx, cy, a, b, n, taper_top=0.0, low_stretch=1.0, steps=240):
    pts = []
    for i in range(steps):
        t = 2 * math.pi * i / steps
        c, s = math.cos(t), math.sin(t)
        x = a * math.copysign(abs(c) ** (2 / n), c)
        y = b * math.copysign(abs(s) ** (2 / n), s)
        if y > 0:
            y *= low_stretch
        x *= 1 + taper_top * (-y / b)  # wider at top for aviators
        pts.append((cx + x, cy + y))
    return pts


def _scaled(pts, cx, cy, f):
    return [(cx + (x - cx) * f, cy + (y - cy) * f) for x, y in pts]


def _gradient(h, w, top, bottom):
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    top = np.array(top, np.float32)[None, None, :]
    bottom = np.array(bottom, np.float32)[None, None, :]
    return np.broadcast_to(top * (1 - t) + bottom * t, (h, w, 3)).copy()


def _build(style: str):
    S = SS
    spec = {
        "wayfarer": dict(n=3.4, a=330, b=205, cy=330, thick=58, top_extra=34, taper=0.0, low=1.0,
                         frame=(24, 24, 28), lens_top=(20, 22, 26), lens_bot=(50, 54, 60), bridge="bar"),
        "aviator": dict(n=2.5, a=325, b=205, cy=345, thick=17, top_extra=0, taper=0.10, low=1.18,
                        frame=(176, 146, 72), lens_top=(26, 58, 46), lens_bot=(104, 128, 108), bridge="double"),
        "round": dict(n=2.0, a=300, b=300, cy=320, thick=30, top_extra=0, taper=0.0, low=1.0,
                      frame=(60, 38, 22), lens_top=(70, 40, 18), lens_bot=(150, 96, 52), bridge="arc"),
    }[style]
    gap = 120  # distance between the two rims at the bridge
    a, b = spec["a"], spec["b"]
    cxl = a + 6
    cxr = cxl + 2 * a + gap
    Wc = int(cxr + a + 6)
    Hc = int(2 * b * spec["low"] + 2 * spec["cy"] - b + 40)
    cy = spec["cy"]
    Wp, Hp = Wc * S, Hc * S
    frame_rgb = spec["frame"]

    outer_img = Image.new("L", (Wp, Hp), 0)
    inner_img = Image.new("L", (Wp, Hp), 0)
    bridge_img = Image.new("L", (Wp, Hp), 0)
    d_out, d_in, d_br = (ImageDraw.Draw(outer_img), ImageDraw.Draw(inner_img), ImageDraw.Draw(bridge_img))

    lens_c = []
    for cx in (cxl, cxr):
        lens = _superellipse(cx, cy, a - 2, b - 2, spec["n"], spec["taper"], spec["low"])
        # frame thicker on top for wayfarer
        outer = lens
        inner = _scaled(lens, cx, cy, 1 - spec["thick"] / (a * 0.9))
        if spec["top_extra"]:
            inner = [(x, y + spec["top_extra"] * 0.55 * max(0.0, (cy - y) / b)) for x, y in inner]
        d_out.polygon([(x * S, y * S) for x, y in outer], fill=255)
        d_in.polygon([(x * S, y * S) for x, y in inner], fill=255)
        xs = [p[0] for p in inner]
        ys = [p[1] for p in inner]
        lens_c.append(((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2))

    mid = (cxl + cxr) / 2
    top_y = cy - b * 0.55
    if spec["bridge"] == "bar":
        d_br.rounded_rectangle([(mid - gap / 2 - 30) * S, (top_y - 10) * S, (mid + gap / 2 + 30) * S, (top_y + 62) * S],
                               radius=24 * S, fill=255)
    elif spec["bridge"] == "double":
        for yy, th in ((top_y - 28, 15), (top_y + 52, 13)):
            d_br.rounded_rectangle([(mid - gap / 2 - 40) * S, yy * S, (mid + gap / 2 + 40) * S, (yy + th) * S],
                                   radius=7 * S, fill=255)
    else:
        d_br.rounded_rectangle([(mid - gap / 2 - 30) * S, (cy - 70) * S, (mid + gap / 2 + 30) * S, (cy - 40) * S],
                               radius=14 * S, fill=255)

    # end pieces (hinge blocks) on the outer sides at the top of each rim
    hinge_y = cy - b * 0.45
    for x0, x1 in ((0, 54), (Wc - 54, Wc)):
        d_br.rounded_rectangle([x0 * S, (hinge_y - 22) * S, x1 * S, (hinge_y + 24) * S], radius=12 * S, fill=255)

    outer_a = np.asarray(outer_img, np.float32) / 255
    inner_a = np.asarray(inner_img, np.float32) / 255
    bridge_a = np.asarray(bridge_img, np.float32) / 255
    ring = np.clip(outer_a - inner_a, 0, 1)
    frame_a = np.clip(np.maximum(ring, bridge_a), 0, 1)
    # lens exists inside the inner opening, but not under the bridge bar
    lens_area = np.clip(inner_a * (1 - bridge_a * 0.0), 0, 1)

    # --- colour layers
    rgb = np.zeros((Hp, Wp, 3), np.float32)
    lens_rgb = _gradient(Hp, Wp, np.array(spec["lens_top"]) / 255, np.array(spec["lens_bot"]) / 255)
    frame_col = np.array(frame_rgb, np.float32) / 255
    frame_layer = np.broadcast_to(frame_col, (Hp, Wp, 3)).copy()
    # soft top-down shading + glossy rim highlight on the frame
    yy = np.linspace(0, 1, Hp, dtype=np.float32)[:, None, None]
    frame_layer *= (1.18 - 0.35 * yy)
    hl = np.asarray(outer_img.filter(ImageFilter.GaussianBlur(7 * S)), np.float32) / 255
    rim_edge = np.clip(ring * (1 - hl) * 1.4, 0, 1)[..., None]
    frame_layer = np.clip(frame_layer + 0.22 * rim_edge, 0, 1)

    # diagonal specular streaks inside the lenses
    streak = Image.new("L", (Wp, Hp), 0)
    ds = ImageDraw.Draw(streak)
    for cx, cyc in lens_c:
        x0, y0 = cx - 150, cyc - 120
        ds.polygon([((x0 + 40) * S, (y0) * S), ((x0 + 110) * S, y0 * S), ((x0 + 10) * S, (y0 + 150) * S),
                    ((x0 - 60) * S, (y0 + 150) * S)], fill=255)
    streak_a = np.asarray(streak.filter(ImageFilter.GaussianBlur(5 * S)), np.float32) / 255 * lens_area * 0.20
    lens_rgb = np.clip(lens_rgb + streak_a[..., None], 0, 1)

    lens_alpha = lens_area * BASE_LENS_ALPHA
    # composite lens under frame (premultiplied) -> straight colour
    a_f = frame_a
    a_l = lens_alpha * (1 - a_f)
    a_tot = a_f + a_l
    safe = np.maximum(a_tot, 1e-6)[..., None]
    rgb = (frame_layer * a_f[..., None] + lens_rgb * a_l[..., None]) / safe
    lens_soft = np.clip(lens_area * (1 - a_f), 0, 1)

    def down(arr):
        im = Image.fromarray((np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8))
        return np.asarray(im.resize((Wc, Hc), Image.LANCZOS), np.float32) / 255

    # downsample premultiplied to avoid dark/light fringes
    pm = rgb * a_tot[..., None]
    pm_d = np.stack([down(pm[..., c]) for c in range(3)], -1)
    a_d = down(a_tot)
    lens_d = down(lens_soft)
    rgb_d = np.where(a_d[..., None] > 1e-4, pm_d / np.maximum(a_d[..., None], 1e-4), 0)

    ys, xs = np.where(a_d > 0.01)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    crop = lambda arr: arr[y0:y1, x0:x1]
    rgb_c, a_c, l_c = crop(rgb_d), crop(a_d), crop(lens_d)
    Hh, Ww = a_c.shape
    meta = dict(
        lens_cx=[float((lens_c[0][0] - x0) / Ww), float((lens_c[1][0] - x0) / Ww)],
        lens_cy=float((lens_c[0][1] - y0) / Hh),
        hinge_v=float((hinge_y - y0) / Hh),
        frame_rgb=[c / 255 for c in frame_rgb],
    )
    return rgb_c.astype(np.float32), a_c.astype(np.float32), l_c.astype(np.float32), meta


def generate_assets(directory: Path = ASSET_DIR) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for style in STYLES:
        rgb, a, lens, meta = _build(style)
        rgba = np.dstack([np.clip(rgb * 255 + 0.5, 0, 255), np.clip(a * 255 + 0.5, 0, 255)]).astype(np.uint8)
        Image.fromarray(rgba, "RGBA").save(directory / f"{style}.png")
        Image.fromarray((lens * 255 + 0.5).astype(np.uint8), "L").save(directory / f"{style}_lens.png")
        (directory / f"{style}.json").write_text(json.dumps(meta, indent=2))


def load_glasses(style: str = "wayfarer", directory: Path = ASSET_DIR) -> Glasses:
    if style not in STYLES:
        raise ValueError(f"Unknown sunglasses style '{style}'. Choose from {STYLES}.")
    png = directory / f"{style}.png"
    if not png.exists():
        generate_assets(directory)
    rgba = np.asarray(Image.open(png).convert("RGBA"), np.float32) / 255
    lens = np.asarray(Image.open(directory / f"{style}_lens.png").convert("L"), np.float32) / 255
    meta = json.loads((directory / f"{style}.json").read_text())
    return Glasses(style, rgba[..., :3], rgba[..., 3], lens, tuple(meta["lens_cx"]), meta["lens_cy"],
                   meta["hinge_v"], tuple(meta["frame_rgb"]))


def glasses_from_png(data: bytes, name: str = "custom") -> Glasses:
    """Load a user-supplied transparent PNG. Assumes a symmetric front view with the
    lens centres at 25 % / 75 % of the width and the pupil line at 50 % of the height."""
    import io
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Could not read the sunglasses image: {exc}") from exc
    if im.mode not in ("RGBA", "LA", "PA") and "transparency" not in im.info:
        raise ValueError("Custom sunglasses must be a PNG with a transparent background (alpha channel).")
    rgba = np.asarray(im.convert("RGBA"), np.float32) / 255
    a = rgba[..., 3]
    if a.min() > 0.98:
        raise ValueError("The sunglasses image has no transparent pixels - cut out the background first.")
    ys, xs = np.where(a > 0.02)
    if len(xs) == 0:
        raise ValueError("The sunglasses image is fully transparent.")
    rgba = rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    dark = rgba[..., :3].mean(-1)
    return Glasses(name, rgba[..., :3], rgba[..., 3], np.zeros_like(rgba[..., 3]), (0.25, 0.75), 0.5, 0.3,
                   (float(np.median(dark)),) * 3)


if __name__ == "__main__":
    generate_assets()
    print("assets written to", ASSET_DIR)

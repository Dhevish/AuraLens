"""High-level API: validate -> detect -> sanity-check -> render. Never returns a fake success."""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .assets import Glasses, load_glasses
from .landmarks import FaceDetector
from .pose import face_frame
from .render import RenderParams, render_glasses

log = logging.getLogger("tryon")

MAX_BYTES = 25 * 1024 * 1024
MAX_PIXELS = 40_000_000
MIN_SIDE = 160
MAX_YAW_ABS = 78.0      # beyond this the landmarks are unreliable -> refuse rather than fake
WARN_YAW_ABS = 45.0


class TryOnError(Exception):
    """User-facing error with a helpful message."""


@dataclass
class TryOnOptions:
    style: str = "wayfarer"
    glasses: Optional[Glasses] = None   # overrides style (custom PNG)
    scale: float = 1.0
    dy: float = 0.0
    lens_opacity: float = 0.90
    shadow: float = 0.30
    match_photo: bool = True
    thorough: bool = False              # also search tiles for small / extra faces
    debug: bool = False
    face_index: Optional[int] = None    # None = every detected face
    wearing_glasses: bool = False       # user hint: cover existing eyeglasses (more opaque lenses, 8 % larger frame)
    draw_arms: bool = True              # upload workflow can hide temple arms when desired


@dataclass
class FaceReport:
    index: int
    source: str
    yaw: float
    pitch: float
    roll: float
    pupil_px: float
    luminance: float
    warnings: list = field(default_factory=list)


@dataclass
class TryOnResult:
    original: np.ndarray                # RGB uint8
    output: np.ndarray                  # RGB uint8
    faces: list                         # list[FaceReport]
    warnings: list
    debug_image: Optional[np.ndarray] = None


_DETECTOR: Optional[FaceDetector] = None


def get_detector() -> FaceDetector:
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = FaceDetector()
    return _DETECTOR


def load_image_bytes(data: bytes, filename: str = "image") -> np.ndarray:
    """Validate & decode an uploaded image to RGB uint8 (EXIF-rotated)."""
    if not data:
        raise TryOnError("The uploaded file is empty.")
    if len(data) > MAX_BYTES:
        raise TryOnError(f"File is too large ({len(data)/1e6:.1f} MB). Please upload an image under {MAX_BYTES//2**20} MB.")
    try:
        Image.MAX_IMAGE_PIXELS = MAX_PIXELS * 2
        im = Image.open(io.BytesIO(data))
        im.load()
    except UnidentifiedImageError:
        raise TryOnError(f"'{filename}' is not a readable image. Use a JPG, PNG, WEBP or BMP photo "
                         "(HEIC/RAW are not supported - export as JPG first).") from None
    except Image.DecompressionBombError:
        raise TryOnError("Image dimensions are unreasonably large.") from None
    except Exception as exc:  # noqa: BLE001
        raise TryOnError(f"Could not decode the image: {exc}") from exc
    im = ImageOps.exif_transpose(im)
    if im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info:
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im.convert("RGBA"), mask=im.convert("RGBA").split()[-1])
        im = bg
    elif im.mode in ("I;16", "I", "F"):
        arr = np.asarray(im, np.float32)
        arr = (255 * (arr - arr.min()) / max(arr.ptp(), 1)).astype(np.uint8)
        im = Image.fromarray(arr).convert("RGB")
    else:
        im = im.convert("RGB")
    w, h = im.size
    if min(w, h) < MIN_SIDE:
        raise TryOnError(f"Image is too small ({w}x{h}px). Please use a photo at least {MIN_SIDE}px on each side.")
    if w * h > MAX_PIXELS:
        s = (MAX_PIXELS / (w * h)) ** 0.5
        im = im.resize((int(w * s), int(h * s)), Image.LANCZOS)
    return np.asarray(im, np.uint8).copy()


def run_tryon(rgb: np.ndarray, opts: TryOnOptions = TryOnOptions(), detector: Optional[FaceDetector] = None) -> TryOnResult:
    if rgb is None or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise TryOnError("Internal error: expected an RGB image.")
    detector = detector or get_detector()
    try:
        glasses = opts.glasses or load_glasses(opts.style)
    except Exception as exc:  # noqa: BLE001
        raise TryOnError(f"Could not load the sunglasses asset: {exc}") from exc

    try:
        faces = detector.detect(rgb, thorough=opts.thorough)
    except Exception as exc:  # noqa: BLE001
        log.exception("face detection failed")
        raise TryOnError(f"Face detection failed unexpectedly: {exc}") from exc
    if not faces:
        raise TryOnError(
            "No face was found. Tips: use a clear photo where the eyes are visible, face the camera "
            "(or turn at most ~60 degrees), avoid heavy shadows, hands or masks over the eyes, and make sure the face "
            "is not tiny in the frame. Nothing was drawn.")
    if opts.face_index is not None:
        if not 0 <= opts.face_index < len(faces):
            raise TryOnError(f"Face #{opts.face_index + 1} does not exist; {len(faces)} face(s) found.")
        faces = [faces[opts.face_index]]

    out = rgb.copy()
    reports, global_warn, debug = [], [], None
    for i, face in enumerate(faces):
        fr = face_frame(face.pts)
        warns = []
        if abs(fr.yaw) > MAX_YAW_ABS:
            msg = (f"Face #{i+1}: head turned {abs(fr.yaw):.0f} degrees - too extreme for reliable placement, skipped.")
            global_warn.append(msg)
            reports.append(FaceReport(i, face.source, fr.yaw, fr.pitch, fr.roll, fr.pupil_dist, 0, 0, [msg]))
            continue
        if abs(fr.yaw) > WARN_YAW_ABS:
            warns.append(f"Head turned ~{abs(fr.yaw):.0f} degrees: depth estimate is rough, expect some misalignment.")
        if fr.pupil_dist < 30:
            warns.append(f"Face is small ({fr.pupil_dist:.0f}px between pupils); result will be low-detail.")
        if abs(fr.roll) > 35:
            warns.append(f"Strong head tilt ({fr.roll:.0f} degrees).")
        x0, y0, x1, y1 = face.bbox
        if x0 < -0.02 * rgb.shape[1] or y0 < -0.02 * rgb.shape[0] or x1 > 1.02 * rgb.shape[1] or y1 > 1.02 * rgb.shape[0]:
            warns.append("Face is partly outside the frame.")
        if face.source != "direct":
            warns.append(f"Face needed a fallback detector ({face.source}); landmarks may be less accurate.")
        opacity, scale = opts.lens_opacity, opts.scale
        if opts.wearing_glasses:
            opacity, scale = max(opacity, 0.95), scale * 1.08
            warns.append("Existing-glasses mode: frames of the original glasses may still peek out around the edges.")
        params = RenderParams(scale=scale, dy=opts.dy, lens_opacity=opacity, shadow=opts.shadow,
                              match_photo=opts.match_photo, debug=opts.debug, draw_arms=opts.draw_arms)
        try:
            out, info = render_glasses(out, face, glasses, params)
        except Exception as exc:  # noqa: BLE001
            log.exception("render failed")
            msg = f"Face #{i+1}: could not render sunglasses ({exc})."
            global_warn.append(msg)
            reports.append(FaceReport(i, face.source, fr.yaw, fr.pitch, fr.roll, fr.pupil_dist, 0, [msg]))
            continue
        if opts.debug and "debug_image" in info:
            debug = info["debug_image"]
        reports.append(FaceReport(i, face.source, fr.yaw, fr.pitch, fr.roll, fr.pupil_dist, info["luminance"], warns))

    rendered = [r for r in reports if not any(w.startswith(f"Face #{r.index+1}:") for w in r.warnings)]
    if not rendered:
        raise TryOnError("A face was found but sunglasses could not be placed reliably: " + " ".join(global_warn))
    return TryOnResult(rgb, out, reports, global_warn, debug)

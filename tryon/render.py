"""Photo-realistic compositing of an RGBA sunglasses asset onto a face.

Placement model (weak-perspective):
  * the asset is a flat plate in the face frame, `FRONT_OFFSET * P` in front of the pupils,
    scaled so its lens centres sit on the pupils (blended with temple width for wide faces);
  * three plate corners are projected with the 3-D face frame -> an affine matrix handles roll,
    yaw foreshortening/shear and pitch in a single warp of the premultiplied RGBA asset;
  * temple arms are drawn from the hinges backwards toward the ears; the far arm is hidden by the
    face silhouette;
  * realism: soft contact shadow on the skin, brightness / colour / blur / noise matched to the photo.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .assets import Glasses
from .landmarks import FACE_OVAL, IRIS_L, IRIS_R, Face
from .pose import FaceFrame, face_frame

FRONT_OFFSET = 0.24   # plate distance in front of pupils, in units of pupil distance
ARM_DEPTH = 2.05      # temple arm length (hinge -> ear top) in units of pupil distance
ARM_DROP = 0.30       # how far the arm end sits below the hinge (ear is lower), units of P


@dataclass
class RenderParams:
    scale: float = 1.0          # user size multiplier
    dy: float = 0.0             # vertical offset in units of pupil distance (+ = down)
    lens_opacity: float = 0.90  # 0.3 .. 1.0
    shadow: float = 0.30        # strength of the shadow cast on the face
    match_photo: bool = True    # match brightness / noise / blur to the photo
    draw_arms: bool = True
    debug: bool = False


def _project(p3):
    return np.asarray(p3, np.float64)[..., :2]


def face_mask(shape, pts, expand=0.0):
    h, w = shape[:2]
    poly = pts[FACE_OVAL, :2].astype(np.float32)
    if expand:
        c = poly.mean(0)
        poly = c + (poly - c) * (1 + expand)
    m = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(m, np.round(poly).astype(np.int32), 255)
    return m.astype(np.float32) / 255


def _gain(lum):
    """Brightness multiplier for the glasses so they sit in the photo's exposure (dark photo -> dark glasses)."""
    return float(np.clip(1.35 * max(lum, 1e-3) ** 0.65, 0.10, 1.10))


def estimate_photo_stats(rgb, pts):
    """Luminance (0-1), mean colour cast, noise sigma (0-255 scale) and softness from the skin region."""
    m = face_mask(rgb.shape, pts)
    m = cv2.erode(m, np.ones((9, 9), np.uint8)) > 0.5
    if m.sum() < 50:
        m = face_mask(rgb.shape, pts) > 0.5
    if m.sum() < 10:
        return 0.5, np.ones(3), 0.0, 0.0
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lum = float(gray[m].mean() / 255)
    mean_rgb = rgb[m].mean(0)
    cast = mean_rgb / max(mean_rgb.mean(), 1e-3)
    lap = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)
    vals = lap[m]
    sigma_n = float(1.4826 * np.median(np.abs(vals - np.median(vals))) / np.sqrt(20))
    sharp = float(np.var(vals))
    return lum, cast, min(sigma_n, 12.0), sharp


def render_glasses(rgb: np.ndarray, face: Face, glasses: Glasses, params: RenderParams = RenderParams()):
    """Return (output_rgb_uint8, info dict)."""
    H, W = rgb.shape[:2]
    fr: FaceFrame = face_frame(face.pts)
    pts = face.pts.astype(np.float64)
    Wa, Ha = glasses.size

    # ---- scale: lens centres on the pupils, widened slightly for wide faces -------------------
    lens_dx = (glasses.lens_cx[1] - glasses.lens_cx[0]) * Wa
    s_pupil = 1.10 * fr.pupil_dist / lens_dx            # real frames are ~10 % wider than the pupil spacing implies
    s_temple = np.clip(fr.temple_dist * 1.0 / Wa, 0.90 * s_pupil, 1.25 * s_pupil)
    s = (0.5 * s_pupil + 0.5 * s_temple) * params.scale

    O = fr.origin + FRONT_OFFSET * fr.pupil_dist * fr.z + params.dy * fr.pupil_dist * fr.y
    v0 = glasses.lens_cy * Ha
    u0 = Wa / 2

    def plate(u, v):
        return O + ((u - u0) * s) * fr.x + ((v - v0) * s) * fr.y

    # ---- effective asset alpha (lens opacity control) & premultiplied colour -----------------
    a = glasses.alpha.copy()
    if glasses.lens_mask.max() > 0:
        lm = glasses.lens_mask
        # Sunglass lenses are transparent optical material, not opaque stickers. Keep
        # the frame solid while allowing the eyes, skin, and scene lighting to show
        # through the lens tint.
        lens_alpha = 0.30 + 0.45 * float(np.clip(params.lens_opacity, 0.0, 1.0))
        frame_alpha = np.clip(a - lm, 0.0, 1.0)
        a = np.maximum(frame_alpha, lm * lens_alpha)
    rgb_g = glasses.rgb.copy()

    # ---- photo statistics for matching ------------------------------------------------------
    lum, cast, noise_sigma, sharp = estimate_photo_stats(rgb, face.pts)
    if params.match_photo:
        gain = _gain(lum)
        tint = 1 + 0.5 * (cast - 1)
        rgb_g = np.clip(rgb_g * gain * tint[None, None, :], 0, 1)
    pm = np.dstack([rgb_g * a[..., None], a]).astype(np.float32)   # premultiplied RGBA

    # ---- affine: asset pixels -> image pixels ----------------------------------------------
    P0, P1, P2 = _project(plate(0, 0)), _project(plate(Wa, 0)), _project(plate(0, Ha))
    M = np.array([[P1[0] - P0[0], P2[0] - P0[0], P0[0]],
                  [P1[1] - P0[1], P2[1] - P0[1], P0[1]]]) 
    M[:, :2] /= np.array([[Wa, Ha], [Wa, Ha]])
    corners = np.array([[0, 0], [Wa, 0], [Wa, Ha], [0, Ha]], np.float64)
    img_corners = corners @ M[:, :2].T + M[:, 2]

    # ---- temple arms ------------------------------------------------------------------------
    arm_polys = []
    near_side = None
    if params.draw_arms:
        thick = 0.075 * fr.pupil_dist
        hinge_y_off = (glasses.hinge_v * Ha - v0) * s
        for side in (-1, +1):
            hinge = O + side * (Wa / 2) * s * fr.x + hinge_y_off * fr.y
            end = hinge - ARM_DEPTH * fr.pupil_dist * fr.z + ARM_DROP * fr.pupil_dist * fr.y \
                - side * 0.04 * fr.pupil_dist * fr.x
            quad = [hinge - thick * fr.y, hinge + 1.2 * thick * fr.y, end + 0.7 * thick * fr.y, end - 0.5 * thick * fr.y]
            arm_polys.append((side, np.array([_project(q) for q in quad]), hinge, end))
        # near side = hinge closer to the camera (smaller z)
        near_side = -1 if arm_polys[0][2][2] < arm_polys[1][2][2] else +1

    # ---- ROI -----------------------------------------------------------------------------
    allp = [img_corners] + [p[1] for p in arm_polys]
    allp = np.vstack(allp)
    margin = int(0.5 * fr.pupil_dist) + 6
    x0 = int(max(0, np.floor(allp[:, 0].min()) - margin)); x1 = int(min(W, np.ceil(allp[:, 0].max()) + margin))
    y0 = int(max(0, np.floor(allp[:, 1].min()) - margin)); y1 = int(min(H, np.ceil(allp[:, 1].max()) + margin))
    if x1 - x0 < 8 or y1 - y0 < 8:
        raise ValueError("Computed sunglasses position falls outside the image.")
    rw, rh = x1 - x0, y1 - y0

    # pre-shrink the asset to ~ target resolution to avoid aliasing, then warp (premultiplied)
    sx, sy = np.hypot(*M[:, 0]), np.hypot(*M[:, 1])
    f = float(np.clip(max(sx, sy) * 1.0, 0.03, 1.0))
    pm_s = cv2.resize(pm, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else pm
    Mr = M.copy()
    Mr[:, :2] = M[:, :2] / f
    Mr[:, 2] -= [x0, y0]
    front = cv2.warpAffine(pm_s, Mr, (rw, rh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # ---- arms layer (4x supersampled anti-aliased polygons) ---------------------------------
    SSF = 4
    arms = np.zeros((rh, rw, 4), np.float32)
    fm_roi = face_mask(rgb.shape, face.pts, expand=0.0)[y0:y1, x0:x1]
    if arm_polys:
        for side, poly, hinge, end in arm_polys:
            if np.linalg.norm(poly[2] - poly[0]) < 2.0:
                continue
            m = np.zeros((rh * SSF, rw * SSF), np.uint8)
            cv2.fillConvexPoly(m, np.round((poly - [x0, y0]) * SSF).astype(np.int32), 255, lineType=cv2.LINE_AA)
            am = cv2.resize(m, (rw, rh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
            if side != near_side:                          # far arm is hidden behind the face
                hide = cv2.GaussianBlur(face_mask(rgb.shape, face.pts, expand=-0.04)[y0:y1, x0:x1], (0, 0), 1.2)
                am = am * (1 - hide)
            col = np.array(glasses.frame_rgb, np.float32)
            col = col * (_gain(lum) if params.match_photo else 1.0)
            arms[..., :3] = np.maximum(arms[..., :3], col[None, None, :] * am[..., None])
            arms[..., 3] = np.maximum(arms[..., 3], am)

    # combine: front over arms
    fa = front[..., 3:4]
    layer = front + arms * (1 - fa)
    alpha = layer[..., 3]

    # ---- match blur & noise to the photo -----------------------------------------------------
    if params.match_photo:
        blur = float(0.6 + 0.9 * np.clip((60.0 - sharp) / 60.0, 0, 1))   # softer photo -> softer glasses
        layer = cv2.GaussianBlur(layer, (0, 0), blur)
        if noise_sigma > 0.8:
            rng = np.random.default_rng(1234)
            n = rng.normal(0, noise_sigma * 0.8 / 255.0, (rh, rw, 1)).astype(np.float32)
            layer[..., :3] = np.clip(layer[..., :3] + n * layer[..., 3:4], 0, None)
        alpha = layer[..., 3]

    # ---- shadow cast on skin ------------------------------------------------------------------
    roi = rgb[y0:y1, x0:x1].astype(np.float32) / 255
    if params.shadow > 0:
        off = (0.0, 0.09 * fr.pupil_dist)
        Msh = np.float32([[1, 0, off[0]], [0, 1, off[1]]])
        sh = cv2.warpAffine(alpha, Msh, (rw, rh))
        sh = cv2.GaussianBlur(sh, (0, 0), max(1.0, 0.10 * fr.pupil_dist))
        skin = cv2.GaussianBlur(fm_roi, (0, 0), 1.5)
        roi = roi * (1 - params.shadow * sh * skin)[..., None]

    # ---- composite (premultiplied "over") ------------------------------------------------
    out_roi = roi * (1 - alpha[..., None]) + layer[..., :3]
    out = rgb.copy()
    out[y0:y1, x0:x1] = np.clip(out_roi * 255 + 0.5, 0, 255).astype(np.uint8)

    info = dict(yaw=fr.yaw, pitch=fr.pitch, roll=fr.roll, pupil_dist_px=fr.pupil_dist, scale_px_per_asset_px=float(s),
                roi=(x0, y0, x1, y1), luminance=lum, noise_sigma=noise_sigma, near_side=near_side,
                lens_centres=[_project(plate(glasses.lens_cx[i] * Wa, v0)).tolist() for i in (0, 1)])
    if params.debug:
        dbg = out.copy()
        for idx in (IRIS_R, IRIS_L):
            cv2.circle(dbg, tuple(np.round(face.pts[idx, :2]).astype(int)), max(2, int(fr.pupil_dist * 0.05)), (0, 255, 0), -1)
        for c in info["lens_centres"]:
            cv2.drawMarker(dbg, tuple(np.round(c).astype(int)), (255, 0, 0), cv2.MARKER_CROSS, int(fr.pupil_dist * 0.3), 2)
        cv2.polylines(dbg, [np.round(img_corners).astype(np.int32)], True, (255, 255, 0), 1)
        info["debug_image"] = dbg
    return out, info

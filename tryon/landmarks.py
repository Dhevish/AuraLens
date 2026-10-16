"""Face landmark detection with MediaPipe FaceMesh (478 points incl. iris) plus fallbacks.

Detection strategy (stops at the first stage that finds a plausible face):
  1. direct run on the (possibly down-scaled) image
  2. contrast-enhanced copy (CLAHE + gamma) - helps dark / low-light photos
  3. rotated copies (+-25, 50, 90, 180 deg) - helps strongly tilted / sideways heads
  4. overlapping zoomed tiles - helps small faces in large photos / group shots
Landmarks are always mapped back into the original image's pixel coordinates.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

import cv2
import numpy as np

# MediaPipe landmark indices
IRIS_R, IRIS_L = 468, 473            # subject's right / left iris centre
EYE_OUTER_R, EYE_OUTER_L = 33, 263
EYE_INNER_R, EYE_INNER_L = 133, 362
TEMPLE_R, TEMPLE_L = 127, 356
NOSE_TIP, NOSE_BRIDGE, CHIN, FOREHEAD = 1, 168, 152, 10
MOUTH_R, MOUTH_L = 61, 291
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152,
             148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]


@dataclass
class Face:
    pts: np.ndarray                  # (478, 3): x, y in px of the ORIGINAL image; z in px-scale (smaller = closer)
    source: str = "direct"           # which stage found it
    notes: list = field(default_factory=list)

    @property
    def bbox(self):
        x0, y0 = self.pts[:, :2].min(0)
        x1, y1 = self.pts[:, :2].max(0)
        return float(x0), float(y0), float(x1), float(y1)

    @property
    def width(self):
        x0, _, x1, _ = self.bbox
        return x1 - x0


def enhance_for_detection(rgb: np.ndarray) -> np.ndarray:
    """CLAHE on luminance plus automatic gamma lift for dark images (detection only)."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    mean = l.mean() / 255.0
    gamma = 1.0
    if mean < 0.4:
        gamma = max(0.35, np.log(0.5) / np.log(max(mean, 0.02)))
        l = (255.0 * (l / 255.0) ** gamma).astype(np.uint8)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)


def plausible(pts: np.ndarray) -> bool:
    """Reject geometry that cannot be a real face (guards against false positives)."""
    try:
        eye_d = np.linalg.norm(pts[EYE_OUTER_L, :2] - pts[EYE_OUTER_R, :2])
        face_w = np.linalg.norm(pts[454, :2] - pts[234, :2])
        face_h = np.linalg.norm(pts[CHIN, :2] - pts[FOREHEAD, :2])
        if eye_d < 8 or face_w < 20 or face_h < 20:
            return False
        eye_mid = (pts[EYE_OUTER_L, :2] + pts[EYE_OUTER_R, :2]) / 2
        up = pts[FOREHEAD, :2] - pts[CHIN, :2]
        up /= np.linalg.norm(up) + 1e-9
        # eyes must lie between chin and forehead along the face axis, nose below eyes
        t_eye = np.dot(eye_mid - pts[CHIN, :2], up) / (face_h + 1e-9)
        t_nose = np.dot(pts[NOSE_TIP, :2] - pts[CHIN, :2], up) / (face_h + 1e-9)
        return 0.35 < t_eye < 0.95 and t_nose < t_eye and 0.25 < eye_d / (face_w + 1e-9) < 1.2 and 0.4 < face_h / (face_w + 1e-9) < 2.6
    except Exception:  # noqa: BLE001
        return False


class FaceDetector:
    """Thread-safe wrapper around mediapipe.solutions.face_mesh.FaceMesh."""

    def __init__(self, max_faces: int = 5):
        try:
            import mediapipe as mp
            self._mp_mesh = mp.solutions.face_mesh
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "MediaPipe could not be imported. Install the pinned version with "
                "`pip install mediapipe==0.10.14` (see requirements.txt)."
            ) from exc
        self.max_faces = max_faces
        self._lock = threading.Lock()
        self._strict = self._make(0.5)
        self._loose = self._make(0.4)   # 0.3 produced a false "face" on a dog photo

    def _make(self, conf):
        return self._mp_mesh.FaceMesh(static_image_mode=True, max_num_faces=self.max_faces,
                                      refine_landmarks=True, min_detection_confidence=conf)

    def _run(self, rgb, loose=False):
        h, w = rgb.shape[:2]
        with self._lock:
            res = (self._loose if loose else self._strict).process(np.ascontiguousarray(rgb))
        out = []
        for lm in (res.multi_face_landmarks or []):
            p = np.array([[l.x * w, l.y * h, l.z * w] for l in lm.landmark], np.float32)
            if plausible(p):
                out.append(p)
        return out

    # -- fallbacks ---------------------------------------------------------------
    def _rotated(self, rgb, angle, loose):
        h, w = rgb.shape[:2]
        c = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(c, angle, 1.0)
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
        M[0, 2] += nw / 2 - c[0]
        M[1, 2] += nh / 2 - c[1]
        rot = cv2.warpAffine(rgb, M, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        faces = self._run(rot, loose)
        Minv = cv2.invertAffineTransform(M)
        res = []
        for p in faces:
            q = p.copy()
            q[:, :2] = p[:, :2] @ Minv[:, :2].T + Minv[:, 2]
            res.append(q)
        return res

    def _tiles(self, rgb, loose=True):
        h, w = rgb.shape[:2]
        found = []
        short = min(h, w)
        for frac in (0.55, 0.35):
            win = int(short * frac)
            if win < 64:
                continue
            step = max(1, int(win * 0.5))
            for y0 in range(0, max(1, h - win + step), step):
                for x0 in range(0, max(1, w - win + step), step):
                    x1, y1 = min(w, x0 + win), min(h, y0 + win)
                    crop = rgb[y0:y1, x0:x1]
                    scale = 640 / max(crop.shape[:2])
                    if scale > 1:
                        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                    else:
                        scale = 1.0
                    for p in self._run(crop, loose):
                        q = p.copy()
                        q[:, :2] = p[:, :2] / scale + [x0, y0]
                        q[:, 2] = p[:, 2] / scale
                        found.append(q)
        return found

    def _verify(self, rgb, p):
        """Second opinion for weak (tile / low-confidence) detections: re-run the STRICT detector on a zoomed crop
        centred on the candidate. Rejects most false positives and gives more accurate landmarks for small faces."""
        h, w = rgb.shape[:2]
        c = p[:, :2].mean(0)
        wd = float(np.ptp(p[:, 0]))
        half = 1.35 * wd
        x0, y0 = int(max(0, c[0] - half)), int(max(0, c[1] - half))
        x1, y1 = int(min(w, c[0] + half)), int(min(h, c[1] + half))
        crop = rgb[y0:y1, x0:x1]
        if min(crop.shape[:2]) < 16:
            return None
        sc = 512 / max(crop.shape[:2])
        crop = cv2.resize(crop, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC if sc > 1 else cv2.INTER_AREA)
        best = None
        for q in self._run(crop, loose=False):
            r = q.copy()
            r[:, :2] = q[:, :2] / sc + [x0, y0]
            r[:, 2] = q[:, 2] / sc
            d = np.linalg.norm(r[:, :2].mean(0) - c)
            if d < 0.3 * wd and 0.6 < np.ptp(r[:, 0]) / wd < 1.6 and (best is None or d < best[0]):
                best = (d, r)
        return None if best is None else best[1]

    @staticmethod
    def _dedupe(cands):
        keep = []
        for src, p in cands:
            c = p[:, :2].mean(0)
            wd = np.ptp(p[:, 0])
            if any(np.linalg.norm(c - k[1][:, :2].mean(0)) < 0.45 * max(wd, np.ptp(k[1][:, 0])) for k in keep):
                continue
            keep.append((src, p))
        return keep

    # -- public ----------------------------------------------------------------
    def detect(self, rgb: np.ndarray, thorough: bool = False, max_det_side: int = 1600):
        """Return a list[Face] (possibly empty). `thorough` also searches tiles for extra/small faces."""
        h, w = rgb.shape[:2]
        scale = min(1.0, max_det_side / max(h, w))
        small = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else rgb

        def back(plist):
            res = []
            for p in plist:
                q = p.copy()
                q[:, :3] = p[:, :3] / scale
                res.append(q)
            return res

        cands = [("direct", p) for p in back(self._run(small))]
        if not cands:
            enh = enhance_for_detection(small)
            cands = [("enhanced", p) for p in back(self._run(enh))]
            if not cands:
                cands = [("enhanced-loose", p) for p in back(self._run(enh, loose=True))]
        if not cands:
            enh = enhance_for_detection(small)
            for ang in (25, -25, 50, -50, 90, -90, 180):
                found = back(self._rotated(enh, ang, loose=True))
                if found:
                    cands = [(f"rotated{ang:+d}", p) for p in found]
                    break
        if not cands or thorough:
            enh = enhance_for_detection(rgb)
            cands += [("tile", p) for p in self._tiles(enh)]
        verified = []
        for src, p in self._dedupe(cands):
            if src in ("tile", "enhanced-loose"):
                q = self._verify(enhance_for_detection(rgb) if src == "tile" else rgb, p)
                if q is None:
                    continue
                verified.append((src, q))
            else:
                verified.append((src, p))
        faces = [Face(p, src) for src, p in self._dedupe(verified)]
        faces.sort(key=lambda f: -f.width)
        return faces[: self.max_faces]

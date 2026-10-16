"""Head pose from the 3-D landmark cloud -> an orthonormal face frame.

MediaPipe gives every landmark an (x, y, z) triple (z in the same scale as x, smaller = closer to the
camera). Rather than fitting a generic head model with solvePnP, we build a coordinate frame directly
from stable landmarks of THIS face:

  x_hat : outer-eye-corner(right) -> outer-eye-corner(left)   (image-right for an upright face)
  y_hat : eye-midpoint -> chin, made orthogonal to x_hat       (face-down)
  z_hat : toward the camera / out of the face (sign fixed using the nose tip)

Everything is measured in pixels, so scale (pupil distance P) comes for free.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .landmarks import (CHIN, EYE_OUTER_L, EYE_OUTER_R, IRIS_L, IRIS_R, NOSE_TIP, TEMPLE_L, TEMPLE_R)


@dataclass
class FaceFrame:
    origin: np.ndarray   # 3-D midpoint between the two pupils
    x: np.ndarray        # unit vectors (3,), image coordinates, z grows away from the camera
    y: np.ndarray
    z: np.ndarray        # points OUT of the face toward the camera
    pupil_dist: float    # 3-D pupil distance (px)
    temple_dist: float   # 3-D distance between temple landmarks (px)
    yaw: float           # degrees, +ve = face turned toward image-right (subject's left cheek hidden)
    pitch: float         # degrees, +ve = chin up
    roll: float          # degrees, +ve = clockwise in the image


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def face_frame(pts: np.ndarray) -> FaceFrame:
    pts = pts.astype(np.float64)
    iris_r, iris_l = pts[IRIS_R], pts[IRIS_L]
    origin = (iris_r + iris_l) / 2
    eye_mid = (pts[EYE_OUTER_R] + pts[EYE_OUTER_L]) / 2

    x = _unit(pts[EYE_OUTER_L] - pts[EYE_OUTER_R])
    down = pts[CHIN] - eye_mid
    y = _unit(down - np.dot(down, x) * x)
    z = np.cross(x, y)                      # points away from the camera (into the screen)
    z = -z                                  # now toward the camera
    if np.dot(z, pts[NOSE_TIP] - eye_mid) < 0:  # nose tip must lie in front of the eye line
        z = -z
    z = _unit(z)

    yaw = float(np.degrees(np.arcsin(np.clip(-x[2], -1, 1))))
    pitch = float(np.degrees(np.arcsin(np.clip(y[2], -1, 1))))
    roll = float(np.degrees(np.arctan2(x[1], x[0])))
    return FaceFrame(origin, x, y, z,
                     float(np.linalg.norm(iris_l - iris_r)),
                     float(np.linalg.norm(pts[TEMPLE_L] - pts[TEMPLE_R])),
                     yaw, pitch, roll)

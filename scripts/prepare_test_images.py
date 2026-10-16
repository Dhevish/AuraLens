#!/usr/bin/env python
"""Download the public sample photos used by evaluate.py into tests/images/ (max side 1280 px).

Real photos come from public GitHub repos (used for local testing only - they are third-party images,
please do not redistribute). Two cases are *synthetic* variants made here and labelled as such:
  * tilted_angelina_rot25  = real photo rotated by 25 degrees
  * lowlight_cooper_synth  = real photo darkened (gamma 2.4, x0.35) + sensor-like noise
"""
import urllib.request
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "tests" / "images"
GH = "https://raw.githubusercontent.com/"
SRC = {
    "frontal_rudd": GH + "timesler/facenet-pytorch/master/data/test_images/paul_rudd/1.jpg",
    "frontal_obama": GH + "ageitgey/face_recognition/master/examples/obama.jpg",
    "turned_lena": GH + "opencv/opencv/4.x/samples/data/lena.jpg",
    "turned_obama2": GH + "ageitgey/face_recognition/master/examples/obama2.jpg",
    "turned_lin": GH + "ageitgey/face_recognition/master/examples/lin-manuel-miranda.png",
    "dim_alex": GH + "ageitgey/face_recognition/master/examples/alex-lacamoire.png",
    "glasses_trio": GH + "davisking/dlib/master/examples/faces/2008_002506.jpg",
    "glasses_woman": GH + "davisking/dlib/master/examples/faces/2008_001009.jpg",
    "group_trio": GH + "davisking/dlib/master/examples/faces/2008_001322.jpg",
    "group_friends": GH + "timesler/facenet-pytorch/master/data/multiface.jpg",
    "_cooper": GH + "timesler/facenet-pytorch/master/data/test_images/bradley_cooper/1.jpg",
    "_angelina": GH + "timesler/facenet-pytorch/master/data/test_images/angelina_jolie/1.jpg",
    "negative_dogs": GH + "davisking/dlib/master/examples/faces/dogs.jpg",
}


def fetch(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        arr = np.frombuffer(r.read(), np.uint8)
    im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if im is None:
        raise RuntimeError(f"cannot decode {url}")
    s = 1280 / max(im.shape[:2])
    return cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else im


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cache = {}
    for name, url in SRC.items():
        cache[name] = im = fetch(url)
        if not name.startswith("_"):
            cv2.imwrite(str(OUT / f"{name}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
            print("saved", name, im.shape)
    # synthetic: 25-degree head tilt (rotate the whole photo)
    im = cache["_angelina"]
    h, w = im.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 25, 1.0)
    cv2.imwrite(str(OUT / "tilted_angelina_rot25.jpg"),
                cv2.warpAffine(im, M, (w, h), borderMode=cv2.BORDER_REFLECT), [cv2.IMWRITE_JPEG_QUALITY, 92])
    # synthetic: low light
    im = cache["_cooper"].astype(np.float32) / 255
    dark = 0.35 * im ** 2.4
    rng = np.random.default_rng(7)
    dark = np.clip(dark + rng.normal(0, 0.012, dark.shape), 0, 1)
    cv2.imwrite(str(OUT / "lowlight_cooper_synth.jpg"), (dark * 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("done ->", OUT)


if __name__ == "__main__":
    main()

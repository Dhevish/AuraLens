#!/usr/bin/env python
"""Run the real pipeline on every image in tests/images and write results/ (outputs, before/after
screenshots, results.json and results_table.md). Nothing here is hand-edited: it reports what happened."""
import json
import time
from pathlib import Path

import cv2
import numpy as np

from tryon.pipeline import TryOnError, TryOnOptions, load_image_bytes, run_tryon

ROOT = Path(__file__).parent
IMG = ROOT / "tests" / "images"
OUT = ROOT / "results"


def label(img, text):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def side_by_side(a, b, h=640):
    def fit(x):
        s = h / x.shape[0]
        return cv2.resize(x, (max(1, int(x.shape[1] * s)), h), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    return np.hstack([label(fit(a), "BEFORE"), label(fit(b), "AFTER")])


def main():
    OUT.mkdir(exist_ok=True)
    rows = []
    for path in sorted(IMG.glob("*.jpg")):
        name = path.stem
        row = dict(image=path.name, status="", faces=0, seconds=0.0, details=[], error="")
        try:
            rgb = load_image_bytes(path.read_bytes(), path.name)
            t = time.time()
            opts = TryOnOptions(wearing_glasses=name.startswith("glasses"), thorough=name.startswith("group"))
            res = run_tryon(rgb, opts)
            row["seconds"] = round(time.time() - t, 2)
            row["status"] = "rendered"
            row["faces"] = len(res.faces)
            for r in res.faces:
                row["details"].append(dict(face=r.index + 1, source=r.source, yaw=round(r.yaw, 1), pitch=round(r.pitch, 1),
                                           roll=round(r.roll, 1), pupil_px=round(r.pupil_px), warnings=r.warnings))
            cv2.imwrite(str(OUT / f"{name}_after.jpg"), cv2.cvtColor(res.output, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(OUT / f"{name}_compare.jpg"),
                        cv2.cvtColor(side_by_side(res.original, res.output), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        except TryOnError as exc:
            row["status"] = "refused (error shown to user)"
            row["error"] = str(exc)
        except Exception as exc:  # noqa: BLE001  - an unexpected crash is a bug; record it loudly
            row["status"] = "CRASH"
            row["error"] = repr(exc)
        rows.append(row)
        print(f"{path.name:32s} {row['status']:34s} faces={row['faces']} {row['seconds']}s {row['error'][:70]}")
    (OUT / "results.json").write_text(json.dumps(rows, indent=2))
    lines = ["| Image | Result | Faces | yaw / pitch / roll (deg) | pupil px | time (s) | Warnings / error |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["details"]:
            for d in r["details"]:
                lines.append(f"| {r['image']} #{d['face']} | {r['status']} | {r['faces']} | {d['yaw']} / {d['pitch']} / {d['roll']} | "
                             f"{d['pupil_px']} | {r['seconds']} | {'; '.join(d['warnings']) or '-'} |")
        else:
            lines.append(f"| {r['image']} | {r['status']} | 0 | - | - | {r['seconds']} | {r['error']} |")
    (OUT / "results_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

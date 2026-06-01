"""
Build class_names.json from folder IDs + Mixed_breed_dataset_normalized.xlsx.
Each folder N uses all rows whose Image Name starts with "N_" ; breed scores are
averaged and the argmax breed becomes the human-readable label for that class.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import ARTIFACTS, DATA_DIR, EXCEL_PATH  # noqa: E402


def breed_columns(df: pd.DataFrame) -> list[str]:
    skip = {"Image Name", "SUM"}
    cols = []
    for c in df.columns:
        if c in skip or str(c).startswith("Unnamed"):
            continue
        cols.append(c)
    return cols


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--excel", type=Path, default=EXCEL_PATH)
    parser.add_argument("--out", type=Path, default=ARTIFACTS / "class_names.json")
    args = parser.parse_args()

    if not args.excel.is_file():
        raise SystemExit(f"Excel not found: {args.excel}")

    df = pd.read_excel(args.excel)
    df["Image Name"] = df["Image Name"].astype(str)
    breeds = breed_columns(df)

    folder_ids: list[int] = []
    for p in args.data_dir.iterdir():
        if p.is_dir() and p.name.isdigit():
            folder_ids.append(int(p.name))
    folder_ids.sort()

    if not folder_ids:
        raise SystemExit(f"No numeric class folders under {args.data_dir}")

    classes: list[dict] = []
    for fid in folder_ids:
        prefix = f"{fid}_"
        sub = df[df["Image Name"].str.startswith(prefix)]
        if len(sub) == 0:
            label = f"class_{fid}"
        else:
            means = sub[breeds].astype(float).mean(axis=0)
            label = str(means.idxmax())

        classes.append({"folder_id": fid, "label": label})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "data_dir": str(args.data_dir.resolve()),
        "num_classes": len(classes),
        "classes": classes,
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.out} with {len(classes)} classes.")


if __name__ == "__main__":
    main()

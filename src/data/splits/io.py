from pathlib import Path

import pandas as pd

from .temporal import TemporalSplit


def save_split(split: TemporalSplit, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split.train.to_parquet(out_dir / "train.parquet", index=False)
    split.val.to_parquet(out_dir / "val.parquet", index=False)
    split.test.to_parquet(out_dir / "test.parquet", index=False)
    print(f"Saved: train={len(split.train)}, "
          f"val={len(split.val)}, test={len(split.test)}")


def load_split(in_dir: Path) -> TemporalSplit:
    in_dir = Path(in_dir)
    return TemporalSplit(
        train=pd.read_parquet(in_dir / "train.parquet"),
        val=pd.read_parquet(in_dir / "val.parquet"),
        test=pd.read_parquet(in_dir / "test.parquet"),
    )
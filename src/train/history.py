from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def history_to_dataframe(history: Iterable[dict]) -> pd.DataFrame:
    """
    Convert list of epoch metric dicts to a pandas DataFrame.
    """
    return pd.DataFrame(list(history))


def save_history_csv(history: Iterable[dict], out_path: str | Path) -> Path:
    """
    Save training history to CSV.
    """
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    df = history_to_dataframe(history)
    df.to_csv(p, index=False)
    return p
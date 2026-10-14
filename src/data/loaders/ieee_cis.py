from pathlib import Path

import pandas as pd

from .base import BaseLoader


class IEEECISLoader(BaseLoader):
    name = "ieee_cis"
    target_col = "isFraud"
    time_col = "TransactionDT"

    # Колонки, которые в identity-файле
    IDENTITY_COLS = ["id_01", "id_02", "id_03", "id_04", "id_05", "id_06",
                     "id_07", "id_08", "id_09", "id_10", "id_11", "id_12",
                     "id_13", "id_14", "id_15", "id_16", "id_17", "id_18",
                     "id_19", "id_20", "id_21", "id_22", "id_23", "id_24",
                     "id_25", "id_26", "id_27", "id_28", "id_29", "id_30",
                     "id_31", "id_32", "id_33", "id_34", "id_35", "id_36",
                     "id_37", "id_38", "DeviceType", "DeviceInfo"]

    def __init__(self, data_dir: Path, cache: bool = True):
        super().__init__(data_dir, cache)

    def load_raw(self) -> pd.DataFrame:
        tx_path = self.data_dir / "train_transaction.csv"
        id_path = self.data_dir / "train_identity.csv"

        tx = pd.read_csv(tx_path, low_memory=False)
        id_ = pd.read_csv(id_path, low_memory=False)

        for c in ("DeviceType", "DeviceInfo"):
            if c in id_.columns:
                id_[c] = id_[c].astype("category")

        # Защита от повторного merge
        if any(c.startswith("id_") for c in tx.columns):
            df = tx
        else:
            df = tx.merge(id_, on="TransactionID", how="left")

        # Downcast float64 → float32
        float_cols = df.select_dtypes(include=["float64"]).columns.tolist()
        print(f"[downcast] float64 columns: {len(float_cols)}")   # временный принт
        for c in float_cols:
            df[c] = df[c].astype("float32")

        # Downcast int64 → int32 (где безопасно)
        int_cols = df.select_dtypes(include=["int64"]).columns.tolist()
        for c in int_cols:
            if df[c].min() >= -2**31 and df[c].max() <= 2**31 - 1:
                df[c] = df[c].astype("int32")

        return df


    def get_schema(self) -> dict:
        cols = pd.read_csv(self.data_dir / "train_transaction.csv",
                           nrows=0).columns.tolist()
        return {
            "target": self.target_col,
            "time": self.time_col,
            "numeric": [c for c in ["TransactionAmt", "dist1", "dist2"]
                        if c in cols],
            "categorical": [c for c in ["ProductCD", "card4", "card6",
                                         "P_emaildomain", "DeviceType",
                                         "DeviceInfo"]],
            "v_features": [c for c in cols if c.startswith("V")],
            "id_features": self.IDENTITY_COLS,
            "card_features": [c for c in cols if c.startswith("card")],
            "d_features": [c for c in cols if c.startswith("D")],
            "c_features": [c for c in cols if c.startswith("C")],
            "m_features": [c for c in cols if c.startswith("M")],
        }

from __future__ import annotations
import logging

import numpy as np
import pandas as pd
import xgboost as xgb

from . import config

logger = logging.getLogger(__name__)


def _prepare_features(feat_row: pd.Series, event_dir: int, scanner_score: float,
                      event_features: dict | None) -> pd.Series:
    """Apply directional + event-aligned features (shared across all model types)."""
    row = feat_row.copy()
    d = event_dir

    if "obi" in row.index:
        row["dir_obi"] = row["obi"] * d
    if "aggressor_ratio" in row.index:
        row["dir_aggressor"] = (row["aggressor_ratio"] - 0.5) * d
    if "net_taker_vol_ratio" in row.index:
        row["dir_net_taker"] = row["net_taker_vol_ratio"] * d
    for key in ["ema20_dist", "ema50_dist"]:
        if key in row.index:
            row[f"dir_{key}"] = row[key] * d
    if "z_vwap" in row.index:
        row["dir_z_vwap"] = row["z_vwap"] * d
    for key in ["ret_1", "ret_5", "ret_20"]:
        if key in row.index:
            row[f"dir_{key}"] = row[key] * d

    row["event_dir"] = event_dir
    row["scanner_score"] = scanner_score
    if event_features:
        row["sec_in_bar"] = event_features.get("sec_in_bar", 59.0)
        row["event_return"] = event_features.get("event_return", 0.0)
        row["event_effort_vs_result"] = event_features.get("event_effort_vs_result", 0.0)
        row["event_rejection_strength"] = event_features.get("event_rejection_strength", 0.0)
        row["time_to_reject_s"] = event_features.get("time_to_reject_s", 15.0)
    else:
        row["sec_in_bar"] = 59.0
        row["event_return"] = 0.0
        row["event_effort_vs_result"] = 0.0
        row["event_rejection_strength"] = 0.0
        row["time_to_reject_s"] = 15.0

    row["dir_event_return"] = row["event_return"] * d
    return row


class ModelInference:
    """Loads model(s) based on config.MODEL_TYPE and runs prediction."""

    def __init__(self, threshold: float = config.THRESHOLD):
        self.threshold = threshold
        self._model_type = config.MODEL_TYPE
        self._feature_names: list[str] | None = None

        if self._model_type == "ensemble":
            self._load_ensemble()
        elif self._model_type == "stacked":
            self._load_stacked()
        elif self._model_type == "lgb":
            self._load_lgb()
        elif self._model_type == "catboost":
            self._load_catboost()
        else:
            self._load_xgb()

        logger.info(f"Model loaded: type={self._model_type}, threshold={threshold}")

    # --- loaders ---

    def _load_xgb(self):
        # Try multi-model path first, fall back to legacy single-model path
        import os
        path = config.MODEL_PATH_XGB if os.path.exists(config.MODEL_PATH_XGB) else config.MODEL_PATH
        self._booster = xgb.Booster()
        self._booster.load_model(path)
        self._feature_names = self._booster.feature_names

    def _load_lgb(self):
        import lightgbm as lgb
        self._lgb_model = lgb.Booster(model_file=config.MODEL_PATH_LGB)
        self._feature_names = self._lgb_model.feature_name()

    def _load_catboost(self):
        from catboost import CatBoostClassifier
        self._cb_model = CatBoostClassifier()
        self._cb_model.load_model(config.MODEL_PATH_CB)
        self._feature_names = self._cb_model.feature_names_

    def _load_ensemble(self):
        self._load_xgb()
        import lightgbm as lgb
        self._lgb_model = lgb.Booster(model_file=config.MODEL_PATH_LGB)
        from catboost import CatBoostClassifier
        self._cb_model = CatBoostClassifier()
        self._cb_model.load_model(config.MODEL_PATH_CB)
        # Use XGBoost feature names as canonical order
        # (all models trained on same feature set)

    def _load_stacked(self):
        """Load all 3 base models + meta-learner for stacking ensemble."""
        self._load_ensemble()
        import pickle
        meta_path = str(config.PROJECT_ROOT / "model_meta_learner.pkl")
        with open(meta_path, "rb") as f:
            self._meta_model = pickle.load(f)
        logger.info(f"Stacking meta-learner loaded from {meta_path}")

    # --- prediction helpers ---

    def _extract_values(self, row: pd.Series) -> list[float]:
        values = []
        for fname in self._feature_names:
            v = row.get(fname, np.nan)
            values.append(float(v) if np.isfinite(v) else np.nan)
        return values

    def _predict_xgb(self, row: pd.Series) -> float:
        values = self._extract_values(row)
        dmat = xgb.DMatrix(np.array([values]), feature_names=self._feature_names, missing=np.nan)
        return float(self._booster.predict(dmat)[0])

    def _predict_lgb(self, row: pd.Series) -> float:
        feat_names = self._lgb_model.feature_name()
        values = [float(row.get(f, np.nan)) if np.isfinite(row.get(f, np.nan)) else np.nan for f in feat_names]
        return float(self._lgb_model.predict(np.array([values]))[0])

    def _predict_catboost(self, row: pd.Series) -> float:
        feat_names = self._cb_model.feature_names_
        values = [float(row.get(f, np.nan)) if np.isfinite(row.get(f, np.nan)) else np.nan for f in feat_names]
        return float(self._cb_model.predict_proba(np.array([values]))[0, 1])

    # --- public API ---

    def predict(self, feat_row: pd.Series, event_dir: int, scanner_score: float,
                event_features: dict | None = None) -> dict:
        row = _prepare_features(feat_row, event_dir, scanner_score, event_features)

        if self._model_type == "stacked":
            xgb_p = self._predict_xgb(row)
            lgb_p = self._predict_lgb(row)
            cb_p = self._predict_catboost(row)
            meta_input = np.array([[xgb_p, lgb_p, cb_p]])
            prob = float(self._meta_model.predict_proba(meta_input)[0, 1])
        elif self._model_type == "ensemble":
            prob = (self._predict_xgb(row) + self._predict_lgb(row) + self._predict_catboost(row)) / 3.0
        elif self._model_type == "lgb":
            prob = self._predict_lgb(row)
        elif self._model_type == "catboost":
            prob = self._predict_catboost(row)
        else:
            prob = self._predict_xgb(row)

        signal = prob >= self.threshold
        return {
            "prob": prob,
            "signal": signal,
            "direction": "long" if event_dir == 1 else "short",
            "threshold": self.threshold,
        }

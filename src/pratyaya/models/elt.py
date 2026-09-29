"""Effective Lead Time (ELT) Transform and Bust Direction Head (Nodes G4, G5, docs/03 §7.3, docs/06 §1.1).

Converts calibrated bust probabilities into Effective Lead Time (ELT) by inverting the NWP
climatological bust curve p_bar(L). Translates abstract probability into actionable operational units:
    e.g. "Day 6 forecast behaving like Day 9.4".
Enforces strict edge-case handling:
    - Below Day 1: "better than a typical Day-1 forecast" (clipped to 0.5)
    - Above Day 12: "beyond our measured range" (clipped to 12.0)

Implements the 3-class Bust Direction Head:
    - 'wet': Reality wetter than forecast (signed_expect < -deadband)
    - 'dry': Reality drier than forecast (signed_expect > +deadband)
    - 'none': Normal variance or unresolvable direction
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from pratyaya.api.schemas import (
    BAND_THRESHOLDS,
    BustDirection,
    ConfidenceBand,
    classify_confidence_band,
)

logger = logging.getLogger(__name__)

ELT_MIN_CLIP: float = 0.5
ELT_MAX_CLIP: float = 12.0
DEFAULT_DIRECTION_DEADBAND: float = 2.0  # mm/day


@dataclass
class ELTResult:
    """Detailed Effective Lead Time evaluation result for a region-lead prediction."""
    effective_lead_time: float
    nominal_lead_day: int
    p_calibrated: float
    headline: str
    is_better_than_day1: bool
    is_beyond_range: bool
    confidence_band: ConfidenceBand
    direction: BustDirection
    signed_expect: float


class EffectiveLeadTimeTransformer:
    """Inverts the climatological bust curve p_bar(L) into Effective Lead Time (ELT) (docs/03 §7.3).

    Monotone in lead time:
        ELT(p) = L' such that p_bar(L') = p_calibrated.
    """

    def __init__(
        self,
        min_clip: float = ELT_MIN_CLIP,
        max_clip: float = ELT_MAX_CLIP,
    ) -> None:
        self.min_clip = min_clip
        self.max_clip = max_clip

        self.leads_: np.ndarray = np.arange(1, 11, dtype=float)
        self.national_curve_: Optional[np.ndarray] = None  # Monotone p_bar(L) for L in 1..10
        self.regional_curves_: Dict[str, np.ndarray] = {}
        self.is_fitted: bool = False

    def fit(
        self,
        train_df: pd.DataFrame,
        bust_col: str = "is_bust",
        lead_col: str = "lead_day",
        region_col: str = "region_id",
    ) -> EffectiveLeadTimeTransformer:
        """Estimate the climatological bust curve p_bar(L) from training years.

        Args:
            train_df: Training fold DataFrame.
            bust_col: Binary bust column.
            lead_col: Lead day column (1..10).
            region_col: Region identifier column.

        Returns:
            EffectiveLeadTimeTransformer: Self instance.
        """
        df = train_df.copy()
        df["lead_day_int"] = df[lead_col].astype(int)
        df["is_bust_int"] = df[bust_col].astype(float)

        # 1. Compute national climatological bust curve across leads 1..10
        national_grp = df.groupby("lead_day_int")["is_bust_int"].mean()
        national_p = []
        for lead in range(1, 11):
            val = float(national_grp.get(lead, 0.04 * (lead / 5.0)))
            national_p.append(val)

        # Enforce strict monotonicity via cumulative maximum / isotonic smoothing
        self.national_curve_ = np.maximum.accumulate(np.array(national_p, dtype=float))
        # Ensure non-zero slope
        for i in range(1, len(self.national_curve_)):
            if self.national_curve_[i] <= self.national_curve_[i - 1]:
                self.national_curve_[i] = self.national_curve_[i - 1] + 0.001

        # 2. Compute regional curves where sufficient data exists
        self.regional_curves_ = {}
        if region_col in df.columns:
            for reg, r_df in df.groupby(region_col):
                r_grp = r_df.groupby("lead_day_int")["is_bust_int"].mean()
                r_p = []
                for lead in range(1, 11):
                    val = float(r_grp.get(lead, self.national_curve_[lead - 1]))
                    r_p.append(val)
                r_curve = np.maximum.accumulate(np.array(r_p, dtype=float))
                for i in range(1, len(r_curve)):
                    if r_curve[i] <= r_curve[i - 1]:
                        r_curve[i] = r_curve[i - 1] + 0.001
                self.regional_curves_[str(reg)] = r_curve

        self.is_fitted = True
        logger.info(
            "Fitted EffectiveLeadTimeTransformer: national curve range [%.4f at D1 -> %.4f at D10]",
            self.national_curve_[0],
            self.national_curve_[-1],
        )
        return self

    def transform_single(
        self,
        p_calibrated: float,
        nominal_lead_day: int,
        region_id: Optional[str] = None,
        signed_expect: float = 0.0,
        direction_deadband: float = DEFAULT_DIRECTION_DEADBAND,
    ) -> ELTResult:
        """Invert calibrated bust probability into Effective Lead Time for a single prediction.

        Args:
            p_calibrated: Calibrated bust probability in [0.0, 1.0].
            nominal_lead_day: The actual forecast lead day (1..10).
            region_id: Optional subdivision ID for regional curve lookup.
            signed_expect: Analogue signed error expectation (forecast - observed).
            direction_deadband: Threshold magnitude for wet/dry determination.

        Returns:
            ELTResult: Dataclass containing ELT, headline, flags, confidence band, and direction.
        """
        if not self.is_fitted or self.national_curve_ is None:
            raise RuntimeError("EffectiveLeadTimeTransformer must be fitted before transform.")

        curve = self.national_curve_
        if region_id is not None and str(region_id) in self.regional_curves_:
            curve = self.regional_curves_[str(region_id)]

        p_val = float(np.clip(p_calibrated, 0.0, 1.0))
        p_min = curve[0]
        p_max = curve[-1]

        # Inversion via linear interpolation
        if p_val <= p_min:
            # Below Day 1 baseline
            if p_min > 1e-6:
                elt_raw = 1.0 * (p_val / p_min)
            else:
                elt_raw = 1.0
        elif p_val >= p_max:
            # Above Day 10 baseline: linear extrapolation beyond Day 10
            slope = (curve[-1] - curve[-2]) if len(curve) > 1 else 0.01
            slope = max(1e-4, slope)
            delta_p = p_val - p_max
            elt_raw = 10.0 + (delta_p / slope)
        else:
            elt_raw = float(np.interp(p_val, curve, self.leads_))

        # Boundary checks & clipping
        is_better = bool(elt_raw < 1.0)
        is_beyond = bool(elt_raw > 12.0)
        elt_clipped = float(np.clip(elt_raw, self.min_clip, self.max_clip))

        # Headline generation per docs/06 §1.1
        if is_better:
            headline = f"Day {nominal_lead_day} forecast behaving better than a typical Day-1 forecast"
        elif is_beyond:
            headline = f"Day {nominal_lead_day} forecast behaving beyond our measured range"
        else:
            headline = f"Day {nominal_lead_day} forecast behaving like Day {elt_clipped:.1f}"

        # Direction determination
        direction = classify_bust_direction(signed_expect, deadband=direction_deadband)
        band = classify_confidence_band(p_val)

        return ELTResult(
            effective_lead_time=round(elt_clipped, 1),
            nominal_lead_day=nominal_lead_day,
            p_calibrated=round(p_val, 4),
            headline=headline,
            is_better_than_day1=is_better,
            is_beyond_range=is_beyond,
            confidence_band=band,
            direction=direction,
            signed_expect=round(float(signed_expect), 2),
        )

    def transform_series(
        self,
        p_calibrated: np.ndarray,
        nominal_leads: np.ndarray,
        region_ids: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        """Batch vector transform returning array of effective lead times."""
        if not self.is_fitted or self.national_curve_ is None:
            raise RuntimeError("EffectiveLeadTimeTransformer must be fitted before transform_series.")

        p_arr = np.asarray(p_calibrated, dtype=float)
        leads_arr = np.asarray(nominal_leads, dtype=int)
        elts = []

        for i, (p, lead) in enumerate(zip(p_arr, leads_arr)):
            reg = region_ids[i] if region_ids is not None else None
            res = self.transform_single(p, lead, region_id=reg)
            elts.append(res.effective_lead_time)

        return np.array(elts, dtype=float)


def classify_bust_direction(
    signed_expect: float,
    deadband: float = DEFAULT_DIRECTION_DEADBAND,
) -> BustDirection:
    """Classify 3-class bust direction head (Node G5, docs/03 §4.1, docs/06 §1.1).

    Sign convention:
        error_signed = forecast_value - observed_value
        - signed_expect < -deadband: Forecast was drier than observed -> WET bust (wetter than forecast)
        - signed_expect > +deadband: Forecast was wetter than observed -> DRY bust (drier than forecast)
        - |signed_expect| <= deadband: Normal error / unresolvable -> NONE

    Args:
        signed_expect: Analogue-weighted signed forecast error expectation.
        deadband: Deadband magnitude below which direction is NONE (default: 2.0 mm/day).

    Returns:
        BustDirection: WET, DRY, or NONE.
    """
    if signed_expect < -deadband:
        return BustDirection.WET
    elif signed_expect > deadband:
        return BustDirection.DRY
    return BustDirection.NONE

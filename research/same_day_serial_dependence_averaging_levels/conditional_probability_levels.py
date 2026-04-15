from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq  # type: ignore
from tqdm import tqdm  # type: ignore

RESEARCH_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RESEARCH_DIR.parents[1]
DATA_ROOT = PROJECT_ROOT.parent / "anki-revlogs-10k"
REVLOG_ROOT = DATA_ROOT / "revlogs"
VALID_RATINGS = {1, 2, 3, 4}
REVIEW = 2
MAX_DURATION_MS = 1_200_000
ORDER_CANDIDATES = (
    "review_th",
    "review_time",
    "review_time_ms",
    "review_timestamp",
    "review_ts",
    "review_datetime",
    "timestamp",
    "id",
)


def detect_order_column(data_path: Path) -> str | None:
    try:
        dataset = pq.ParquetDataset(str(data_path))
    except Exception:
        return None
    for column in ORDER_CANDIDATES:
        if column in dataset.schema.names:
            return column
    return None


ORDER_COL = detect_order_column(REVLOG_ROOT)


def make_pooled_agg() -> dict[str, float]:
    return {
        "ps_num": 0.0,
        "ps_den": 0.0,
        "pf_num": 0.0,
        "pf_den": 0.0,
        "pair_count": 0.0,
        "user_count": 0.0,
        "user_day_count": 0.0,
    }


def make_mean_agg() -> dict[str, float]:
    return {
        "ps_prob_sum": 0.0,
        "ps_prob_cnt": 0.0,
        "pf_prob_sum": 0.0,
        "pf_prob_cnt": 0.0,
        "common_gap_sum": 0.0,
        "common_gap_cnt": 0.0,
        "unit_count": 0.0,
        "pair_count": 0.0,
    }


def merge_agg(dst: dict[str, float], src: dict[str, float]) -> None:
    for key in dst:
        dst[key] += src.get(key, 0.0)


def finalize_pooled(agg: dict[str, float]) -> dict[str, float | int | None]:
    ps = agg["ps_num"] / agg["ps_den"] if agg["ps_den"] else None
    pf = agg["pf_num"] / agg["pf_den"] if agg["pf_den"] else None
    gap = ps - pf if ps is not None and pf is not None else None
    return {
        "p_success_given_prev_success": round(ps, 6) if ps is not None else None,
        "p_success_given_prev_fail": round(pf, 6) if pf is not None else None,
        "gap": round(gap, 6) if gap is not None else None,
        "pair_count": int(agg["pair_count"]),
        "user_count": int(agg["user_count"]),
        "user_day_count": int(agg["user_day_count"]),
    }


def finalize_mean(agg: dict[str, float]) -> dict[str, float | int | None]:
    ps = agg["ps_prob_sum"] / agg["ps_prob_cnt"] if agg["ps_prob_cnt"] else None
    pf = agg["pf_prob_sum"] / agg["pf_prob_cnt"] if agg["pf_prob_cnt"] else None
    gap = ps - pf if ps is not None and pf is not None else None
    common_gap = (
        agg["common_gap_sum"] / agg["common_gap_cnt"] if agg["common_gap_cnt"] else None
    )
    return {
        "p_success_given_prev_success": round(ps, 6) if ps is not None else None,
        "p_success_given_prev_fail": round(pf, 6) if pf is not None else None,
        "gap": round(gap, 6) if gap is not None else None,
        "common_support_gap": round(common_gap, 6) if common_gap is not None else None,
        "defined_prev_success_units": int(agg["ps_prob_cnt"]),
        "defined_prev_fail_units": int(agg["pf_prob_cnt"]),
        "both_defined_units": int(agg["common_gap_cnt"]),
        "unit_count": int(agg["unit_count"]),
        "pair_count": int(agg["pair_count"]),
    }


def counts_from_pairs(
    prev: np.ndarray,
    nxt: np.ndarray,
) -> tuple[float, float, float, float]:
    prev_success = prev == 1
    prev_fail = prev == 0
    return (
        float(nxt[prev_success].sum()),
        float(prev_success.sum()),
        float(nxt[prev_fail].sum()),
        float(prev_fail.sum()),
    )


def mean_agg_from_unit_counts(
    ps_num: np.ndarray,
    ps_den: np.ndarray,
    pf_num: np.ndarray,
    pf_den: np.ndarray,
    pair_count: int,
) -> dict[str, float]:
    agg = make_mean_agg()
    agg["unit_count"] = float(len(ps_den))
    agg["pair_count"] = float(pair_count)

    ps_mask = ps_den > 0
    if np.any(ps_mask):
        agg["ps_prob_sum"] = float((ps_num[ps_mask] / ps_den[ps_mask]).sum())
        agg["ps_prob_cnt"] = float(ps_mask.sum())

    pf_mask = pf_den > 0
    if np.any(pf_mask):
        agg["pf_prob_sum"] = float((pf_num[pf_mask] / pf_den[pf_mask]).sum())
        agg["pf_prob_cnt"] = float(pf_mask.sum())

    both_mask = ps_mask & pf_mask
    if np.any(both_mask):
        common_gap = (ps_num[both_mask] / ps_den[both_mask]) - (
            pf_num[both_mask] / pf_den[both_mask]
        )
        agg["common_gap_sum"] = float(common_gap.sum())
        agg["common_gap_cnt"] = float(both_mask.sum())

    return agg


def user_day_counts_from_pairs(
    prev: np.ndarray,
    nxt: np.ndarray,
    current_days: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(prev) == 0:
        zeros = np.zeros(0, dtype=float)
        return zeros, zeros, zeros, zeros

    _, inverse = np.unique(current_days, return_inverse=True)
    prev_success = (prev == 1).astype(float)
    prev_fail = (prev == 0).astype(float)
    next_success = nxt.astype(float)

    ps_den = np.bincount(inverse, weights=prev_success)
    pf_den = np.bincount(inverse, weights=prev_fail)
    ps_num = np.bincount(inverse, weights=prev_success * next_success)
    pf_num = np.bincount(inverse, weights=prev_fail * next_success)
    return ps_num, ps_den, pf_num, pf_den


def restrict_to_common_support_user_days(
    prev: np.ndarray,
    nxt: np.ndarray,
    current_days: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(prev) == 0:
        return prev, nxt, current_days

    day_ps_num, day_ps_den, day_pf_num, day_pf_den = user_day_counts_from_pairs(
        prev,
        nxt,
        current_days,
    )
    keep_days = (day_ps_den > 0) & (day_pf_den > 0)
    if not np.any(keep_days):
        empty = np.zeros(0, dtype=prev.dtype)
        return empty, empty, np.zeros(0, dtype=current_days.dtype)

    _, inverse = np.unique(current_days, return_inverse=True)
    keep_pairs = keep_days[inverse]
    return prev[keep_pairs], nxt[keep_pairs], current_days[keep_pairs]


def load_user_revlogs(user_id: int) -> pd.DataFrame:
    columns = ["card_id", "day_offset", "rating", "state", "duration", "elapsed_days"]
    if ORDER_COL is not None and ORDER_COL not in columns:
        columns.append(ORDER_COL)

    revlog_path = REVLOG_ROOT / f"user_id={user_id}" / "data.parquet"
    df = pd.read_parquet(revlog_path, columns=columns)
    df = df[df["rating"].isin(VALID_RATINGS)].copy()
    df = df[(df["duration"] > 0) & (df["duration"] < MAX_DURATION_MS)].copy()
    if df.empty:
        return df

    if ORDER_COL is not None and ORDER_COL in df.columns:
        df = df.sort_values(ORDER_COL).copy()

    df["y"] = (df["rating"] > 1).astype(np.int8)
    df["first_of_day_card"] = df.groupby(["day_offset", "card_id"]).cumcount().eq(0)
    return df


def sequence_stats_from_pairs(
    prev: np.ndarray,
    nxt: np.ndarray,
    current_days: np.ndarray,
) -> dict[str, dict[str, float]]:
    pooled = make_pooled_agg()
    equal_user_mean = make_mean_agg()
    equal_user_day_mean = make_mean_agg()
    pooled_on_common_support_user_days = make_pooled_agg()
    equal_user_mean_on_common_support_user_days = make_mean_agg()

    if len(prev) == 0:
        return {
            "pooled": pooled,
            "equal_user_mean": equal_user_mean,
            "equal_user_day_mean": equal_user_day_mean,
            "pooled_on_common_support_user_days": pooled_on_common_support_user_days,
            "equal_user_mean_on_common_support_user_days": equal_user_mean_on_common_support_user_days,
        }

    ps_num, ps_den, pf_num, pf_den = counts_from_pairs(prev, nxt)
    pooled.update(
        {
            "ps_num": ps_num,
            "ps_den": ps_den,
            "pf_num": pf_num,
            "pf_den": pf_den,
            "pair_count": float(len(prev)),
            "user_count": 1.0,
            "user_day_count": float(len(np.unique(current_days))),
        }
    )

    equal_user_mean = mean_agg_from_unit_counts(
        np.array([ps_num]),
        np.array([ps_den]),
        np.array([pf_num]),
        np.array([pf_den]),
        pair_count=len(prev),
    )

    day_ps_num, day_ps_den, day_pf_num, day_pf_den = user_day_counts_from_pairs(
        prev,
        nxt,
        current_days,
    )
    equal_user_day_mean = mean_agg_from_unit_counts(
        day_ps_num,
        day_ps_den,
        day_pf_num,
        day_pf_den,
        pair_count=len(prev),
    )

    common_prev, common_nxt, common_days = restrict_to_common_support_user_days(
        prev,
        nxt,
        current_days,
    )
    if len(common_prev) > 0:
        common_ps_num, common_ps_den, common_pf_num, common_pf_den = counts_from_pairs(
            common_prev, common_nxt
        )
        pooled_on_common_support_user_days.update(
            {
                "ps_num": common_ps_num,
                "ps_den": common_ps_den,
                "pf_num": common_pf_num,
                "pf_den": common_pf_den,
                "pair_count": float(len(common_prev)),
                "user_count": 1.0,
                "user_day_count": float(len(np.unique(common_days))),
            }
        )
        equal_user_mean_on_common_support_user_days = mean_agg_from_unit_counts(
            np.array([common_ps_num]),
            np.array([common_ps_den]),
            np.array([common_pf_num]),
            np.array([common_pf_den]),
            pair_count=len(common_prev),
        )

    return {
        "pooled": pooled,
        "equal_user_mean": equal_user_mean,
        "equal_user_day_mean": equal_user_day_mean,
        "pooled_on_common_support_user_days": pooled_on_common_support_user_days,
        "equal_user_mean_on_common_support_user_days": equal_user_mean_on_common_support_user_days,
    }


def analyze_raw_long_term(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    if df.empty:
        return {
            "pooled": make_pooled_agg(),
            "equal_user_mean": make_mean_agg(),
            "equal_user_day_mean": make_mean_agg(),
            "pooled_on_common_support_user_days": make_pooled_agg(),
            "equal_user_mean_on_common_support_user_days": make_mean_agg(),
        }

    seq_df = df[df["elapsed_days"] > 0].copy()
    y = seq_df["y"].to_numpy(dtype=np.int8)
    if len(y) < 2:
        return {
            "pooled": make_pooled_agg(),
            "equal_user_mean": make_mean_agg(),
            "equal_user_day_mean": make_mean_agg(),
            "pooled_on_common_support_user_days": make_pooled_agg(),
            "equal_user_mean_on_common_support_user_days": make_mean_agg(),
        }

    prev = y[:-1]
    nxt = y[1:]
    current_days = seq_df["day_offset"].to_numpy()[1:]
    return sequence_stats_from_pairs(prev, nxt, current_days)


def analyze_same_day_first_of_day_review(
    df: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    if df.empty:
        return {
            "pooled": make_pooled_agg(),
            "equal_user_mean": make_mean_agg(),
            "equal_user_day_mean": make_mean_agg(),
            "pooled_on_common_support_user_days": make_pooled_agg(),
            "equal_user_mean_on_common_support_user_days": make_mean_agg(),
        }

    first_df = df[df["first_of_day_card"]].copy()
    review_df = first_df[first_df["state"] == REVIEW].copy()
    y = review_df["y"].to_numpy(dtype=np.int8)
    day = review_df["day_offset"].to_numpy()
    if len(y) < 2:
        return {
            "pooled": make_pooled_agg(),
            "equal_user_mean": make_mean_agg(),
            "equal_user_day_mean": make_mean_agg(),
            "pooled_on_common_support_user_days": make_pooled_agg(),
            "equal_user_mean_on_common_support_user_days": make_mean_agg(),
        }

    same_day = day[1:] == day[:-1]
    prev = y[:-1][same_day]
    nxt = y[1:][same_day]
    current_days = day[1:][same_day]
    return sequence_stats_from_pairs(prev, nxt, current_days)


def analyze_user(user_id: int) -> dict[str, dict[str, dict[str, float]]]:
    df = load_user_revlogs(user_id)
    return {
        "raw_long_term": analyze_raw_long_term(df),
        "same_day_first_of_day_review": analyze_same_day_first_of_day_review(df),
    }


def parse_user_id(path: Path) -> int:
    return int(path.name.split("=")[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--limit-users", type=int, default=None)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=RESEARCH_DIR / "results" / "conditional_probability_levels.json",
    )
    args = parser.parse_args()

    user_paths = sorted(REVLOG_ROOT.glob("user_id=*"), key=parse_user_id)
    user_ids = [parse_user_id(path) for path in user_paths]
    if args.limit_users is not None:
        user_ids = user_ids[: args.limit_users]

    aggregate = {
        sequence: {
            "pooled": make_pooled_agg(),
            "equal_user_mean": make_mean_agg(),
            "equal_user_day_mean": make_mean_agg(),
            "pooled_on_common_support_user_days": make_pooled_agg(),
            "equal_user_mean_on_common_support_user_days": make_mean_agg(),
        }
        for sequence in ("raw_long_term", "same_day_first_of_day_review")
    }

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(analyze_user, user_id): user_id for user_id in user_ids
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Computing conditional-probability levels",
        ):
            result = future.result()
            for sequence, sequence_result in result.items():
                merge_agg(aggregate[sequence]["pooled"], sequence_result["pooled"])
                merge_agg(
                    aggregate[sequence]["equal_user_mean"],
                    sequence_result["equal_user_mean"],
                )
                merge_agg(
                    aggregate[sequence]["equal_user_day_mean"],
                    sequence_result["equal_user_day_mean"],
                )
                merge_agg(
                    aggregate[sequence]["pooled_on_common_support_user_days"],
                    sequence_result["pooled_on_common_support_user_days"],
                )
                merge_agg(
                    aggregate[sequence]["equal_user_mean_on_common_support_user_days"],
                    sequence_result["equal_user_mean_on_common_support_user_days"],
                )

    output = {
        "metadata": {
            "data_root": str(DATA_ROOT),
            "review_order_column": ORDER_COL,
            "requested_user_count": len(user_ids),
            "sequence_notes": {
                "raw_long_term": [
                    "retain ratings in {1, 2, 3, 4} only",
                    "valid duration only",
                    "adjacent pairs are formed inside the elapsed_days > 0 subsequence",
                    "when no explicit order column exists, adjacency follows stored parquet row order",
                    "pair day is assigned from the current event",
                ],
                "same_day_first_of_day_review": [
                    "retain ratings in {1, 2, 3, 4} only",
                    "valid duration only",
                    "keep the first stored-order occurrence of each card within each day",
                    "restrict to state == Review",
                    "keep same-day adjacent pairs inside the state == Review subsequence",
                    "within a day, each pair is between two different cards by construction",
                ],
            },
            "averaging_notes": [
                "pooled weights every qualifying pair equally",
                "equal_user_mean weights every user equally after computing that user's conditional probability",
                "equal_user_day_mean weights every user-day equally after computing that user-day's conditional probability",
                "equal_user_day_mean does not give every user equal total weight; users with more retained days contribute more user-day units",
                "for mean levels, the prev-success and prev-fail averages are computed over the units where each conditional is defined",
                "common_support_gap averages the unit-level gap only over units where both conditionals are defined",
                "pooled_on_common_support_user_days and equal_user_mean_on_common_support_user_days first restrict to user-days where both contexts appear, then re-aggregate pairs or users on that restricted subset",
            ],
        },
        "sequences": {
            sequence: {
                "pooled": finalize_pooled(levels["pooled"]),
                "equal_user_mean": finalize_mean(levels["equal_user_mean"]),
                "equal_user_day_mean": finalize_mean(levels["equal_user_day_mean"]),
                "pooled_on_common_support_user_days": finalize_pooled(
                    levels["pooled_on_common_support_user_days"]
                ),
                "equal_user_mean_on_common_support_user_days": finalize_mean(
                    levels["equal_user_mean_on_common_support_user_days"]
                ),
            }
            for sequence, levels in aggregate.items()
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

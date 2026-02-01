from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import os
import threading
import pandas as pd
import numpy as np
import json
from pathlib import Path
import warnings
import pyarrow.parquet as pq  # type: ignore
from tqdm import tqdm  # type: ignore
from markov_chain import FirstOrderMarkovChain

warnings.filterwarnings("ignore")

New = 0
Learning = 1
Review = 2
Relearning = 3
Filtered = 4

DATA_PATH = "../anki-revlogs-10k/revlogs"


def analyze(user_id):
    df_raw = pd.read_parquet(
        DATA_PATH,
        filters=[("user_id", "=", user_id)],
        columns=["card_id", "state", "elapsed_days", "duration", "rating"],
    )
    df_raw["review_th"] = np.arange(1, df_raw.shape[0] + 1)
    df_raw.sort_values(by=["card_id", "review_th"], inplace=True)
    df_raw["state"] = df_raw["state"].replace({New: Learning})
    df_raw["delta_t"] = df_raw["elapsed_days"].clip(lower=0)
    df_raw["real_days"] = df_raw.groupby("card_id", sort=False)["delta_t"].cumsum()
    df_raw = df_raw[(df_raw["duration"] > 0) & (df_raw["duration"] < 1200000)]

    state_rating_costs = (
        df_raw[df_raw["state"] != Filtered]
        .groupby(["state", "rating"])["duration"]
        .mean()
        .unstack(fill_value=0)
    ) / 1000

    # Ensure all ratings (1-4) exist in columns
    for rating in range(1, 5):
        if rating not in state_rating_costs.columns:
            state_rating_costs[rating] = 0

    # Ensure all states exist in index
    for state in [Learning, Review, Relearning]:
        if state not in state_rating_costs.index:
            state_rating_costs.loc[state] = 0

    group_keys = ["card_id", "real_days"]
    g = df_raw.groupby(group_keys, sort=False)

    first_state = g["state"].first()
    first_rating = g["rating"].first()
    sum_duration = g["duration"].sum()
    review_count = g.size()
    second = g["rating"].nth(1)
    if not second.empty:
        second_keys = df_raw.loc[second.index, group_keys]
        second_index = pd.MultiIndex.from_frame(second_keys)
        second = pd.Series(second.to_numpy(), index=second_index)
    next_rating = second.reindex(first_state.index).fillna(0).astype(int)
    same_day_ratings = g["rating"].agg(list).reindex(first_state.index)

    rating_counts = (
        df_raw.groupby(group_keys + ["rating"], sort=False)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=[1, 2, 3, 4], fill_value=0)
    )
    rating_counts = rating_counts.reindex(first_rating.index)
    rating_counts_values = rating_counts.to_numpy(copy=True)
    first_rating_values = first_rating.to_numpy()
    row_idx = np.arange(rating_counts_values.shape[0])
    col_idx = first_rating_values - 1
    valid_mask = (col_idx >= 0) & (col_idx < rating_counts_values.shape[1])
    rating_counts_values[row_idx[valid_mask], col_idx[valid_mask]] -= 1
    rating_counts = pd.DataFrame(
        rating_counts_values, index=rating_counts.index, columns=rating_counts.columns
    )

    df = pd.DataFrame(
        {
            "first_state": first_state.to_numpy(),
            "first_rating": first_rating.to_numpy(),
            "next_rating": next_rating.to_numpy(),
            "same_day_ratings": same_day_ratings.to_numpy(),
            "sum_duration": sum_duration.to_numpy(),
            "review_count": review_count.to_numpy(),
        },
        index=first_state.index,
    )
    df = df.join(rating_counts, how="left").reset_index()
    df["y"] = (df["first_rating"] > 1).astype(int)
    true_retention = df["y"].mean()

    model = FirstOrderMarkovChain()
    learning_step_rating_sequences = df[df["first_state"] == Learning][
        "same_day_ratings"
    ]
    learning_step_transition = model.fit(
        learning_step_rating_sequences
    ).transition_counts[:3]
    relearning_step_rating_sequences = df[
        (df["first_state"] == Review) & (df["first_rating"] == 1)
    ]["same_day_ratings"]
    relearning_step_transition = model.fit(
        relearning_step_rating_sequences
    ).transition_counts[:3]

    long_term_rating_sequences = (
        df[df["first_state"] == Review]
        .groupby(by=["card_id"])["first_rating"]
        .apply(list)
    )
    long_term_transition = model.fit(long_term_rating_sequences).transition_counts

    cost_dict = (
        df.groupby(by=["first_state", "first_rating"])["sum_duration"].mean().to_dict()
    )
    learn_costs = np.array([cost_dict.get((1, i), 0) / 1000 for i in range(1, 5)])
    review_costs = np.array([cost_dict.get((2, i), 0) / 1000 for i in range(1, 5)])
    button_usage_dict = (
        df.groupby(by=["first_state", "first_rating"])["card_id"].count().to_dict()
    )
    learn_buttons = np.array([button_usage_dict.get((1, i), 0) for i in range(1, 5)])
    review_buttons = np.array([button_usage_dict.get((2, i), 0) for i in range(2, 5)])
    first_rating_prob = learn_buttons / learn_buttons.sum()
    review_rating_prob = review_buttons / review_buttons.sum()

    rating_cols_no_again = [2, 3, 4]
    learning_counts = df[df["first_state"] == Learning][rating_cols_no_again].sum()
    learning_total = learning_counts.sum()
    learning_rating_prob = (
        (learning_counts / learning_total).reindex(rating_cols_no_again).to_numpy()
        if learning_total > 0
        else np.array([np.nan, np.nan, np.nan])
    )

    relearning_counts = df[
        (df["first_state"] == Review) & (df["first_rating"] == 1)
    ][rating_cols_no_again].sum()
    relearning_total = relearning_counts.sum()
    relearning_rating_prob = (
        (relearning_counts / relearning_total).reindex(rating_cols_no_again).to_numpy()
        if relearning_total > 0
        else np.array([np.nan, np.nan, np.nan])
    )

    df2 = df.groupby(by=["first_state", "first_rating"])[[1, 2, 3, 4]].mean().round(2)
    rating_offset_dict = sum([df2[g] * (g - 3) for g in range(1, 5)]).to_dict()
    session_len_dict = sum([df2[g] for g in range(1, 5)]).to_dict()
    first_rating_offset = np.array(
        [rating_offset_dict.get((1, i), 0) for i in range(1, 5)]
    )
    first_session_len = np.array([session_len_dict.get((1, i), 0) for i in range(1, 5)])
    forget_rating_offset = rating_offset_dict.get((2, 1), 0)
    forget_session_len = session_len_dict.get((2, 1), 0)

    def calculate_recall_rate(group):
        total = len(group)
        remembered = len(group[group["next_rating"] > 1])
        recall_rate = remembered / total if total > 0 else 1
        return recall_rate

    df3 = (
        df[df["next_rating"] > 0]
        .groupby(by=["first_state", "first_rating"])
        .apply(calculate_recall_rate)
        .to_dict()
    )
    short_term_recall = np.array(
        [df3.get((1, i), 0) for i in range(1, 4)] + [df3.get((2, 1), 0)]
    )
    result = {
        "user": user_id,
        "review_cnt": df["review_count"].sum().item(),
        "card_cnt": df["card_id"].nunique(),
        "first_rating_prob": first_rating_prob.round(4).tolist(),
        "review_rating_prob": review_rating_prob.round(4).tolist(),
        "learning_rating_prob": learning_rating_prob.round(4).tolist(),
        "relearning_rating_prob": relearning_rating_prob.round(4).tolist(),
        "learn_costs": learn_costs.round(2).tolist(),
        "review_costs": review_costs.round(2).tolist(),
        "first_rating_offset": first_rating_offset.round(2).tolist(),
        "first_session_len": first_session_len.round(2).tolist(),
        "forget_rating_offset": round(forget_rating_offset, 2),
        "forget_session_len": round(forget_session_len, 2),
        "short_term_recall": short_term_recall.round(4).tolist(),
        "learning_step_transition": learning_step_transition.astype(int).tolist(),
        "relearning_step_transition": relearning_step_transition.astype(int).tolist(),
        "long_term_transition": long_term_transition.astype(int).tolist(),
        "state_rating_costs": state_rating_costs.values.round(2).tolist(),
        "true_retention": round(true_retention, 3),
    }
    return result


def analyze_batch(user_ids, progress_q=None):
    results = []
    errors = []
    for user_id in user_ids:
        try:
            results.append(analyze(user_id))
        except Exception as e:
            errors.append((user_id, str(e)))
        finally:
            if progress_q is not None:
                progress_q.put(1)
    return results, errors


def sort_jsonl(file):
    data = list(map(lambda x: json.loads(x), open(file).readlines()))
    data.sort(key=lambda x: x["user"])
    with file.open("w", encoding="utf-8") as jsonl_file:
        for json_data in data:
            jsonl_file.write(json.dumps(json_data, ensure_ascii=False) + "\n")
    return data


if __name__ == "__main__":
    result_file = Path(f"button_usage.jsonl")
    if result_file.exists():
        data = sort_jsonl(result_file)
        processed_user = set(map(lambda x: x["user"], data))
    else:
        processed_user = set()

    dataset = pq.ParquetDataset(DATA_PATH)
    unprocessed_users = []
    for user_id in dataset.partitioning.dictionaries[0]:
        if user_id.as_py() in processed_user:
            continue
        unprocessed_users.append(user_id.as_py())

    unprocessed_users.sort()

    max_workers = int(os.getenv("MAX_WORKERS", os.cpu_count() or 4))
    chunksize = int(os.getenv("CHUNKSIZE", 32))
    batches = [
        unprocessed_users[i : i + chunksize]
        for i in range(0, len(unprocessed_users), chunksize)
    ]
    total_users = len(unprocessed_users)

    with tqdm(total=total_users, position=0, leave=True) as pbar:
        with mp.Manager() as manager:
            progress_q = manager.Queue()

            def monitor_progress():
                processed = 0
                while processed < total_users:
                    try:
                        inc = progress_q.get()
                    except Exception:
                        break
                    if inc:
                        processed += inc
                        pbar.update(inc)

            monitor = threading.Thread(target=monitor_progress, daemon=True)
            monitor.start()

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(analyze_batch, batch, progress_q)
                    for batch in batches
                ]
                with open(result_file, "a") as f:
                    for future in as_completed(futures):
                        try:
                            results, errors = future.result()
                            for result in results:
                                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                            for user_id, err in errors:
                                tqdm.write(f"{user_id}: {err}")
                        except Exception as e:
                            tqdm.write(str(e))

            monitor.join()

    sort_jsonl(result_file)

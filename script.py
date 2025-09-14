from concurrent.futures import ProcessPoolExecutor, as_completed
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
    df = pd.read_parquet(DATA_PATH, filters=[("user_id", "=", user_id)])
    df["review_th"] = range(1, df.shape[0] + 1)
    df.sort_values(by=["card_id", "review_th"], inplace=True)
    df["state"] = df["state"].map(lambda x: x if x != New else Learning)
    df["delta_t"] = df["elapsed_days"].map(lambda x: max(0, x))
    df["real_days"] = df.groupby("card_id")["delta_t"].cumsum()
    df["i"] = df.groupby("card_id").cumcount() + 1
    df = df[(df["duration"] > 0) & (df["duration"] < 1200000)]
    df["y"] = df["rating"].map(lambda x: 1 if x > 1 else 0)
    true_retention = df["y"].mean()

    def rating_counts(x):
        tmp = x.value_counts().to_dict()
        first = x.iloc[0]
        tmp[first] -= 1
        for i in range(1, 5):
            if i not in tmp:
                tmp[i] = 0
        return tmp

    def next_rating(x):
        if x.shape[0] > 1:
            return x.iloc[1]
        return 0

    state_rating_costs = (
        df[df["state"] != Filtered]
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

    df = (
        df.groupby(by=["card_id", "real_days"])
        .agg(
            {
                "state": "first",
                "rating": ["first", rating_counts, next_rating, list],
                "duration": "sum",
                "i": "size",
            }
        )
        .reset_index()
    )
    df.columns = [
        "card_id",
        "real_days",
        "first_state",
        "first_rating",
        "rating_counts",
        "next_rating",
        "same_day_ratings",
        "sum_duration",
        "review_count",
    ]
    rating_counts_df = df["rating_counts"].apply(pd.Series).fillna(0).astype(int)
    df = pd.concat([df.drop("rating_counts", axis=1), rating_counts_df], axis=1)

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
        "size": df.shape[0],
        "first_rating_prob": first_rating_prob.round(4).tolist(),
        "review_rating_prob": review_rating_prob.round(4).tolist(),
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

    with tqdm(total=len(unprocessed_users), position=0, leave=True) as pbar:
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(analyze, user_id) for user_id in unprocessed_users
            ]
            for future in as_completed(futures):
                try:
                    result = future.result()
                    with open(result_file, "a") as f:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    pbar.update(1)
                except Exception as e:
                    tqdm.write(str(e))

    sort_jsonl(result_file)

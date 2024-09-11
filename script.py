from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import numpy as np
import json
from pathlib import Path
import warnings

from tqdm import tqdm

warnings.filterwarnings("ignore")

New = 0
Learning = 1
Review = 2
Relearning = 3


def analyze(dataset):
    df = pd.read_csv(dataset)
    df["state"] = df["state"].map(lambda x: x if x != New else Learning)
    df["delta_t"] = df["delta_t"].map(lambda x: max(0, x))
    df["real_days"] = df.groupby("card_id")["delta_t"].cumsum().reset_index(drop=True)
    df["i"] = df.groupby("card_id").cumcount() + 1

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

    df = (
        df[(df["duration"] > 0) & (df["duration"] < 1200000)]
        .groupby(by=["card_id", "real_days"])
        .agg(
            {
                "state": "first",
                "rating": ["first", rating_counts, next_rating],
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
        "sum_duration",
        "review_count",
    ]
    rating_counts_df = df["rating_counts"].apply(pd.Series).fillna(0).astype(int)
    df = pd.concat([df.drop("rating_counts", axis=1), rating_counts_df], axis=1)

    cost_dict = (
        df.groupby(by=["first_state", "first_rating"])["sum_duration"]
        .median()
        .to_dict()
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
        "user": int(dataset.stem),
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
    }
    return result


if __name__ == "__main__":
    result_file = Path(f"button_usage.jsonl")
    if result_file.exists():
        data = list(map(lambda x: json.loads(x), open(result_file).readlines()))
        data.sort(key=lambda x: x["user"])
        with result_file.open("w", encoding="utf-8") as jsonl_file:
            for json_data in data:
                jsonl_file.write(json.dumps(json_data, ensure_ascii=False) + "\n")
        processed_user = set(map(lambda x: x["user"], data))
    else:
        processed_user = set()
    for dir in (1, 2):
        datasets = sorted(
            Path(f"../FSRS-Anki-20k/dataset/{dir}").glob("*.csv"),
            key=lambda x: int(x.stem.split(".")[0]),
        )
        with tqdm(total=len(datasets), position=0, leave=True) as pbar:
            with ProcessPoolExecutor() as executor:
                futures = [
                    executor.submit(analyze, dataset)
                    for dataset in datasets
                    if int(dataset.stem) not in processed_user
                ]
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        with open(result_file, "a") as f:
                            f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        pbar.update(1)
                    except Exception as e:
                        tqdm.write(str(e))

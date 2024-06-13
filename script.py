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

    new_card_revlog = df[(df["state"] == New) & (df["rating"].isin([1, 2, 3, 4]))]
    first_rating_prob = np.zeros(4)
    first_rating_prob[new_card_revlog["rating"].value_counts().index - 1] = (
        new_card_revlog["rating"].value_counts() / new_card_revlog["rating"].count()
    )
    recall_card_revlog = df[(df["state"] == Review) & (df["rating"].isin([2, 3, 4]))]
    review_rating_prob = np.zeros(3)
    review_rating_prob[recall_card_revlog["rating"].value_counts().index - 2] = (
        recall_card_revlog["rating"].value_counts()
        / recall_card_revlog["rating"].count()
    )

    df["state"] = df["state"].map(lambda x: x if x != New else Learning)

    recall_costs = np.zeros(3)
    recall_card_revlog = recall_card_revlog[
        (recall_card_revlog["duration"] > 0) & (df["duration"] < 1200000)
    ]
    recall_costs_agg = recall_card_revlog.groupby(by="rating")["duration"].median()
    recall_costs[recall_costs_agg.index - 2] = recall_costs_agg / 1000

    state_sequence = np.array(
        df[(df["duration"] > 0) & (df["duration"] < 1200000)]["state"]
    )
    duration_sequence = np.array(
        df[(df["duration"] > 0) & (df["duration"] < 1200000)]["duration"]
    )
    learn_cost = round(
        df[
            (df["state"] == Learning)
            & (df["duration"] > 0)
            & (df["duration"] < 1200000)
        ]
        .groupby("card_id")
        .agg({"duration": "sum"})["duration"]
        .median()
        / 1000,
        1,
    )

    state_durations = dict()
    last_state = state_sequence[0]
    state_durations[last_state] = [duration_sequence[0]]
    for i, state in enumerate(state_sequence[1:], start=1):
        if state not in state_durations:
            state_durations[state] = []
        if state == Review:
            state_durations[state].append(duration_sequence[i])
        else:
            if state == last_state:
                state_durations[state][-1] += duration_sequence[i]
            else:
                state_durations[state].append(duration_sequence[i])
        last_state = state

    recall_cost = round(np.median(state_durations[Review]) / 1000, 1)
    forget_cost = round(np.median(state_durations[Relearning]) / 1000 + recall_cost, 1)
    result = {
        "user_id": int(dataset.stem),
        "size": df.shape[0],
        "first_rating_prob": first_rating_prob.round(4).tolist(),
        "review_rating_prob": review_rating_prob.round(4).tolist(),
        "recall_costs": recall_costs.round(4).tolist(),
        "forget_cost": forget_cost,
        "learn_cost": learn_cost,
    }
    return result


if __name__ == "__main__":
    result_file = Path(f"button_usage.jsonl")
    if result_file.exists():
        data = list(map(lambda x: json.loads(x), open(result_file).readlines()))
        data.sort(key=lambda x: x["user_id"])
        with result_file.open("w", encoding="utf-8") as jsonl_file:
            for json_data in data:
                jsonl_file.write(json.dumps(json_data, ensure_ascii=False) + "\n")
        processed_user = set(map(lambda x: x["user_id"], data))
    for dir in (1, 2):
        datasets = sorted(
            Path(f"../FSRS-Anki-20k/dataset/{dir}").glob("*.csv"),
            key=lambda x: int(x.stem.split(".")[0]),
        )
        with tqdm(total=len(datasets), position=0, leave=True) as pbar:
            with ProcessPoolExecutor(max_workers=8) as executor:
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

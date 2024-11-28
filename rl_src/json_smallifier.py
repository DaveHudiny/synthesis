import json
import os

import numpy as np


def summarize_each_episode_stats_to_variance(data):
    # remove each_episode_successes
    try:
        data.pop("each_episode_successes")
    except:
        pass
    # convert each_episode_returns to variance
    # data["each_episode_returns_variance"] = np.var(data["each_episode_returns"], axis=-1)
    data["each_episode_returns"] = None
    return data


def go_through_all_episodez_and_smallify(path="./"):
    for file in os.listdir(path):
        print(file)
        if "grid-large-30-5_Stochastic_PPO_training.json" in file:
            continue
        if file.endswith(".json"):
            try:
                file = os.path.join(path, file)
                with open(file, "r") as f:
                    data = json.load(f)
                print(data.keys())
                data = summarize_each_episode_stats_to_variance(data)
                print(data.keys())
                with open(file, "w") as f:
                    json.dump(data, f)
            except Exception as e:
                print(f"Error in {file}: {e}")


if __name__ == "__main__":
    go_through_all_episodez_and_smallify("./experiments_vectorized_Valuations")

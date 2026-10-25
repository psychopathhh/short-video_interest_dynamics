import numpy as np
import polars as pl

from common import *

BLOCK, MIN_POSITIVES = 50, 3
LAGS = (1, 2, 4, 8, 16, 32, 64, 128)


def block_profiles(history, emb):
    blocks = (
        history.with_columns(block=pl.int_range(pl.len()).over("uid") // BLOCK)
        .filter(pl.col("label"))
        .with_columns(n=pl.len().over("uid", "block"))
        .filter(pl.col("n") >= MIN_POSITIVES)
    )
    keys = blocks.select("uid", "block").unique(maintain_order=True).with_row_index("row")
    blocks = blocks.join(keys, on=["uid", "block"])
    ones = np.ones(len(blocks), np.float32)
    vectors = interest_matrix(blocks["row"].to_numpy(), blocks["idx"].to_numpy(), ones, len(keys), emb)
    return keys, normalize(vectors)


def main():
    download()
    items = build_items()
    weeks = scan_weeks(range(N_TRAIN_WEEKS))
    users = weeks.select("user_id").unique().sort("user_id").with_row_index("uid").collect()
    history = add_signals(weeks, items, users).collect(engine="streaming")
    emb = center(load_embeddings(items["item_id"].to_numpy()), history)

    profiles, vectors = block_profiles(history, emb)
    rng = np.random.default_rng(0)

    rows = []
    for lag in LAGS:
        later = profiles.with_columns(pl.col("block") - lag).rename({"row": "row_later"})
        pairs = profiles.join(later, on=["uid", "block"])
        a, b = pairs["row"].to_numpy(), pairs["row_later"].to_numpy()
        own = (vectors[a] * vectors[b]).sum(1).mean()
        # для сравнения берём блок случайного другого пользователя
        other = (vectors[a] * vectors[rng.permutation(b)]).sum(1).mean()
        rows.append(dict(impressions=lag * BLOCK, pairs=len(a), own=own, random_user=other, taste=own - other))

    result = pl.DataFrame(rows).with_columns(retained=pl.col("taste") / pl.col("taste").first())
    print(result)
    result.write_csv("results/drift.csv")


if __name__ == "__main__":
    main()

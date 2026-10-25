import os
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp
from huggingface_hub import hf_hub_download

REPO, SUBSAMPLE = "deepvk/VK-LSVD", "up0.001_ip0.001"
ROOT = Path(os.environ.get("VK_LSVD_ROOT", "VK-LSVD"))
N_TRAIN_WEEKS, VAL_WEEK, DIM = 25, 25, 32
# скипов больше половины показов, с большим весом они перебивают позитивы
SKIP_WEIGHT = 0.2


def download():
    files = [f"subsamples/{SUBSAMPLE}/train/week_{w:02}.parquet" for w in range(N_TRAIN_WEEKS)]
    files += [f"subsamples/{SUBSAMPLE}/validation/week_{VAL_WEEK}.parquet", "metadata/items_metadata.parquet",
              "metadata/item_embeddings.npz"]
    for file in files:
        if not (ROOT / file).exists():
            hf_hub_download(REPO, file, repo_type="dataset", local_dir=ROOT)


def week_file(week):
    split = "validation" if week == VAL_WEEK else "train"
    return ROOT / f"subsamples/{SUBSAMPLE}/{split}/week_{week:02}.parquet"


def scan_weeks(weeks):
    # timestamp в данных нет, но строки отсортированы по времени, поэтому используем номер строки
    return pl.concat([pl.scan_parquet(week_file(w)) for w in weeks]).with_row_index("t")


def build_items():
    ids = scan_weeks(range(VAL_WEEK + 1)).select("item_id").unique()
    return (
        pl.scan_parquet(ROOT / "metadata/items_metadata.parquet")
        .join(ids, on="item_id", how="semi")
        .select("item_id", "duration")
        .sort("item_id")
        .with_row_index("idx")
        .collect(engine="streaming")
    )


def load_embeddings(item_ids):
    cache = ROOT / f"{SUBSAMPLE}_embeddings.npy"
    if not cache.exists():
        with np.load(ROOT / "metadata/item_embeddings.npz") as npz:
            ids, emb = npz["item_id"], npz["embedding"]
        order = np.argsort(ids)
        pos = order[np.searchsorted(ids, item_ids, sorter=order)]
        assert np.array_equal(ids[pos], item_ids)
        np.save(cache, emb[pos])
    # компоненты отсортированы по важности, можно брать первые DIM
    return np.load(cache)[:, :DIM].astype(np.float32)


def add_signals(lf, items, users):
    ratio = pl.col("timespent") / pl.col("duration").clip(1)
    return (
        lf.join(users.lazy(), on="user_id")
        .join(items.lazy(), on="item_id")
        .with_columns(label=pl.col("like") | pl.col("share") | pl.col("bookmark") | (ratio >= 0.9),
                      skip=pl.col("dislike") | (ratio < 0.2))
        .with_columns(w=pl.col("label").cast(pl.Float32) - SKIP_WEIGHT * pl.col("skip").cast(pl.Float32))
        .sort("uid", "t")
        .select("uid", "idx", "label", "w")
    )


def center(emb, history):
    # без центрирования все профили смотрят в сторону среднего эмбеддинга (норма ~0.48),
    # и персональный скор получается хуже случайного
    return emb - np.bincount(history["idx"].to_numpy(), minlength=len(emb)) @ emb / len(history)


def item_logit(history, n_items, strength=20):
    # сглаживание: у редко показанных видео доля досмотров тянется к средней
    stats = history.group_by("idx").agg(pos=pl.col("label").sum(), n=pl.len())
    prior = history["label"].mean()
    rate = np.full(n_items, prior, np.float32)
    rate[stats["idx"].to_numpy()] = ((stats["pos"] + strength * prior) / (stats["n"] + strength)).to_numpy()
    return np.log(rate / (1 - rate))


def interest_matrix(rows, cols, values, n_rows, emb):
    return np.asarray(sp.csr_matrix((values, (rows, cols)), shape=(n_rows, len(emb))) @ emb, dtype=np.float32)


def normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)

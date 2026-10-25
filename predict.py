import numpy as np
import polars as pl

from common import *

CHUNK, TOP_K = 20, 10
LAMBDAS = (0.9, 0.97, 0.99, 0.997, 0.999, 0.9997)
ALPHAS = np.linspace(0, 1, 11)


def to_chunks(events, n_users):
    events = (
        events.with_columns(pos=pl.int_range(pl.len(), dtype=pl.UInt32).over("uid"), n=pl.len().over("uid"))
        .filter(pl.col("pos") < pl.col("n") // CHUNK * CHUNK)
    )
    n_chunks = events["n"].max() // CHUNK
    rows, cols = events["uid"].to_numpy(), events["pos"].to_numpy()

    shape = (n_users, n_chunks, CHUNK)
    idx = np.zeros(shape, np.int64)
    labels = np.zeros(shape, np.float32)
    w = np.zeros(shape, np.float32)
    idx[rows, cols // CHUNK, cols % CHUNK] = events["idx"]
    labels[rows, cols // CHUNK, cols % CHUNK] = events["label"]
    w[rows, cols // CHUNK, cols % CHUNK] = events["w"]
    return idx, labels, w


def simulate(emb, long, short, lam, idx, w):
    long, short = long.copy(), short.copy()
    s_long, s_short = np.zeros(idx.shape, np.float32), np.zeros(idx.shape, np.float32)
    # EMA s = lam * s + w * e сразу для всего чанка, у последнего события вес lam^0
    decay = lam ** np.arange(CHUNK - 1, -1, -1, dtype=np.float32)

    for c in range(idx.shape[1]):
        e = emb[idx[:, c]]
        s_long[:, c] = np.einsum("ukd,ud->uk", e, normalize(long))
        s_short[:, c] = np.einsum("ukd,ud->uk", e, normalize(short))
        # обновляем профили только после того, как чанк отранжирован
        long += np.einsum("uk,ukd->ud", w[:, c], e)
        short = lam**CHUNK * short + np.einsum("uk,ukd->ud", w[:, c] * decay, e)
    return s_long, s_short


def ndcg_per_user(scores, labels):
    discounts = 1 / np.log2(np.arange(2, TOP_K + 2))
    top = np.argsort(-scores, axis=-1)[..., :TOP_K]
    dcg = (np.take_along_axis(labels, top, axis=-1) * discounts).sum(-1)
    idcg = (-np.sort(-labels, axis=-1)[..., :TOP_K] * discounts).sum(-1)
    ndcg = np.where(idcg > 0, dcg / np.maximum(idcg, 1e-9), np.nan)
    return np.nanmean(ndcg, axis=-1)


def run_period(items, emb_raw, eval_week):
    history_lf, eval_lf = scan_weeks(range(eval_week)), scan_weeks([eval_week])
    users = eval_lf.select("user_id").unique().sort("user_id").with_row_index("uid").collect()
    history = add_signals(history_lf, items, users).collect(engine="streaming")
    emb, logit = center(emb_raw, history), item_logit(history, len(items))

    history = (
        history.with_columns(k=pl.int_range(pl.len(), dtype=pl.UInt32).reverse().over("uid"))
        .filter(pl.col("w") != 0)
    )
    uid, idx_h, w_h, k = (history[c].to_numpy() for c in ("uid", "idx", "w", "k"))
    long0 = interest_matrix(uid, idx_h, w_h, len(users), emb)
    idx, labels, w = to_chunks(add_signals(eval_lf, items, users).collect(), len(users))

    scores = {"random": np.random.default_rng(0).random(idx.shape), "video quality": logit[idx]}
    for lam in LAMBDAS:
        # дальше вес события меньше 0.001, такие события можно не учитывать
        recent = k < np.log(1e-3) / np.log(lam)
        short0 = interest_matrix(uid[recent], idx_h[recent], w_h[recent] * lam ** k[recent], len(users), emb)
        scores["old tastes"], scores[f"short {lam}"] = simulate(emb, long0, short0, lam, idx, w)
    return {"labels": labels, **scores}


def evaluate(period, name, score):
    per_user = ndcg_per_user(score, period["labels"])
    return dict(model=name, ndcg=float(np.nanmean(per_user)), per_user=per_user)


def bootstrap(a, b, n=2000):
    diff = (a - b)[~np.isnan(a - b)]
    means = np.random.default_rng(0).choice(diff, (n, len(diff))).mean(1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def candidates(period, lam, alpha):
    personal = alpha * period["old tastes"] + (1 - alpha) * period[f"short {lam}"]
    return {
        "random": period["random"],
        "old tastes": period["old tastes"],
        f"recent views (lambda={lam})": period[f"short {lam}"],
        "old + recent": personal,
        "video quality": period["video quality"],
        "video quality + old tastes": period["video quality"] + period["old tastes"],
        "video quality + old + recent": period["video quality"] + personal,
    }


def main():
    download()
    items = build_items()
    emb_raw = load_embeddings(items["item_id"].to_numpy())

    tune = run_period(items, emb_raw, VAL_WEEK - 1)
    best, memory = -1, []
    for l in LAMBDAS:
        memory.append(dict(lam=l, memory_impressions=round(1 / (1 - l)),
                           recent_only_ndcg=evaluate(tune, "", tune[f"short {l}"])["ndcg"]))
        for a in ALPHAS:
            score = tune["video quality"] + a * tune["old tastes"] + (1 - a) * tune[f"short {l}"]
            ndcg = evaluate(tune, "", score)["ndcg"]
            if ndcg > best:
                best, lam, alpha = ndcg, l, a
    memory = pl.DataFrame(memory)

    test = run_period(items, emb_raw, VAL_WEEK)
    results = [evaluate(test, name, score) for name, score in candidates(test, lam, alpha).items()]
    by_name = {r["model"]: r.pop("per_user") for r in results}
    table = pl.DataFrame(results)

    gains = []
    for new, base in [("old + recent", "old tastes"), ("video quality + old tastes", "video quality"),
                      ("video quality + old + recent", "video quality + old tastes")]:
        low, high = bootstrap(by_name[new], by_name[base])
        gains.append(dict(change=f"{base} -> {new}", gain=float(np.nanmean(by_name[new] - by_name[base])),
                          ci_low=low, ci_high=high))

    pl.Config.set_tbl_rows(-1)
    pl.Config.set_fmt_str_lengths(60)
    print(f"tuned on week {VAL_WEEK - 1}: lambda={lam}, alpha={alpha:.1f}")
    print(memory, table, pl.DataFrame(gains), sep="\n")
    memory.write_csv("results/memory.csv")
    table.write_csv("results/predict.csv")
    pl.DataFrame(gains).write_csv("results/gains.csv")


if __name__ == "__main__":
    main()

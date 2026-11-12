"""Общие функции для дизайна и анализа A/B-теста алгоритма ленты.

Основная метрика эксперимента — глобальный CTR группы:
    CTR = sum(likes) / sum(views)
Она проверяется t-тестом на линеаризованных лайках и подтверждается
пуассоновским бутстрапом. Одно и то же определение используется
в 01_experiment_design и 02_ab_test_analysis.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.05
POWER = 0.80
PRIMARY_METRIC = "global_ctr"
GUARDRAIL_METRICS = ["views_per_user", "likes_per_user"]

CONTROL_GROUP = 1
TEST_GROUP = 2
GROUPS = f"{CONTROL_GROUP}, {TEST_GROUP}"
PRE_PERIOD = {"start": "2026-08-21", "end": "2026-08-27"}
EXP_PERIOD = {"start": "2026-08-28", "end": "2026-09-03"}

SQL_DIR = Path(__file__).resolve().parent.parent / "sql"


def load_env(path=None):
    path = Path(path) if path else Path(__file__).resolve().parent.parent / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip()


def get_connection(database=None):
    load_env()
    return {
        "host": os.getenv("CH_HOST", ""),
        "user": os.getenv("CH_USER", ""),
        "password": os.getenv("CH_PASSWORD", ""),
        "database": database or os.getenv("CH_DATABASE", ""),
    }


def load_sql(name, connection=None, **params):
    text = (SQL_DIR / f"{name}.sql").read_text(encoding="utf-8")
    db = (connection or get_connection())["database"]
    return text.format(db=db, **params)


def fetch(name, connection=None, **params):
    import pandahouse

    connection = connection or get_connection()
    return pandahouse.read_clickhouse(load_sql(name, connection, **params), connection=connection)


def global_ctr(likes, views):
    return np.sum(likes) / np.sum(views)


def linearize(likes, views, ctr_control):
    return np.asarray(likes) - ctr_control * np.asarray(views)


def split_groups(df, group_a, group_b, group_col="exp_group"):
    a = df[df[group_col] == group_a]
    b = df[df[group_col] == group_b]
    return a, b


def linearized_ttest(likes_a, views_a, likes_b, views_b):
    ctr_a = global_ctr(likes_a, views_a)
    lin_a = linearize(likes_a, views_a, ctr_a)
    lin_b = linearize(likes_b, views_b, ctr_a)
    return stats.ttest_ind(lin_a, lin_b, equal_var=False).pvalue


def compare_ctr(df, group_a, group_b, group_col="exp_group"):
    a, b = split_groups(df, group_a, group_b, group_col)
    ctr_a = global_ctr(a.likes, a.views)
    ctr_b = global_ctr(b.likes, b.views)
    user_ctr_a = a.likes / a.views
    user_ctr_b = b.likes / b.views
    return {
        "n_a": len(a),
        "n_b": len(b),
        "ctr_a": ctr_a,
        "ctr_b": ctr_b,
        "abs_diff": ctr_b - ctr_a,
        "rel_diff": ctr_b / ctr_a - 1,
        "p_linearized": linearized_ttest(a.likes, a.views, b.likes, b.views),
        "p_ttest_user_ctr": stats.ttest_ind(user_ctr_a, user_ctr_b, equal_var=False).pvalue,
        "p_mannwhitney_user_ctr": stats.mannwhitneyu(user_ctr_a, user_ctr_b).pvalue,
    }


def poisson_bootstrap(likes_a, views_a, likes_b, views_b, n_bootstrap=2000, seed=None):
    rng = np.random.default_rng(seed)
    likes_a, views_a = np.asarray(likes_a), np.asarray(views_a)
    likes_b, views_b = np.asarray(likes_b), np.asarray(views_b)
    w_a = rng.poisson(1, (n_bootstrap, len(likes_a)))
    w_b = rng.poisson(1, (n_bootstrap, len(likes_b)))
    ctr_a = (w_a * likes_a).sum(axis=1) / (w_a * views_a).sum(axis=1)
    ctr_b = (w_b * likes_b).sum(axis=1) / (w_b * views_b).sum(axis=1)
    return ctr_a, ctr_b


def bootstrap_ci(diff, alpha=ALPHA):
    lo, hi = np.quantile(diff, [alpha / 2, 1 - alpha / 2])
    p = 2 * min(np.mean(diff <= 0), np.mean(diff >= 0))
    return lo, hi, min(p, 1.0)


def bootstrap_ctr_diff(df, group_a, group_b, group_col="exp_group", n_bootstrap=2000, seed=None, alpha=ALPHA):
    a, b = split_groups(df, group_a, group_b, group_col)
    ctr_a, ctr_b = poisson_bootstrap(a.likes, a.views, b.likes, b.views, n_bootstrap, seed)
    diff = ctr_b - ctr_a
    lo, hi, p = bootstrap_ci(diff, alpha)
    return {"ctr_a": ctr_a, "ctr_b": ctr_b, "diff": diff, "ci_low": lo, "ci_high": hi, "p_value": p}


def bucket_test(df_bucket, group_a, group_b, metric="bucket_ctr", group_col="exp_group"):
    a, b = split_groups(df_bucket, group_a, group_b, group_col)
    return {
        "mean_a": a[metric].mean(),
        "mean_b": b[metric].mean(),
        "p_ttest": stats.ttest_ind(a[metric], b[metric], equal_var=False).pvalue,
        "p_mannwhitney": stats.mannwhitneyu(a[metric], b[metric]).pvalue,
    }


def srm_test(counts, expected_share=None, alpha=0.001):
    counts = np.asarray(counts, dtype=float)
    if expected_share is None:
        expected_share = np.full(len(counts), 1 / len(counts))
    expected = counts.sum() * np.asarray(expected_share, dtype=float)
    chi2, p = stats.chisquare(counts, expected)
    return {"chi2": chi2, "p_value": p, "srm": p < alpha}


def aa_simulation(df, group_a, group_b, group_col="exp_group", n_iter=10000, sample_size=500, metric="linearized", seed=None):
    rng = np.random.default_rng(seed)
    a, b = split_groups(df, group_a, group_b, group_col)
    la, va = a.likes.to_numpy(), a.views.to_numpy()
    lb, vb = b.likes.to_numpy(), b.views.to_numpy()
    p_values = np.empty(n_iter)
    for i in range(n_iter):
        ia = rng.choice(len(la), sample_size, replace=False)
        ib = rng.choice(len(lb), sample_size, replace=False)
        if metric == "linearized":
            p_values[i] = linearized_ttest(la[ia], va[ia], lb[ib], vb[ib])
        else:
            p_values[i] = stats.ttest_ind(la[ia] / va[ia], lb[ib] / vb[ib], equal_var=False).pvalue
    return p_values


def false_positive_rate(p_values, alpha=ALPHA):
    return np.mean(np.asarray(p_values) < alpha)


def uplift_ctr(rel_uplift):
    def effect(views, likes, rng):
        free = views - likes
        q = np.divide(rel_uplift * likes, free, out=np.zeros(len(views)), where=free > 0)
        return views, likes + rng.binomial(free, np.clip(q, 0, 1))
    return effect


def add_views_heavy_users(threshold=30, share=0.9):
    def effect(views, likes, rng):
        n = len(views)
        extra = (1 + rng.binomial(1, 0.5, n)) * rng.binomial(1, share, n) * (views >= threshold)
        ctr = np.divide(likes, views, out=np.zeros(n), where=views > 0)
        return views + extra, likes + rng.binomial(extra, ctr)
    return effect


def simulate_power(df_users, n, effect_fn, n_iter=1000, metric="ctr", min_views=1, alpha=ALPHA, seed=None):
    rng = np.random.default_rng(seed)
    views = df_users["views"].to_numpy().astype(int)
    likes = df_users["likes"].to_numpy().astype(int)
    p_values = np.empty(n_iter)
    for i in range(n_iter):
        ia = rng.integers(0, len(views), n)
        ib = rng.integers(0, len(views), n)
        views_a, likes_a = views[ia], likes[ia]
        views_b, likes_b = effect_fn(views[ib], likes[ib], rng)
        mask_a = views_a >= min_views
        mask_b = views_b >= min_views
        if metric == "ctr":
            p_values[i] = linearized_ttest(likes_a[mask_a], views_a[mask_a], likes_b[mask_b], views_b[mask_b])
        else:
            p_values[i] = stats.ttest_ind(likes_a[mask_a], likes_b[mask_b], equal_var=False).pvalue
    return np.mean(p_values < alpha)


def power_curve(df_users, n, uplifts, n_iter=500, alpha=ALPHA, seed=None):
    rows = []
    for u in uplifts:
        power = simulate_power(df_users, n, uplift_ctr(u), n_iter=n_iter, alpha=alpha, seed=seed)
        rows.append({"rel_uplift": u, "power": power})
    return pd.DataFrame(rows)


def mde_ctr(likes, views, n_per_group, alpha=ALPHA, power=POWER):
    likes, views = np.asarray(likes), np.asarray(views)
    ctr = global_ctr(likes, views)
    sd = linearize(likes, views, ctr).std(ddof=1)
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    abs_mde = z * sd * np.sqrt(2 / n_per_group) / views.mean()
    return {"ctr": ctr, "abs_mde": abs_mde, "rel_mde": abs_mde / ctr}


def sample_size_ctr(likes, views, rel_mde, alpha=ALPHA, power=POWER):
    likes, views = np.asarray(likes), np.asarray(views)
    ctr = global_ctr(likes, views)
    sd = linearize(likes, views, ctr).std(ddof=1)
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    delta_lin = rel_mde * ctr * views.mean()
    return int(np.ceil(2 * (z * sd / delta_lin) ** 2))


def add_segment(df, col, q=None, bins=None, labels=None, name=None):
    df = df.copy()
    name = name or f"{col}_segment"
    if bins is not None:
        df[name] = pd.cut(df[col], bins=bins, labels=labels)
    else:
        df[name] = pd.qcut(df[col], q=q or 4, labels=labels, duplicates="drop")
    return df


def segment_effects(df, segment_col, group_a, group_b, group_col="exp_group", n_bootstrap=1000, seed=None, alpha=ALPHA):
    rows = []
    for segment, part in df.groupby(segment_col, observed=True):
        a, b = split_groups(part, group_a, group_b, group_col)
        if len(a) < 2 or len(b) < 2:
            continue
        boot = bootstrap_ctr_diff(part, group_a, group_b, group_col, n_bootstrap, seed, alpha)
        rows.append({
            segment_col: segment,
            "n_a": len(a),
            "n_b": len(b),
            "ctr_a": global_ctr(a.likes, a.views),
            "ctr_b": global_ctr(b.likes, b.views),
            "abs_diff": global_ctr(b.likes, b.views) - global_ctr(a.likes, a.views),
            "ci_low": boot["ci_low"],
            "ci_high": boot["ci_high"],
            "p_linearized": linearized_ttest(a.likes, a.views, b.likes, b.views),
        })
    return pd.DataFrame(rows)


def guardrail_check(df, group_a, group_b, group_col="exp_group", max_drop=0.0, alpha=ALPHA):
    a, b = split_groups(df, group_a, group_b, group_col)
    rows = []
    for col, name in [("views", "views_per_user"), ("likes", "likes_per_user")]:
        rel = b[col].mean() / a[col].mean() - 1
        p = stats.ttest_ind(a[col], b[col], equal_var=False).pvalue
        rows.append({
            "metric": name,
            "mean_a": a[col].mean(),
            "mean_b": b[col].mean(),
            "rel_diff": rel,
            "p_value": p,
            "violated": (rel < -max_drop) and (p < alpha),
        })
    return pd.DataFrame(rows)

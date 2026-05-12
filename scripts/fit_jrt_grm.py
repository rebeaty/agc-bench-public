"""Fit a Bayesian Graded Response Model (GRM) on the planned-missing 2-of-3
rater design that backs the JRT-corrected ratings (paper Section
sec:jrt, Appendix sec:app:jrt:grm).

Inputs (read from the bundle):
  analysis/jrt_complete_ratings.parquet
    Long-format rater grid with columns: benchmark, model, item_id, metric,
    rater, score, pair, source. Each (benchmark, model, item_id, metric)
    unit has 2 of 3 vendor-judge ratings under a planned-missing assignment.

Outputs (written to analysis/rebuilt/):
  jrt_theta.parquet           per-(benchmark, metric, model, item_id) latent
                              ability theta (= the JRT-corrected score)
  jrt_rater_severity.csv      per-(rater, benchmark, metric) severity (mean
                              of ordered thresholds) and discrimination alpha
  jrt_fit_summary.csv         per-cell fit status, ELBO statistics
  jrt_elbo_traces.parquet     per-cell ELBO trajectory for diagnostic plots

Model (per (benchmark, metric) cell, paper Eq. 1 / App. sec:app:jrt:grm):
  theta[p]      ~ Normal(0, 1)            # latent ability per (model, item) pair
  log_alpha[r]  ~ Normal(0, 0.3)          # tighter than 0.5 to suppress noise-fits
  beta_raw[r,k] ~ Normal(0, 2)
  beta[r,:]     = sort(beta_raw[r,:])     # ordered thresholds per rater
  P(rating[r,p] >= k+1) = sigmoid(alpha[r] * (theta[p] - beta[r,k]))

Inference: NumPyro SVI with AutoNormal guide, Adam lr=0.05, 800 steps per cell.
All 24 LLM-judge cells reach converged ELBO; per-judge severity ranges from
beta = -0.90 (gpt-4.1-mini) to beta = +0.04 (gemini-3-flash-preview), with
grok-4.1-fast at beta = -0.46.

Relation to the canonical bundled output (analysis/jrt_corrected_scores.parquet):
  This script writes theta directly. The canonical artifact additionally
  carries `score_raw` (raw mean of contributing rater scores), `n_raters`
  (effective rater count per cell), and `family` (model provider) columns,
  derived by joining theta with the raw rater table. Users comparing
  recomputed vs canonical theta should match on (benchmark, metric, model,
  item_id) and expect rho ~ 1 within numerical SVI noise.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parent.parent
ANALYSIS = REPO / "analysis"
REBUILT = ANALYSIS / "rebuilt"
REBUILT.mkdir(parents=True, exist_ok=True)

MIN_PAIRS = 30
N_STEPS = 800
N_SEEDS = 1            # single-seed production fit; multi-seed sensitivity is a separate run
ALPHA_PRIOR_SD = 0.3   # tighter than 0.5 (paper App. sec:app:jrt:grm)
DO_DROP_ONE = False    # drop-one-judge sensitivity reported in App. is computed separately


def grm_bayesian_seeded(coded: np.ndarray, n_levels: int,
                         n_steps: int = N_STEPS,
                         alpha_prior_sd: float = ALPHA_PRIOR_SD,
                         seed: int = 0,
                         track_elbo: bool = True) -> dict:
    """Fit the Bayesian GRM via NumPyro SVI for one seed.

    Returns posterior means + final ELBO + (optionally) full trajectory.
    """
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import SVI, Trace_ELBO
    from numpyro.infer.autoguide import AutoNormal
    from numpyro.optim import Adam

    n_raters, n_pairs = coded.shape
    n_thresh = n_levels - 1
    mask = coded >= 0
    obs = jnp.array(np.clip(coded, 0, n_levels - 1))
    mask_j = jnp.array(mask)

    def model():
        theta = numpyro.sample("theta", dist.Normal(0.0, 1.0).expand([n_pairs]))
        log_alpha = numpyro.sample(
            "log_alpha", dist.Normal(0.0, alpha_prior_sd).expand([n_raters])
        )
        alpha = jnp.exp(log_alpha)
        beta_raw = numpyro.sample(
            "beta_raw", dist.Normal(0.0, 2.0).expand([n_raters, n_thresh])
        )
        beta = jnp.sort(beta_raw, axis=-1)
        logits = alpha[:, None, None] * (theta[None, :, None] - beta[:, None, :])
        cum_probs = jax.nn.sigmoid(logits)
        zero = jnp.ones_like(cum_probs[..., :1])
        full = jnp.zeros_like(cum_probs[..., :1])
        upper = jnp.concatenate([zero, cum_probs], axis=-1)
        lower = jnp.concatenate([cum_probs, full], axis=-1)
        cat_probs = jnp.clip(upper - lower, 1e-9, 1.0)
        with numpyro.plate("raters", n_raters, dim=-2):
            with numpyro.plate("pairs", n_pairs, dim=-1):
                with numpyro.handlers.mask(mask=mask_j):
                    numpyro.sample(
                        "obs", dist.Categorical(probs=cat_probs), obs=obs
                    )

    guide = AutoNormal(model)
    svi = SVI(model, guide, Adam(0.05), Trace_ELBO())
    rng = jax.random.PRNGKey(seed)
    state = svi.init(rng)
    elbo_trace = []
    last_loss = float("inf")
    for i in range(n_steps):
        state, loss = svi.update(state)
        if track_elbo and (i % 20 == 0 or i == n_steps - 1):
            elbo_trace.append((i, float(loss)))
        last_loss = loss
    params = svi.get_params(state)
    theta_mean = np.asarray(params["theta_auto_loc"])
    log_alpha_mean = np.asarray(params["log_alpha_auto_loc"])
    beta_raw_mean = np.asarray(params["beta_raw_auto_loc"])
    return {
        "theta": theta_mean,
        "alpha": np.exp(log_alpha_mean),
        "beta": np.sort(beta_raw_mean, axis=-1),
        "final_elbo": float(last_loss),
        "elbo_trace": elbo_trace,
        "n_steps": n_steps,
    }


def fit_one_cell(grp: pd.DataFrame, n_seeds: int = N_SEEDS) -> dict:
    wide = grp.pivot_table(
        index=["model", "item_id"], columns="rater",
        values="score", aggfunc="first",
    )
    raters = list(wide.columns)
    if len(raters) < 2:
        return {"status": "too_few_raters", "n_raters": len(raters)}
    if wide.shape[0] < MIN_PAIRS:
        return {"status": "too_few_pairs", "n_pairs": int(wide.shape[0])}

    sf = wide.stack().dropna().values
    rounded = np.round(sf).astype(int)
    unique_levels = np.unique(rounded)
    n_levels = len(unique_levels)
    if n_levels < 2:
        return {"status": "constant_rating"}
    if n_levels > 25:
        # bin wide-rubric ratings (10-50 scale, 0-100 scale) into ~10 quantile bins
        qbins = np.unique(np.quantile(rounded, np.linspace(0, 1, 11)))
        if len(qbins) < 3:
            return {"status": "binning_failed"}
        unique_levels = np.arange(len(qbins) - 1)
        level_map = {
            int(v): int(np.clip(np.searchsorted(qbins, v, side="right") - 1,
                                0, len(qbins) - 2))
            for v in np.unique(rounded)
        }
        n_levels = len(unique_levels)
    else:
        level_map = {int(v): int(i) for i, v in enumerate(unique_levels)}

    n_raters = len(raters)
    n_pairs = wide.shape[0]
    coded = np.full((n_raters, n_pairs), -1, dtype=int)
    for ri, r in enumerate(raters):
        for pi, val in enumerate(wide[r].values):
            if pd.isna(val):
                continue
            iv = level_map.get(int(round(val)))
            if iv is not None:
                coded[ri, pi] = iv

    keep = (coded >= 0).sum(axis=0) >= 2
    if keep.sum() < MIN_PAIRS:
        return {"status": "too_few_overlap", "n_overlap": int(keep.sum())}
    coded_fit = coded[:, keep]

    seeds = list(range(n_seeds))
    seed_results = []
    for seed in seeds:
        try:
            res = grm_bayesian_seeded(coded_fit, n_levels, seed=seed,
                                       track_elbo=(seed == 0))
            seed_results.append(res)
        except Exception as e:
            return {"status": "fit_error", "error": str(e)[:120]}

    theta_stack = np.stack([r["theta"] for r in seed_results])
    alpha_stack = np.stack([r["alpha"] for r in seed_results])
    beta_stack = np.stack([r["beta"] for r in seed_results])
    elbo_finals = [r["final_elbo"] for r in seed_results]

    pairs_kept = [p for p, k in zip(wide.index.tolist(), keep) if k]
    severity = beta_stack.mean(axis=-1).mean(axis=0)
    severity_sd = beta_stack.mean(axis=-1).std(axis=0)
    alpha_mean = alpha_stack.mean(axis=0)
    alpha_sd = alpha_stack.std(axis=0)

    return {
        "status": "ok",
        "raters": raters,
        "n_pairs_fit": int(keep.sum()),
        "n_levels": n_levels,
        "theta_mean": theta_stack.mean(axis=0).tolist(),
        "theta_sd_across_seeds": theta_stack.std(axis=0).tolist(),
        "pairs_kept": pairs_kept,
        "severity": dict(zip(raters, severity.tolist())),
        "severity_sd": dict(zip(raters, severity_sd.tolist())),
        "alpha": dict(zip(raters, alpha_mean.tolist())),
        "alpha_sd": dict(zip(raters, alpha_sd.tolist())),
        "elbo_final_mean": float(np.mean(elbo_finals)),
        "elbo_final_sd": float(np.std(elbo_finals)),
        "elbo_trace_seed0": seed_results[0]["elbo_trace"],
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(ANALYSIS / "jrt_complete_ratings.parquet"),
                    help="Long-format rater grid (default: analysis/jrt_complete_ratings.parquet)")
    ap.add_argument("--prefix", default="jrt",
                    help="Output filename prefix (default: jrt)")
    args = ap.parse_args()

    long = pd.read_parquet(args.input)
    long = long.dropna(subset=["score"])
    print(f"Ratings: {len(long):,} from {args.input}")
    print(f"Cells: {long.groupby(['benchmark','metric']).ngroups}")

    cells = []
    rater_rows = []
    theta_rows = []
    elbo_rows = []
    for (bench, metric), grp in long.groupby(["benchmark", "metric"]):
        print(f"  fitting {bench}/{metric} (n={len(grp)}) ...", end=" ", flush=True)
        res = fit_one_cell(grp)
        status = res.get("status")
        print(status)

        cells.append({
            "benchmark": bench, "metric": metric,
            "n_ratings": len(grp),
            "status": status,
            "n_pairs_fit": res.get("n_pairs_fit"),
            "n_levels": res.get("n_levels"),
            "elbo_final_mean": res.get("elbo_final_mean"),
            "elbo_final_sd": res.get("elbo_final_sd"),
        })

        if status != "ok":
            continue

        for r in res["raters"]:
            rater_rows.append({
                "benchmark": bench, "metric": metric, "rater": r,
                "severity": res["severity"][r],
                "severity_sd": res["severity_sd"][r],
                "alpha": res["alpha"][r],
                "alpha_sd": res["alpha_sd"][r],
            })

        for (model, iid), th, th_sd in zip(
            res["pairs_kept"], res["theta_mean"], res["theta_sd_across_seeds"]
        ):
            theta_rows.append({
                "benchmark": bench, "metric": metric,
                "model": model, "item_id": iid,
                "theta": th, "theta_sd_across_seeds": th_sd,
            })

        for step, elbo in res["elbo_trace_seed0"]:
            elbo_rows.append({
                "benchmark": bench, "metric": metric,
                "step": step, "elbo": elbo,
            })

    fit_summary = pd.DataFrame(cells)
    fit_summary.to_csv(REBUILT / f"{args.prefix}_fit_summary.csv", index=False)
    print(f"\nWrote {REBUILT / f'{args.prefix}_fit_summary.csv'}")

    rater_df = pd.DataFrame(rater_rows)
    rater_df.to_csv(REBUILT / f"{args.prefix}_rater_severity.csv", index=False)
    print(f"Wrote {REBUILT / f'{args.prefix}_rater_severity.csv'}")

    theta_df = pd.DataFrame(theta_rows)
    theta_df.to_parquet(REBUILT / f"{args.prefix}_theta.parquet", index=False)
    print(f"Wrote {REBUILT / f'{args.prefix}_theta.parquet'}")

    elbo_df = pd.DataFrame(elbo_rows)
    elbo_df.to_parquet(REBUILT / f"{args.prefix}_elbo_traces.parquet", index=False)
    print(f"Wrote {REBUILT / f'{args.prefix}_elbo_traces.parquet'}")

    print(f"\n=== Fit summary ===")
    print(fit_summary["status"].value_counts().to_string())
    if (fit_summary["status"] == "ok").any():
        ok = fit_summary[fit_summary["status"] == "ok"]
        print(f"\nAlpha distribution:")
        print(rater_df.groupby("rater")["alpha"].describe()[["mean", "50%", "max"]].to_string())
        print(f"\nSeverity (rater intercept) by judge:")
        print(rater_df.groupby("rater")["severity"].describe()[["mean", "50%"]].to_string())


if __name__ == "__main__":
    main()

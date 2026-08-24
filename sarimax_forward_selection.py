import numpy as np
import pandas as pd


def tune_sarimax(
    train_y, test_y,
    train_exog=None, test_exog=None,
    grid=GRID, m=M,
    lag_cols: list[str] | None = None,
    exog_cols: list[str] | None = None,   # NEW: restrict to this subset of covariates
) -> pd.DataFrame:
    """
    Grid-search SARIMAX; returns a MAPE-sorted DataFrame.
    Identical to Notebook 01, plus an `exog_cols` filter so this can be
    called on any subset of the available exogenous covariates.
    """
    lag_cols = list(lag_cols or [])

    # --- NEW: subset the exog frames to just the candidate covariates ---
    if exog_cols is not None:
        train_exog_use = train_exog[exog_cols] if train_exog is not None else None
        test_exog_use = test_exog[exog_cols] if test_exog is not None else None
    else:
        train_exog_use = train_exog
        test_exog_use = test_exog

    rows = []
    for p, d, q, P, D, Q in grid:
        try:
            current_trend = "c" if D == 1 else "n"
            result = SARIMAX(
                train_y,
                exog=train_exog_use,
                order=(p, d, q),
                seasonal_order=(P, D, Q, m),
                trend=current_trend,
                enforce_stationarity=False, enforce_invertibility=False,
            ).fit(disp=False)

            if lag_cols:
                fc = recursive_forecast_with_exog(
                    result_obj=result,
                    history_y=train_y,
                    future_index=test_y.index,
                    future_cov_df=test_exog_use,
                    lag_cols=lag_cols,
                    all_exog_cols=list(train_exog_use.columns) if train_exog_use is not None else [],
                ).values
            else:
                fc = result.get_forecast(steps=len(test_y), exog=test_exog_use).predicted_mean.values

            yt = test_y.values.astype(float)
            mape = float(np.mean(np.abs((yt - fc) / yt))) * 100
            rows.append({"p": p, "d": d, "q": q, "P": P, "D": D, "Q": Q, "MAPE": round(mape, 4)})
        except Exception:
            pass

    return pd.DataFrame(rows).sort_values("MAPE").reset_index(drop=True)


def derive_baseline_grid(
    train_y, test_y,
    grid=GRID, m=M,
    top_n: int = 3,
):
    """
    Run ONE full grid search on the target series with no exog at all, and
    return the top_n (p,d,q,P,D,Q) orders. This gives a cheap, defensible
    grid_select for forward_select_exog without ever touching the covariates
    (order structure comes from the series' own ACF/PACF/seasonality, not
    from which exog you'll eventually add).
    """
    baseline = tune_sarimax(train_y, test_y, train_exog=None, test_exog=None, grid=grid, m=m)
    if baseline.empty:
        raise ValueError("Baseline (no-exog) grid search produced no valid fits — check GRID/series.")
    top = baseline.head(top_n)[["p", "d", "q", "P", "D", "Q"]]
    return [tuple(row) for row in top.values]


def forward_select_exog(
    train_y, test_y,
    train_exog, test_exog,
    candidate_cols: list[str] | None = None,
    grid_select=None,          # cheap grid used DURING selection; auto-derived from baseline if None
    grid_final=GRID,           # full grid used for the FINAL winning subset
    m=M,
    lag_cols: list[str] | None = None,
    min_improvement: float = 0.0,   # require at least this much MAPE drop (in pp) to keep adding
    verbose: bool = True,
):
    """
    Forward feature selection over exogenous covariates for SARIMAX.

    At each step, tries adding each remaining candidate covariate to the
    currently selected set, tunes SARIMAX on that trial set (grid_select),
    and keeps the covariate whose addition gives the lowest MAPE — as long
    as it improves on the current best by more than `min_improvement`.

    Returns
    -------
    selected_cols : list[str]
        Best subset of covariates found (possibly a single covariate, or empty).
    history : pd.DataFrame
        One row per selection step: which covariate was added, best MAPE
        and best (p,d,q,P,D,Q) for that step.
    final_grid_result : pd.DataFrame
        Full grid-search result (all param combos, sorted by MAPE) for the
        winning covariate subset, using grid_final.
    """
    candidate_cols = list(candidate_cols or train_exog.columns)

    # Auto-derive a cheap, defensible grid for the selection phase from a
    # single no-exog baseline run, rather than guessing a fixed order.
    if grid_select is None:
        grid_select = derive_baseline_grid(train_y, test_y, grid=grid_final, m=m, top_n=3)
        if verbose:
            print(f"[baseline] auto-derived grid_select from no-exog run: {grid_select}")

    selected: list[str] = []
    remaining = candidate_cols.copy()
    best_overall_mape = np.inf
    history_rows = []

    while remaining:
        step_results = []  # (candidate, best_mape_for_trial_set, best_params_row)

        for cand in remaining:
            trial_cols = selected + [cand]
            res = tune_sarimax(
                train_y, test_y,
                train_exog=train_exog, test_exog=test_exog,
                grid=grid_select, m=m,
                lag_cols=lag_cols,
                exog_cols=trial_cols,
            )
            if res.empty:
                continue
            best_row = res.iloc[0]
            step_results.append((cand, best_row["MAPE"], best_row))

        if not step_results:
            # every candidate failed to fit (e.g. singular matrix) — stop
            break

        step_results.sort(key=lambda x: x[1])
        best_cand, best_cand_mape, best_cand_row = step_results[0]

        improved = (best_overall_mape - best_cand_mape) > min_improvement
        if verbose:
            print(f"[step {len(selected)+1}] trying +{best_cand!r}: MAPE={best_cand_mape:.4f} "
                  f"(current best={best_overall_mape:.4f}) -> {'KEEP' if improved else 'STOP'}")

        if not improved:
            break

        selected.append(best_cand)
        remaining.remove(best_cand)
        best_overall_mape = best_cand_mape
        history_rows.append({
            "step": len(selected),
            "added": best_cand,
            "selected_cols": list(selected),
            "MAPE": best_cand_mape,
            **{k: best_cand_row[k] for k in ["p", "d", "q", "P", "D", "Q"]},
        })

    history = pd.DataFrame(history_rows)

    # Final full grid search on the winning subset (or empty exog if none selected)
    final_grid_result = tune_sarimax(
        train_y, test_y,
        train_exog=train_exog, test_exog=test_exog,
        grid=grid_final, m=m,
        lag_cols=lag_cols,
        exog_cols=selected if selected else None,
    )

    return selected, history, final_grid_result

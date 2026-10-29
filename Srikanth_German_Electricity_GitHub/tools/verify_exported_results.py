"""Independent, read-only verification of German demand forecasting exports.

Usage: python verify_exported_results.py PATH_TO/German_Electricity_Forecasting
No model retraining, network requests, or notebook globals are required.
"""
from __future__ import annotations
import argparse
import ast
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def verify(project: Path) -> pd.DataFrame:
    tables = project / "outputs" / "tables"
    models = project / "outputs" / "models"
    data = project / "data"
    records = []

    def check(label, condition, detail=""):
        ok = bool(condition)
        records.append({"test": label, "passed": ok, "detail": str(detail)[:240]})
        return ok

    def read_csv(name):
        path = tables / name
        if not path.is_file():
            raise FileNotFoundError(str(path))
        return pd.read_csv(path)

    weekly = read_csv("weekly_predictions_vs_actual.csv")
    summary = read_csv("all_weekly_model_metrics.csv")
    grid = read_csv("sarima_full_grid.csv")
    historical_weekly = pd.read_csv(data / "german_load_weekly.csv")
    check("104-week fixed-origin evaluation", len(weekly) == 104)
    check("one observation per model per test week", len(summary) == len(weekly.columns) - 2)
    dates = pd.to_datetime(weekly.iloc[:, 0], errors="coerce")
    check("weekly chronology unique and increasing", dates.notna().all() and dates.is_unique and dates.is_monotonic_increasing)
    check("Sunday week endings", (dates.dt.dayofweek == 6).all())
    check("weekly forecasts and observations finite", np.isfinite(weekly.iloc[:, 1:].to_numpy(dtype=float)).all())

    actual = weekly["actual_MW"].to_numpy(dtype=float)
    model_columns = [c for c in weekly.columns if c not in {weekly.columns[0], "actual_MW"}]
    check("every weekly model has an evaluation record", set(model_columns) == set(summary["Model"]))
    by_model = summary.set_index("Model")
    weekly_train = historical_weekly.iloc[:len(historical_weekly) - len(weekly)]["load"].to_numpy(dtype=float)
    seasonal_period = 52
    denominator = np.mean(np.abs(weekly_train[seasonal_period:] - weekly_train[:-seasonal_period]))
    check("weekly MASE training denominator positive", np.isfinite(denominator) and denominator > 0)
    for model_name in model_columns:
        yhat = weekly[model_name].to_numpy(dtype=float)
        diff = actual - yhat
        reconstructed = {
            "RMSE": float(np.sqrt(np.mean(diff ** 2))),
            "MAE": float(np.mean(np.abs(diff))),
            "sMAPE_pct": float(200 * np.mean(np.abs(diff) / (np.abs(actual) + np.abs(yhat)))),
            "MASE": float(np.mean(np.abs(diff)) / denominator),
        }
        available = by_model.loc[model_name]
        ok = all(np.isclose(reconstructed[key], float(available[key]), atol=1e-5, rtol=1e-6)
                 for key in reconstructed)
        check(f"independent weekly metrics: {model_name}", ok)

    observed_specifications = {(tuple(ast.literal_eval(o)), tuple(ast.literal_eval(s)))
                               for o, s in zip(grid["order"], grid["seasonal_order"])}
    expected_specifications = {
        ((p, d, q), (P, D, Q, 52))
        for p, d, q, P, D, Q in itertools.product(range(7), range(3), range(7), range(2), range(2), range(2))
    }
    check("complete 1,176 unique SARIMA specifications", len(grid) == 1176 and observed_specifications == expected_specifications)
    check("SARIMA primary diagnostic files present", all((tables / n).is_file() for n in (
        "sarima_diagnostic_summary.csv", "sarima_ljung_box.csv", "sarima_interval_coverage.csv",
        "sarima_diagnostic_interpretation.csv")))
    if (tables / "sarima_diagnostic_summary.csv").exists():
        diagnostic = read_csv("sarima_diagnostic_summary.csv").iloc[0]
        check("SARIMA residual screen truthfully labelled", bool(diagnostic["passes_residual_screen"]) ==
              (float(diagnostic["minimum_ljung_box_p"]) > float(diagnostic["residual_screen_alpha"])))
    check("all tree models persisted", all((models / (name + ".joblib")).is_file() for name in
          ("rf_conditional", "rf_operational", "gbr_conditional", "gbr_operational")))
    check("tree-model reload verification recorded", (tables / "tree_model_reload_checks.csv").is_file())
    check("tree feature schema persisted", (models / "tree_feature_schema.json").is_file())
    if (tables / "tree_model_reload_checks.csv").is_file():
        reload_rows = read_csv("tree_model_reload_checks.csv")
        check("all four tree models reloaded and verified", len(reload_rows) == 4 and reload_rows["passed"].astype(bool).all())

    hourly_file = tables / "hourly_predictions_vs_actual.csv"
    if hourly_file.exists():
        hourly = pd.read_csv(hourly_file)
        hourly_metrics = read_csv("hourly_lstm_metrics.csv").set_index("Model")
        check("hourly predictions cover whole test horizon", len(hourly) >= 17000)
        check("common hourly MASE denominator", hourly_metrics["MASE_scaling_hours"].eq(168).all()
              and hourly_metrics["MASE_denominator_MW"].nunique() == 1)
        truth = hourly["actual_MW"].to_numpy(dtype=float)
        hourly_ok = True
        for model_name in ("LSTM Fixed-Origin", "Hourly Seasonal Naive", "Hourly Annual Seasonal Naive"):
            pred = hourly[model_name].to_numpy(dtype=float)
            calc_rmse = np.sqrt(np.mean((truth - pred)**2))
            calc_mae = np.mean(np.abs(truth - pred))
            calculated_mase = calc_mae / float(hourly_metrics.loc[model_name, "MASE_denominator_MW"])
            row = hourly_metrics.loc[model_name]
            hourly_ok &= bool(np.isclose(row["RMSE"], calc_rmse, atol=1e-5)
                              and np.isclose(row["MAE"], calc_mae, atol=1e-5)
                              and np.isclose(row["MASE"], calculated_mase, atol=1e-5))
        check("independent hourly RMSE/MAE/MASE", hourly_ok)
        check("LSTM model and training-only scaler persisted", (models / "lstm_final_retrained.keras").is_file()
              and (models / "lstm_standard_scaler.joblib").is_file())
        check("LSTM reload verification recorded", (tables / "lstm_model_reload_checks.csv").is_file())
        if (tables / "lstm_model_reload_checks.csv").is_file():
            lstm_rows = read_csv("lstm_model_reload_checks.csv")
            check("reloaded LSTM passed numerical comparison", len(lstm_rows) == 1 and bool(lstm_rows.loc[0, "passed"]))
    else:
        check("hourly LSTM required by assignment", False, "Hourly forecast table missing")

    check("training-only weather provenance recorded", (tables / "weather_source_provenance.json").is_file()
          and (tables / "forecast_covariate_availability.csv").is_file())
    result = pd.DataFrame(records)
    path = tables / "independent_external_verification.csv"
    result.to_csv(path, index=False)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("project_dir", type=Path)
    args = ap.parse_args()
    try:
        results = verify(args.project_dir)
    except Exception as exc:
        print(f"Verification could not complete: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(results.to_string(index=False))
    passed = int(results["passed"].sum())
    total = len(results)
    print(f"Independent export verification: {passed}/{total} checks passed")
    return 0 if passed == total else 1

if __name__ == "__main__":
    raise SystemExit(main())

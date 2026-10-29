"""Read-only integrity checks for the forecasting repository."""
from __future__ import annotations
import ast
import csv
import hashlib
import io
import itertools
import json
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent.parent


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    lines = (ROOT / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
    for line in lines:
        if not line.strip():
            continue
        expected, filename = line.split("  ", 1)
        actual = digest(ROOT / filename)
        assert actual == expected, f"SHA256 mismatch: {filename}"
    csv_path = ROOT / "checkpoint/sarima_full_grid.csv"
    metadata = json.loads((ROOT / "checkpoint/sarima_full_grid_metadata.json").read_text(encoding="utf-8"))
    with csv_path.open(encoding="utf-8", newline="") as file:
        records = list(csv.DictReader(file))
    observed = {(tuple(ast.literal_eval(r["order"])), tuple(ast.literal_eval(r["seasonal_order"]))) for r in records}
    expected = {((p, d, q), (P, D, Q, 52))
                for p, d, q, P, D, Q in itertools.product(range(7), range(3), range(7), range(2), range(2), range(2))}
    assert len(records) == 1176 and observed == expected and metadata["grid_size"] == 1176, "Incomplete SARIMA grid"
    with ZipFile(ROOT / "checkpoint/German_Electricity_SARIMA_Checkpoint.zip") as archive:
        assert archive.testzip() is None
        assert {"sarima_full_grid.csv", "sarima_full_grid_metadata.json"}.issubset(archive.namelist())
    with ZipFile(ROOT / "results/German_Electricity_Forecasting_Colab_Results.zip") as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == 117
        expected_outputs = {
            "outputs/tables/all_weekly_model_metrics.csv",
            "outputs/tables/weekly_predictions_vs_actual.csv",
            "outputs/tables/hourly_lstm_metrics.csv",
            "outputs/models/lstm_final_retrained.keras",
            "outputs/models/lstm_standard_scaler.joblib",
            "outputs/tables/independent_external_verification.csv",
        }
        assert expected_outputs.issubset(archive.namelist()), "Missing result artifacts"
    notebook = json.loads((ROOT / "German_Electricity_Demand_Forecasting.ipynb").read_text(encoding="utf-8"))
    code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) == 56 and all(c.get("execution_count") is not None for c in code_cells)
    assert not any(o.get("output_type") == "error" for c in code_cells for o in c.get("outputs", []))
    for i, cell in enumerate(code_cells):
        compile("".join(cell["source"]), f"notebook-cell-{i}", "exec")
    print("Repository files validated")
    print("SARIMA grid: 1,176 distinct configurations")
    print("Notebook: 56 executed code cells with no saved exception")
    print("Colab results: 117 original files; ZIP CRC check passed")


if __name__ == "__main__":
    main()

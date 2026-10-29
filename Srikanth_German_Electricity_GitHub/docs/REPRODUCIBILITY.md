# Reproducing the electricity-demand forecasting experiment

## Google Colab

1. Open `German_Electricity_Demand_Forecasting.ipynb` in Colab.
2. Run the upload cell at the top and select `checkpoint/German_Electricity_SARIMA_Checkpoint.zip`. The control extracts `sarima_full_grid.csv` and `sarima_full_grid_metadata.json` into `/content/colab_uploaded_inputs`.
3. Run all remaining cells. The notebook checks the 1,176 search configurations, fingerprint of the training series, and model settings before reusing the grid. It uses `statsmodels==0.15.0` for checkpoint compatibility.
4. Data come from OPSD and Open-Meteo via internet downloads unless suitable local inputs are supplied.
5. Download `/content/German_Electricity_Forecasting_Results.zip` at the end. The notebook also creates a separate SARIMA checkpoint ZIP.

The completed Colab run is preserved in this repository as an executed notebook and a 117-file results archive. The test does not mean every possible operating-system or Python configuration has been verified.

## Local Jupyter

1. Use Python with the versions in `requirements.txt`. Install them with `python -m pip install -r requirements.txt`.
2. Start Jupyter from the repository root, which contains `checkpoint/`. This is important because local paths are based on the current working directory.
3. Open and run `German_Electricity_Demand_Forecasting.ipynb` in order.
4. The Colab file-upload cell is skipped automatically outside Colab. The repository `checkpoint/` folder is detected, so the multi-hour SARIMA parameter search is not repeated when the checkpoint passes compatibility checks.
5. Generated output files are saved in `German_Electricity_Forecasting/outputs/`, and a results ZIP is created in the working directory.

If the dependency versions, training-series fingerprint, or SARIMA configuration differ from the saved checkpoint, the notebook rejects the cached grid rather than silently treating it as compatible. Resolve the mismatch in the environment or provide a genuine matching checkpoint. Do not bypass the fingerprint check or relabel incompatible results. A clean local full execution has not been separately verified.

## Kaggle

Attach the two raw checkpoint files from `checkpoint/` as one Kaggle Dataset using Add Input, and import the notebook. Internet must be enabled for first-time source retrieval. The notebook detects Kaggle-specific input and output locations. Without a compatible checkpoint, 1,176 SARIMA candidates must be fitted.

## Read-only verification of the saved experiment

The archive under `results/` includes the original cache data, all figures and tables, model weights, preprocessing objects and verification records.

```bash
python tools/check_repository.py
mkdir reproduced
# Extract results/German_Electricity_Forecasting_Colab_Results.zip into reproduced/
python -m pip install -r requirements.txt
python tools/verify_exported_results.py reproduced/
```

No training occurs during read-only verification. Existing model weights, scalers and prediction files were saved and reload-verified during the recorded Colab execution. These checks confirm the recorded output, not accuracy on a new dataset.

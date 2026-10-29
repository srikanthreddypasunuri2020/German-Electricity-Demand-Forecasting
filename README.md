# German Electricity Demand Forecasting

Srikanth Pasunoori, student ID 24090495

This project compares weekly electricity-demand forecasts for Germany using four simple benchmarks, SARIMA, SARIMAX, Random Forest, Gradient Boosting and an hourly LSTM. The historical data cover January 2015 to September 2020. The forecast is issued from October 2018 and evaluated on the following 104 weeks.

The notebook distinguishes operational forecasts from conditional experiments that use observed test-period temperature. The latter cannot be produced at the 2018 forecast origin.

## Repository contents

- `German_Electricity_Demand_Forecasting.ipynb`: executed notebook and recorded outputs
- `checkpoint/`: complete 1,176-row SARIMA grid, metadata and upload ZIP
- `results/German_Electricity_Forecasting_Colab_Results.zip`: exported data, plots, predictions, diagnostics and fitted models
- `outputs/figures/`: selected figures for viewing without opening the notebook
- `outputs/tables/`: principal numerical results and validation records
- `tools/`: read-only checks for the repository and forecast tables
- `docs/REPRODUCIBILITY.md`: instructions for Colab, Jupyter and Kaggle
- `requirements.txt`: package versions recorded in the successful Colab run

## Data

The electricity-demand data come from Open Power System Data, release 2020-10-06:
https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv

The notebook uses the German actual-load column `DE_load_actual_entsoe_transparency`. Berlin temperature comes from the Open-Meteo historical archive; national holiday features use the Python `holidays` package. The notebook downloads the electricity and weather data when local copies are unavailable, so internet access is needed for a first run without cached inputs.

## Run in Google Colab

1. Open `German_Electricity_Demand_Forecasting.ipynb` in Colab.
2. Run the first code cell. When prompted, upload `checkpoint/German_Electricity_SARIMA_Checkpoint.zip`.
3. Run the remaining cells in order. The checkpoint is checked against the training data and the grid configuration before use.
4. Download `/content/German_Electricity_Forecasting_Results.zip` after the final cell.

The checkpoint contains results for all 1,176 SARIMA specifications. Reusing it skips the original exhaustive search. The full parameter grid can be calculated from scratch by disabling checkpoint reuse, but this takes substantially longer.

## Run in local Jupyter

Use a compatible Python environment and install the recorded packages:

```bash
python -m pip install -r requirements.txt
jupyter notebook
```

Start Jupyter from the repository directory and open the notebook. The Colab upload cell is skipped outside Colab. The notebook looks for the matching SARIMA files in `checkpoint/`, and stores generated files under `German_Electricity_Forecasting/`.

The completed notebook was tested in Colab. A separate full local run has not been tested. Refer to `docs/REPRODUCIBILITY.md` for dependency and checkpoint requirements.

## Recorded results

| Forecast | RMSE (MW) |
| --- | ---: |
| Seasonal naive, weekly | 2,988.25 |
| Random Forest, conditional weekly | 2,939.49 |
| Gradient Boosting, operational weekly | 2,998.36 |
| Random Forest, operational weekly | 3,003.58 |
| SARIMA(0,0,0)(1,1,1)[52], weekly | 3,890.56 |
| LSTM, hourly | 7,601.61 |

These are results from the recorded execution. All 15 weekly scores are in `outputs/tables/all_weekly_model_metrics.csv`. The seasonal-naive method was the strongest operational weekly benchmark for this fixed-origin evaluation. The conditional Random Forest uses temperature that was not known at the forecast origin.

## Verify the saved results

The following check uses the Python standard library:

```bash
python tools/check_repository.py
```

For the independent forecast-table checks, extract the ZIP under `results/` into a directory named `reproduced`, then run:

```bash
python tools/verify_exported_results.py reproduced/
```

The verifier reads saved results; it does not retrain the models. See `docs/REPRODUCIBILITY.md` for details.

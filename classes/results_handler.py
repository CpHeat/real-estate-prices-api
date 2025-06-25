from abc import ABC

import pandas as pd


class ResultsHandler(ABC):

    @classmethod
    def show_metrics_comparison(cls, results: dict) -> None:

        rows = []

        for model_name, model_results in results.items():
            for loc_type, scores in model_results.items():
                parts = loc_type.lower().split()
                city = parts[0].capitalize()
                type = parts[1]

                # Get metrics
                r2_train = scores['train results']['R²']
                r2_test = scores['test results']['R²']
                mse_train = scores['train results']['MSE']
                mse_test = scores['test results']['MSE']
                rmse_train = scores['train results']['RMSE']
                rmse_test = scores['test results']['RMSE']
                mae_train = scores['train results']['MAE']
                mae_test = scores['test results']['MAE']

                rows.append({
                    'Model': model_name.title(),
                    'City': city,
                    'Type': type,
                    'R² Train': round(r2_train, 4),
                    'R² Test': round(r2_test, 4),
                    'MSE Train': round(mse_train, 4),
                    'MSE Test': round(mse_test, 4),
                    'RMSE Train': round(rmse_train, 4),
                    'RMSE Test': round(rmse_test, 4),
                    'MAE Train': round(mae_train, 4),
                    'MAE Test': round(mae_test, 4)
                })

        df = pd.DataFrame(rows)
        print(df.to_string(index=False))
from typing import Any, Dict
from models import AllModels, MF
from itertools import product
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.sparse import coo_array


def _perform_cross_validation(
    model: AllModels,
    train_data: coo_array,
    test_data: coo_array,
    hyperparameters: tuple[Any],
) -> tuple[str, tuple[Any], float, float]:
    model.fit(train_data, *hyperparameters, seed=42)
    mae = model.accuracy_mae(test_data)
    rmse = model.accuracy_rmse(test_data)
    return model.__class__.__name__, hyperparameters, mae, rmse


# Performs grid search with the given model, datasets and hyperparameters ranges
def grid_search(
    model: AllModels,
    train_set: coo_array,
    test_set: coo_array,
    params_ranges: Dict[str, list[Any]],
):
    if isinstance(model, MF):
        n_factors_range = params_ranges["n_factors"]
        epochs = params_ranges["epochs"]
        lr = params_ranges["lr"]
        reg = params_ranges["reg"]
        batch_size = params_ranges["batch_size"]
        lr_decay_factor = params_ranges["lr_decay_factor"]

        prod = product(n_factors_range, epochs, lr, reg, batch_size, lr_decay_factor)
        hyperparameter_combinations = [comb for comb in prod]
        results: list[tuple[str, tuple[Any], float, float]] = [
            result
            for result in Parallel(n_jobs=-1, backend="loky")(
                delayed(_perform_cross_validation)(model, train_set, test_set, params)
                for params in tqdm(
                    hyperparameter_combinations, desc="Grid search in progress..."
                )
            )
            if result is not None
        ]
        return results
    else:
        raise RuntimeError("Unsupported model")

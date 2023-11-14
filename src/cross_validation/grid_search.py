from typing import Any
from itertools import product
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.sparse import coo_array


def _perform_cross_validation(
    model: Any,
    train_data: coo_array,
    test_data: coo_array,
    hyperparameters: tuple[Any],
) -> tuple[tuple[Any], float, float]:
    model.fit(train_data, *hyperparameters)
    mae = model.accuracy_mae(test_data)
    rmse = model.accuracy_rmse(test_data)
    return hyperparameters, mae, rmse


def grid_search(
    model: Any,
    train_set: coo_array,
    test_set: coo_array,
    prod: product,
    sequential=False,
):
    """
    Performs grid search with the given model, datasets and hyperparameters ranges
    """
    hyperparameter_combinations = [comb for comb in prod]
    results: list[tuple[tuple[Any], float, float]] = []
    if sequential:
        for comb in hyperparameter_combinations:
            results.append(_perform_cross_validation(model, train_set, test_set, comb))
    else:
        results = [
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

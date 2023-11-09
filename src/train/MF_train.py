from cross_validation import grid_search
from models.CF import MF
from scipy.sparse import coo_array
from tabulate import tabulate


def MF_cross_validate(model: MF, train_set: coo_array, test_set: coo_array):
    n_factors_range = [8, 10, 12]
    epochs_range = [10, 20, 30]
    lr_range = [0.005, 0.009, 0.015]
    reg_range = [0.001, 0.002, 0.003]
    batch_size_range = [2, 4, 8, 16]
    lr_decay_factor_range = [0.5, 0.9, 0.99]
    hyperparameters_ranges = {
        "n_factors": n_factors_range,
        "epochs": epochs_range,
        "lr": lr_range,
        "reg": reg_range,
        "batch_size": batch_size_range,
        "lr_decay_factor": lr_decay_factor_range,
    }
    results = grid_search(model, train_set, test_set, hyperparameters_ranges)

    table = []
    for label, hyperparameters, mae, rmse in results:
        table.append([label, hyperparameters, mae, rmse])

    headers = ["Model Label", "MAE", "RMSE", "Runtime (seconds)"]
    print(tabulate(table, headers=headers, tablefmt="grid"))

from abc import ABC, abstractmethod
import math
import os
from typing import Any, Callable, Literal
import joblib
import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm
from data import Data
from utils import (
    precision,
    recall,
    f1_score,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    normalized_discounted_cumulative_gain,
    generate_combinations,
    batch_generator,
    dict_to_hash,
)
from tabulate import tabulate
from joblib import Parallel, delayed
from scipy.sparse import coo_array


class Recommender_System(ABC):
    """
    Abstract class that serves as main blueprint for the recommender systems' APIs
    Also provides useful methods for computing accuracy and crossvalidating
    """

    data: Data
    is_fit: bool
    model_name: str
    is_biased = False
    prediction_metrics = ["MAE", "RMSE"]
    top_n_metrics = [
        "Precision",
        "Precision@k",
        "Recall",
        "Recall@k",
        "F1",
        "F1@k",
        "MRR",
        "NDCG",
    ]

    def __init__(self, data: Data, model_name: str):
        self.model_name = model_name
        self.data = data

    @abstractmethod
    def fit(self, **kwargs):
        pass

    @abstractmethod
    def predict(self, u: int, i: int) -> float:
        pass

    @abstractmethod
    def top_n(self, user_index: int, n: int) -> list[int]:
        pass

    def accuracy_top_n(self, n=30, silent=False, cv=False):
        """
        Compute all accuracy metrics using the test set
        """
        if self.is_fit is None:
            raise RuntimeError("Untrain model, invoke fit before predicting")
        n_users = self.data.interactions_train.shape[0]

        metrics = []

        for user_index in tqdm(
            range(n_users),
            leave=False,
            desc="Computing accuracy for top N...",
            disable=silent,
        ):
            user_id = self.data.user_index_to_id[user_index]
            relevant = self.data.get_liked_movies_indices(
                user_id, self.is_biased, "val_cv" if cv else "test"
            )
            if len(relevant) < 2:
                continue
            n_adj = len(relevant) if len(relevant) < n else n
            recommended = [
                self.data.item_id_to_index[x] for x in self.top_n(user_index, n_adj)
            ]
            if len(recommended) < n_adj:
                continue

            prec = precision(relevant, recommended)
            prec_at_k = precision_at_k(relevant, recommended, n_adj)
            rec = recall(relevant, recommended)
            rec_at_k = recall_at_k(relevant, recommended, n_adj)
            rr = reciprocal_rank(relevant, recommended)
            f1 = f1_score(prec, rec)
            f1_at_k = f1_score(prec_at_k, rec_at_k)
            ndcg = normalized_discounted_cumulative_gain(relevant, recommended)
            metrics.append([prec, prec_at_k, rec, rec_at_k, f1, f1_at_k, rr, ndcg])

        return tuple(np.mean(metrics, axis=0))

    @abstractmethod
    def _predict_all(
        self, test_set: coo_array
    ) -> list[tuple[int, int, int, float | None]]:
        pass

    def _compute_prediction_errors(self, loss_function: Callable, cv: bool):
        """
        Given a test set and a loss function, compute the errors
        """
        if not self.is_fit:
            raise RuntimeError("Untrain model, invoke fit before predicting")
        test_set = self.data.interactions_cv_val if cv else self.data.interactions_test
        predictions = self._predict_all(test_set)
        errors = []
        for prediction in predictions:
            _, _, y_true, y_pred = prediction
            if y_pred is not None:
                errors.append(loss_function(y_true - y_pred))
        return errors

    def _accuracy_mae(self, cv=False):
        """
        Compute the Mean Absolute Error of the trained model on a test set
        """
        errors = self._compute_prediction_errors(lambda x: abs(x), cv)
        return np.mean(errors)

    def _accuracy_rmse(self, cv=False):
        """
        Compute the Root Mean Squared Error of the trained model on a test set
        """
        errors = self._compute_prediction_errors(lambda x: x**2, cv)
        return math.sqrt(np.mean(errors))

    def pretty_print_accuracy_top_n(self, n=30):
        print(f"\n{self.model_name} model top N accuracy:")
        accuracy = list(self.accuracy_top_n(n))
        table = tabulate(
            [accuracy],
            headers=self.top_n_metrics,
            tablefmt="grid",
            floatfmt=".4f",
            numalign="center",
            stralign="center",
        )
        print(table)

    def pretty_print_accuracy_predictions(self):
        print(f"\n{self.model_name} model predictions accuracy:")
        accuracy = (self._accuracy_mae(), self._accuracy_rmse())
        table = tabulate(
            [accuracy],
            headers=self.prediction_metrics,
            tablefmt="grid",
            floatfmt=".4f",
            numalign="center",
            stralign="center",
        )
        print(table)

    def pretty_print_cv_results(
        self, kind: Literal["prediction", "top_n"], results: list[dict[str, Any]]
    ):
        # Sort the results according to the task to optimize
        if kind == "prediction":
            headers = self.prediction_metrics + ["Hyperparameters"]
        else:
            headers = self.top_n_metrics + ["Hyperparameters"]

        # Show the best five results
        metric = headers[0] if kind == "prediction" else headers[5]
        table = [(*res["test_metrics"], res["params"]) for res in results[:5]]
        print(f"\n{self.model_name} CV results sorted by {metric}:")
        print(
            tabulate(
                table,
                headers=headers,
                tablefmt="grid",
                floatfmt=".4f",
                numalign="center",
                stralign="center",
            )
        )

    def gridsearch_cv(
        self,
        kind: Literal["prediction", "top_n"],
        params_space: dict[str, list[int | float] | NDArray[np.int64 | np.float64]],
        print_results: bool,
    ):
        """
        Perform the grid search cross validation for the model with the given params space
        """

        # Obtain the combinations of hyperparameters from the params space and the total number of combinations
        combinations, total = generate_combinations(params_space)

        # Split the combinations generator into a generator of batches, where each batch contains 10% of the data
        batch_size = total // 10 + 1
        batches = batch_generator(combinations, batch_size)

        # Compute the model's partial CV file name from the model's name and the hash string of the params space
        cv_path = os.path.join(
            ".joblib",
            f"partial_cv_{self.model_name.replace(' ', '_').lower()}_{dict_to_hash(params_space)}.job",
        )

        # Check if a partial CV exists for the current model
        try:
            if not os.path.exists(".joblib"):
                os.mkdir(".joblib")
            # Load the partial CV and the last processed batch
            resume_batch, results = joblib.load(cv_path)

            # Compute the next batch to process and the number of items already processed
            start = resume_batch + 1
            initial = resume_batch * batch_size

            # Consume the already processed batches from the generator in order to resume
            for _ in range(resume_batch):
                next(batches)
            print(f"Resuming crossvalidation at {resume_batch * 10}%")
        except FileNotFoundError:
            # If no valid partial CV is found, start from scratch
            start = 1
            initial = 0
            results = []

        # Manual controls for updating correctly the CV progress
        with tqdm(
            dynamic_ncols=True,
            desc=f"Crossvalidating {self.model_name}...",
            total=total,
            initial=initial,
        ) as pbar:
            # Custom joblib callback that updates the progress bar
            class MyCallback(joblib.parallel.BatchCompletionCallBack):
                def __call__(self, *args, **kwargs):
                    pbar.update(n=self.batch_size)
                    return super().__call__(*args, **kwargs)

            # Back the old callback up and replace it with the custom one
            old_cb = joblib.parallel.BatchCompletionCallBack
            joblib.parallel.BatchCompletionCallBack = MyCallback

            def do_cv(kind: Literal["prediction", "top_n"], **kwargs):
                """
                Fit the model with the given parameters and return the accuracy metrics
                alongside the param combination
                """
                self.fit(silent=True, cv=True, **kwargs)
                res: dict[str, Any] = {"params": kwargs}
                if kind == "prediction":
                    res["val_metrics"] = (
                        self._accuracy_mae(cv=True),
                        self._accuracy_rmse(cv=True),
                    )
                else:
                    res["val_metrics"] = self.accuracy_top_n(silent=True, cv=True)
                return res

            # Iterate over the batches
            for i, batch in enumerate(batches, start=start):
                # Compute the crossvalidation for the current batch
                for result in Parallel(n_jobs=6, backend="loky")(
                    delayed(do_cv)(kind, **comb) for comb in batch
                ):
                    # Add the results to the global results list
                    if result is None:
                        raise RuntimeError(
                            f"An error occurred while processing batch {i}"
                        )
                    else:
                        results.append(result)
                # Dump the partial CV to file system after the batch has been processed
                joblib.dump((i, results), cv_path)

        # Restore the original joblib callback
        joblib.parallel.BatchCompletionCallBack = old_cb

        # Sort the results according to the task to optimize
        if kind == "prediction":
            results.sort(key=lambda x: (x["val_metrics"][1], x["val_metrics"][0]))
        else:
            results.sort(key=lambda x: x["val_metrics"][5], reverse=True)

        # Only keep best five
        results = results[:5]

        def cv_final_results(kind: Literal["prediction", "top_n"], res: dict[str, Any]):
            params = res["params"]
            self.fit(silent=True, cv=False, **params)
            if kind == "prediction":
                mae = self._accuracy_mae(cv=False)
                rmse = self._accuracy_rmse(cv=False)
                metrics = (mae, rmse)
            elif kind == "top_n":
                metrics = self.accuracy_top_n(silent=True, cv=False)
            res["test_metrics"] = metrics
            return res

        results_final = [
            result
            for result in Parallel(n_jobs=-1, backend="loky")(
                delayed(cv_final_results)(kind, res) for res in results
            )
            if result is not None
        ]

        # Sort the results according to the task to optimize
        if kind == "prediction":
            results.sort(key=lambda x: (x["val_metrics"][1], x["val_metrics"][0]))
        else:
            results.sort(key=lambda x: x["val_metrics"][5], reverse=True)

        if print_results:
            self.pretty_print_cv_results(kind, results_final)

        return results

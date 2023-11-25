from itertools import product
from typing import Literal
from ..CF_Base import CF_Base
import numpy as np
from pyspark import RDD, SparkContext
from utils import RandomSingleton
from scipy.sparse import coo_array
from numpy.typing import NDArray


class ALS_MR(CF_Base):
    """
    Concrete class for Map Reduce Alternating Least Squares recommender system
    """

    spark: SparkContext | None = None

    def fit(self, train_set: coo_array, n_factors=10, epochs=10, reg=0.01):
        print("Fitting the Map Reduce Alternating Least Squares model...")
        # Spark initialization
        if self.spark is None:
            self.spark = SparkContext(
                master="local", appName="Alternating Least Square"
            )
            self.spark.setLogLevel("WARN")

        self.train_set = train_set
        ratings = self.spark.broadcast(self.train_set)
        n_users, n_items = self.train_set.shape

        self.P = RandomSingleton.get_random_normal(
            loc=0, scale=0.1, size=(n_users, n_factors)
        )
        self.Q = RandomSingleton.get_random_normal(
            loc=0, scale=0.1, size=(n_items, n_factors)
        )

        # Creating the RDDs for both user and item factors, alongside u and i indices
        P_RDD: RDD[tuple[NDArray[np.float64], int]] = self.spark.parallelize(
            self.P
        ).zipWithIndex()
        Q_RDD: RDD[tuple[NDArray[np.float64], int]] = self.spark.parallelize(
            self.Q
        ).zipWithIndex()

        for _ in range(epochs):

            def get_observed(
                train_set: coo_array, index: int, kind: Literal["user", "item"]
            ):
                """
                Return indices and actual values of either a user's or an item's observed ratings.
                Needs to be defined in this scope to avoid PySpark pickling errors
                """
                row, col, data = (train_set.row, train_set.col, train_set.data)
                if kind == "user":
                    indices = np.where((row == index))[0]
                    sliced_axis = col[indices]
                else:
                    indices = np.where((col == index))[0]
                    sliced_axis = row[indices]
                sliced_data = data[indices]
                return (sliced_axis, sliced_data)

            def update_factors(
                x: tuple[NDArray[np.float64], int],
                fixed_factor: NDArray,
                dim: Literal["user", "item"],
            ) -> tuple[NDArray[np.float64], int]:
                """
                Map function that computes the current factors using the other fixed factors
                """
                _, idx = x
                observed_indices, observed_values = get_observed(
                    ratings.value, idx, dim
                )
                res = (
                    observed_values @ fixed_factor[observed_indices, :]
                ) @ np.linalg.inv(
                    np.transpose(fixed_factor[observed_indices, :])
                    @ fixed_factor[observed_indices, :]
                    + reg * np.eye(n_factors)
                )
                return res, idx

            # Fix item factors and update user factors
            Q = np.array([x[0] for x in Q_RDD.collect()])
            P_RDD = P_RDD.map(lambda x: update_factors(x, Q, "user"))

            # Fix user factors and update item factors
            P = np.array([x[0] for x in P_RDD.collect()])
            Q_RDD = Q_RDD.map(lambda x: update_factors(x, P, "item"))

        # Collect the final RDD results into the class' factors
        self.P = np.array([x[0] for x in P_RDD.collect()])
        self.Q = np.array([x[0] for x in Q_RDD.collect()])

        # Stop the Spark application
        self.spark = self.spark.stop()

    def cross_validate_hyperparameters(
        self,
        train_set: coo_array,
        test_set: coo_array,
        n_factors_range: list[int],
        epochs_range: list[int],
        reg_range: list[float],
    ):
        """
        Define the hyperparameter ranges required for crossvalidation, compute the product and invoke the super class' method
        """
        prod = product(n_factors_range, epochs_range, reg_range)
        return self._generic_cv_hyper("ALS", train_set, test_set, prod, True)


def cv_hyper_als_mr_helper(train_set: coo_array, test_set: coo_array):
    print("Grid Search Cross Validation for ALS")
    als_mr = ALS_MR()
    n_factors_range = list(np.linspace(start=2, stop=1000, num=100, dtype=int))
    epochs_range = list(np.linspace(start=10, stop=100, num=20, dtype=int))
    reg_range = list(np.linspace(start=0.001, stop=2.0, num=100, dtype=float))
    als_mr.cross_validate_hyperparameters(
        train_set, test_set, n_factors_range, epochs_range, reg_range
    )

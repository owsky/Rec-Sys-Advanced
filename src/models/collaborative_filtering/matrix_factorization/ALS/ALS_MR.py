from typing import Iterable, Literal
from tqdm import tqdm
import pyspark
from data import Data
from ..MF_Base import MF_Base
import numpy as np
from pyspark import RDD, Accumulator, AccumulatorParam, SparkContext, SparkConf
from utils import RandomSingleton
from numpy.typing import NDArray


class ALS_MR(MF_Base):
    """
    Concrete class for Map Reduce Alternating Least Squares recommender system
    """

    def __init__(self, data: Data):
        super().__init__(data, "Alternating Least Squares")

    def fit(self, silent=False, n_factors=10, epochs=10, reg=0.01):
        class DictAccumulator(AccumulatorParam):
            """
            Custom dictionary accumulator, needed to propagate the factors updates across the cluster
            """

            def zero(self, init_value: dict[int, NDArray[np.float64]]):
                """
                Initialize the accumulator with a given dictionary
                """
                return init_value

            def addInPlace(
                self,
                currDict: dict[int, NDArray[np.float64]],
                newDict: dict[int, NDArray[np.float64]],
            ):
                """
                Substitute into the current dictionary all new entries provided by the new dictionary
                """
                for key, value in newDict.items():
                    currDict.update({key: value})
                return currDict

        self.is_fit = True

        # Spark initialization
        conf = (
            SparkConf()
            .setMaster("local")
            .setAppName("Alternating Least Squares")
            .set("spark.log.level", "ERROR")
        )
        spark = SparkContext(conf=conf)
        spark.setLogLevel("ERROR")

        n_users, n_items = self.data.interactions_train.shape

        # Create and cache the ratings RDD
        ratings_RDD: RDD[tuple[int, int, float]] = spark.parallelize(
            list(
                zip(
                    self.data.interactions_train.row,
                    self.data.interactions_train.col,
                    self.data.interactions_train.data,
                )
            )
        ).persist(storageLevel=pyspark.StorageLevel.MEMORY_AND_DISK_DESER)

        # Initialize the factors' dictionaries
        P_shape = (n_users, n_factors)
        P = RandomSingleton.get_random_normal(loc=0, scale=0.1, size=P_shape)
        Q_shape = (n_items, n_factors)
        Q = RandomSingleton.get_random_normal(loc=0, scale=0.1, size=Q_shape)

        P_dict: dict[int, NDArray[np.float64]] = {u: P[u] for u in range(n_users)}
        Q_dict: dict[int, NDArray[np.float64]] = {i: Q[i] for i in range(n_items)}

        P_acc = spark.accumulator(P_dict, DictAccumulator())
        Q_acc = spark.accumulator(Q_dict, DictAccumulator())

        def dictAccToArr(
            dict: Accumulator[dict[int, NDArray[np.float64]]], shape: tuple[int, int]
        ) -> NDArray[np.float64]:
            """
            Helper function that converts accumulators into numpy arrays
            """
            matrix = np.zeros(shape)
            indices, values = zip(*dict.value.items())
            matrix[indices, :] = values
            return matrix

        def compute_factors(
            index: int,
            r_iter: Iterable[tuple[int, int, float]],
            fixed_factor: NDArray[np.float64],
            kind: Literal["user", "item"],
        ):
            """
            Map function that computes the updates for the factors and pushes them onto the accumulator
            """
            nonlocal P_acc, Q_acc
            r = np.array([x[2] for x in r_iter])
            nz = [x[1] for x in r_iter] if kind == "user" else [x[0] for x in r_iter]

            new_factor = np.linalg.solve(
                (fixed_factor[nz, :].T @ fixed_factor[nz, :]) + reg * np.eye(n_factors),
                (fixed_factor[nz, :].T @ r),
            )

            if kind == "user":
                P_acc.add({index: new_factor})
            else:
                Q_acc.add({index: new_factor})

        for _ in tqdm(
            range(epochs),
            desc="Fitting the MR ALS model...",
            leave=False,
            disable=silent,
            dynamic_ncols=True,
        ):
            # Fix items factors and update users factors
            Q = dictAccToArr(Q_acc, Q_shape)
            ratings_RDD.groupBy(lambda x: x[0]).foreach(
                lambda x: compute_factors(x[0], x[1], Q, "user")
            )

            # Fix users factors and update items factors
            P = dictAccToArr(P_acc, P_shape)
            ratings_RDD.groupBy(lambda x: x[1]).foreach(
                lambda x: compute_factors(x[0], x[1], P, "item")
            )

        # Collect the final RDD results into the class' factors
        self.P = dictAccToArr(P_acc, P_shape)
        self.Q = dictAccToArr(Q_acc, Q_shape)

        # Stop the Spark application
        ratings_RDD.unpersist()
        spark.stop()
        return self

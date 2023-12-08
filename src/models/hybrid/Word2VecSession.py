from gensim.models import Word2Vec
from numpy.typing import NDArray
from data import Data
from ..Recommender_System import Recommender_System
import numpy as np


class Word2VecSession(Recommender_System):
    ratings_train: NDArray[np.float64]

    def __init__(self):
        super().__init__("Word2Vec Session")

    def fit(self, data: Data):
        self.data = data
        self.ratings_train = data.train.toarray()
        self.model = self._train_word2vec_model()
        return self

    def _train_word2vec_model(self):
        corpus = []
        for user_ratings in self.ratings_train:
            liked = [str(x) for x in np.flatnonzero(user_ratings >= 3)]
            corpus.append(liked)

        model = Word2Vec(
            sentences=corpus,
            vector_size=100,
            window=10,
            negative=10,
            sg=1,
            hs=0,
            alpha=0.03,
            min_alpha=0.0007,
            min_count=1,
            workers=12,
            seed=14,
        )
        model.train(corpus, epochs=10, total_examples=len(corpus))
        model.init_sims(replace=True)
        return model

    def similar_products(self, v, n=6):
        ms = self.model.wv.similar_by_vector(v, topn=n + 1)[1:]

        new_ms = []
        for j in ms:
            index = int(j[0])
            id = self.data.item_index_to_id[index]
            new_ms.append(id)

        return new_ms

    def aggregate_vectors(self, products):
        product_vec = []
        for i in products:
            try:
                product_vec.append(self.model.wv[i])
            except KeyError:
                continue
        if len(product_vec) == 0:
            return []
        return np.mean(product_vec, axis=0)

    def predict(self):
        return super().predict()

    def top_n(self, user_index: int, n: int) -> list[int] | NDArray[np.int64]:
        user_ratings = self.ratings_train[user_index]
        liked = [str(x) for x in np.flatnonzero(user_ratings >= 3)]
        v = self.aggregate_vectors(liked)
        if len(v) == 0:
            return []
        return self.similar_products(v, n)

    def _predict_all(self) -> list[tuple[int, int, int, float | None]]:
        return super()._predict_all()

    def crossvalidation_hyperparameters(self):
        return super().crossvalidation_hyperparameters()

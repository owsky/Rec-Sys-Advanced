from gensim.models import Word2Vec
from numpy.typing import NDArray
from data import Data
from ..Recommender_System import Recommender_System
import numpy as np


class Word2VecSession(Recommender_System):
    """
    Word2Vec recommender system which learns similarity through user sessions
    """

    def __init__(self, data: Data):
        super().__init__(data, "Word2Vec Session")

    def fit(self, vector_size: int, window: int, biased: bool, silent=False):
        if not silent:
            print("Fitting Word2Vec model")
        self.model = self._train_word2vec_model(vector_size, window, biased)
        self.is_fit = True
        self.is_biased = biased
        return self

    def _train_word2vec_model(self, vector_size: int, window: int, biased: bool):
        n_users = self.data.interactions_train.shape[0]
        corpus = [self._get_str_liked(user_index) for user_index in range(n_users)]

        model = Word2Vec(
            sentences=corpus,
            vector_size=vector_size,
            window=window,
            min_count=1,
            seed=42,
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

    def top_n(self, user_index: int, n: int) -> list[int] | NDArray[np.int64]:
        liked = self._get_str_liked(user_index)
        v = self.aggregate_vectors(liked)
        if len(v) == 0:
            return []
        return self.similar_products(v, n)

    def _get_str_liked(self, user_index):
        user_id = self.data.user_index_to_id[user_index]
        liked = self.data.get_liked_movies_indices(user_id, self.is_biased, "train")
        return [str(x) for x in liked]

    def predict(self):
        raise RuntimeError(f"Model {self.__class__.__name__} cannot predict ratings")

    def _predict_all(self):
        raise RuntimeError(f"Model {self.__class__.__name__} cannot predict ratings")

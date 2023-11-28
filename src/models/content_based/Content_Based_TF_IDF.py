# from joblib import Parallel, delayed
# import numpy as np
# from numpy.typing import NDArray
# import pandas as pd
# from tqdm import tqdm
# from data import Data
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.metrics import ndcg_score, precision_recall_curve, auc


# class Content_Based_TF_IDF:
#     def __init__(self, data: Data):
#         self.data = data
#         self.train = self.data.train.todense()
#         n_users = self.train.shape[0]
#         self.user_profiles = np.zeros((n_users, len(self.data.genres_labels)))

#         self._preprocess_genres()
#         for user_index in range(n_users):
#             self._create_user_profile(user_index)

#     def _preprocess_genres(self):
#         vectorizer = TfidfVectorizer(
#             vocabulary=[label.lower() for label in self.data.genres_labels]
#         )
#         movies_df = self.data.items
#         df_genres = (
#             pd.DataFrame(
#                 {
#                     "Genres": movies_df[self.data.genres_labels].apply(
#                         lambda row: ", ".join(
#                             [
#                                 genre
#                                 for genre, value in zip(self.data.genres_labels, row)
#                                 if value == 1
#                             ]
#                         ),
#                         axis=1,
#                     ),
#                 }
#             )
#             .values.flatten()
#             .tolist()
#         )

#         vectorizer.fit(self.data.genres_labels)
#         self.encoded: NDArray[np.float64] = vectorizer.transform(df_genres).toarray()  # type: ignore

#     def _create_user_profile(self, user_index: int):
#         ratings = self.train[user_index, :]
#         nz_ratings_indices = np.nonzero(ratings != 0)[0]

#         u_prime = []
#         for rating_index in nz_ratings_indices:
#             r = ratings[rating_index]
#             item = self.encoded[rating_index]
#             u_prime.append((r * item).tolist())
#         u_prime = np.array(u_prime).mean(axis=0)
#         self.user_profiles[user_index] = u_prime

#     def top_n(self, user_index: int, n=10):
#         user_profile = self.user_profiles[user_index]
#         ratings = self.train[user_index]
#         unrated_indices = np.nonzero(ratings == 0)[0]

#         similarities = []
#         for movie_index in unrated_indices:
#             movie_profile = self.encoded[movie_index]
#             similarity = cosine_similarity(
#                 user_profile.reshape(1, -1), movie_profile.reshape(1, -1)
#             )
#             similarities.append(similarity.flatten().tolist())

#         similarities = np.array(similarities).flatten()

#         recommendation_indices = np.argsort(similarities)[::-1][:n]
#         return recommendation_indices

#     def _hit_rate(
#         self, recommended_indices: list[int], relevant_items_indices: list[int]
#     ):
#         hits = np.intersect1d(relevant_items_indices, recommended_indices)
#         return len(hits) / len(recommended_indices)

#     def _average_reciprocal_hit_rank(
#         self, recommended_indices: list[int], relevant_items_indices: list[int]
#     ):
#         hits = np.intersect1d(relevant_items_indices, recommended_indices)
#         ranks = [np.where(recommended_indices == hit)[0][0] + 1 for hit in hits]
#         if len(ranks) == 0:
#             return 0
#         return np.mean([1 / rank for rank in ranks])

#     def _ndcg(self, recommended_indices: list[int], relevant_items_indices: list[int]):
#         binary_relevance = [
#             int(idx in relevant_items_indices) for idx in recommended_indices
#         ]
#         ideal_relevance = sorted(binary_relevance, reverse=True)
#         return ndcg_score(np.array([ideal_relevance]), np.array([binary_relevance]))

#     def accuracy_metrics(self):
#         n_users = self.data.test.shape[0]
#         test = self.data.test.todense()

#         def aux(user_index: int):
#             recommended_indices = self.top_n(user_index, 20).tolist()
#             user_bias = self.data.average_user_rating[user_index]
#             relevant_items_indices = np.nonzero(test[user_index, :] - user_bias > 0)[
#                 0
#             ].tolist()

#             if len(recommended_indices) < 20 or len(relevant_items_indices) < 50:
#                 return None

#             precision = len(
#                 np.intersect1d(relevant_items_indices, recommended_indices)
#             ) / len(recommended_indices)
#             recall = len(
#                 np.intersect1d(relevant_items_indices, recommended_indices)
#             ) / len(relevant_items_indices)
#             hit_rate = self._hit_rate(recommended_indices, relevant_items_indices)
#             arhr = self._average_reciprocal_hit_rank(
#                 recommended_indices, relevant_items_indices
#             )

#             ndcg = self._ndcg(recommended_indices, relevant_items_indices)

#             return precision, recall, hit_rate, arhr, ndcg

#         results = [
#             result
#             for result in Parallel(n_jobs=-1, backend="loky")(
#                 delayed(aux)(user_index)
#                 for user_index in tqdm(
#                     range(n_users), desc="Computing accuracy metrics"
#                 )
#             )
#             if result is not None
#         ]
#         precision, recall, hit_rate, arhr, ndcg = zip(*results)

#         return (
#             np.mean(precision),
#             np.mean(recall),
#             np.mean(hit_rate),
#             np.mean(arhr),
#             np.mean(ndcg),
#         )

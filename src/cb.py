from data import Data
from models import Content_Based


def cb(data: Data):
    model = Content_Based()
    model.fit(data)
    user_id = data.id_to_index(13, "user")
    print(model.get_top_n_recommendations(user_id, 10))

from data import Data
from models import Content_Based, Content_Based_TF_IDF


def cb(data: Data):
    m1 = Content_Based()
    m1.fit(data)
    print(m1.accuracy_metrics())

    model2 = Content_Based_TF_IDF(data)
    print(model2.accuracy_metrics())

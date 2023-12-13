import math
from pandas import Timestamp


def exponential_decay(
    timestamp: Timestamp, base_time: Timestamp, decay_constant=0.001
) -> float:
    age = (base_time - timestamp).days
    weight = math.exp(-decay_constant * age)
    return weight

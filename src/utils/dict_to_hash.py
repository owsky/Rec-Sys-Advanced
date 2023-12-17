import hashlib
import json
import numpy as np
from numpy.typing import NDArray


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def dict_to_hash(
    data_dict: dict[str, list[int | float] | NDArray[np.int64 | np.float64]]
):
    json_string = json.dumps(data_dict, cls=NumpyEncoder, sort_keys=True)
    hash_object = hashlib.sha256(json_string.encode())
    hash_value = hash_object.hexdigest()
    return hash_value

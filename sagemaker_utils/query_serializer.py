from  scipy.sparse import lil_matrix
import scipy.sparse
import json
import scipy.sparse
CONTENT_TYPE = 'application/json'
nFeatures = 0

def to_features(cur_keys, cur_values):
    cur_feature = {}
    cur_feature["keys"] = cur_keys
    cur_feature["values"] = cur_values
    cur_feature["shape"] = [nFeatures]
    cur_data = {}
    cur_data["features"] = cur_feature
    cur_instance = {}
    cur_instance["data"] = cur_data
    return cur_instance

def to_sparse_vectors(X_sparse):
    cx = scipy.sparse.coo_matrix(X_sparse)
    last_i = -1
    cur_keys = []
    cur_value = []
    instances = []
    for i, j, v in zip(cx.row, cx.col, cx.data):
        if i != last_i and last_i != -1:
            cur_instance = to_features(cur_keys, cur_value)
            instances.append(cur_instance)
            cur_keys = []
            cur_value = []
        cur_keys.append(int(j))
        cur_value.append(int(v))
        last_i = i
    cur_instance = to_features(cur_keys, cur_value)
    instances.append(cur_instance)
    return instances

def serialize(data):
    instances = to_sparse_vectors(data)
    js = {}
    js['instances'] = instances
    return json.dumps(js)

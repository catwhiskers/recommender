from  scipy.sparse import lil_matrix
from sagemaker.serializers import BaseSerializer
import scipy.sparse
import json

class SparseFormatSerializer(BaseSerializer):
    def __init__(self, nFeatures):
        self.nFeatures = nFeatures

    CONTENT_TYPE = 'application/json'

    import scipy.sparse
    def to_features(self, cur_keys, cur_values, nFeatures):
        cur_feature = {}
        cur_feature["keys"] = cur_keys
        cur_feature["values"] = cur_values
        cur_feature["shape"] = [nFeatures]
        cur_data = {}
        cur_data["features"] = cur_feature
        cur_instance = {}
        cur_instance["data"] = cur_data
        return cur_instance

    def to_sparse_vectors(self, X_sparse, nFeatures):
        cx = scipy.sparse.coo_matrix(X_sparse)
        last_i = -1
        cur_keys = []
        cur_value = []
        instances = []
        for i, j, v in zip(cx.row, cx.col, cx.data):
            if i != last_i and last_i != -1:
                cur_instance = self.to_features(cur_keys, cur_value, nFeatures)
                instances.append(cur_instance)
                cur_keys = []
                cur_value = []

            cur_keys.append(int(j))
            cur_value.append(int(v))
            last_i = i
        cur_instance = self.to_features(cur_keys, cur_value, nFeatures)
        instances.append(cur_instance)
        return instances

    def serialize(self, data):
        instances = self.to_sparse_vectors(data, self.nFeatures)
        js = {}
        js['instances'] = instances
        return json.dumps(js)
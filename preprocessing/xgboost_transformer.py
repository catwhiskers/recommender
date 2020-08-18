from preprocessing.transformer import Transformer
import copy
import numpy
from  scipy.sparse import lil_matrix

class XGBoostTransformer(Transformer):
    def __init__(self, user_data_dict, item_data_dict, user_item_rating):
        self.get_data_information(user_data_dict, item_data_dict, user_item_rating)


    def to_sparse(self, X_arr, feature_length):
        X = lil_matrix((len(X_arr), feature_length)).astype('float32')
        for i, x in enumerate(X_arr):
            for e in x:
                X[i, e[0]] = e[1]
        return X

    def get_feature_vectors(self, user_data_dict, item_data_dict, user_item_rating):
        feature_length = self.get_feature_length(include_user_item_identify=False)
        X, Y, X_item_cold, Y_item_cold, feature_length \
            = self.get_raw_vectors(user_data_dict, item_data_dict, user_item_rating, include_user_item_identify=False)
        return X, Y, X_item_cold, Y_item_cold, feature_length




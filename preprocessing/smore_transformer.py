from preprocessing.transformer import Transformer
import numpy
from  scipy.sparse import lil_matrix

class SmoreDataTransformer(Transformer):
    def __init__(self, user_data_dict, item_data_dict, user_item_rating):
        self.get_data_information(user_data_dict, item_data_dict, user_item_rating)


    def get_feature_vectors(self, user_data_dict, item_data_dict, user_item_rating):
        feature_length = self.get_feature_length(include_user_item_identify=True)
        X, Y, X_item_cold, Y_item_cold, feature_length \
            = self.get_raw_vectors(user_data_dict, item_data_dict, user_item_rating, include_user_item_identify=True)
        return X, Y, X_item_cold, Y_item_cold, feature_length


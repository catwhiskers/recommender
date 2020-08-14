from preprocessing.transformer import Transformer
import numpy
from  scipy.sparse import lil_matrix

class SmoreTransformer(Transformer):
    def __init__(self, user_data_dict, item_data_dict, user_item_rating):
        users = set()
        items = set()
        for i, info in enumerate(user_item_rating):
            users.add(info[0])
            items.add(info[1])

        # build ordering index for users and items
        uks = sorted(list(users))
        self.u_idx = {}
        for i, u in enumerate(uks):
            self.u_idx[u] = i
        self.i_idx = {}
        iks = sorted(list(items))
        for i, item in enumerate(iks):
            self.i_idx[item] = i

        # get length of user feature vector
        self.len_uinfo = 0
        if len(user_data_dict.values()) > 0:
            self.len_uinfo = len(list(user_data_dict.values())[0])

        # get length of item feature vector
        self.len_iinfo = 0
        if len(item_data_dict.values()) > 0:
            self.len_iinfo = len(list(item_data_dict.values())[0])



    def get_feature_vectors(self, user_data_dict, item_data_dict, user_item_rating):
        # total feature length
        feature_length =  self.len_uinfo + self.len_iinfo

        # user and item exist in training data
        Y = []
        X_arr = []

        # item cold start data
        Y_item_cold = []
        X_item_cold_arr = []

        i = 0
        for info in user_item_rating:
            uid = info[0]
            iid = info[1]
            rating = info[2]

            cur_feature = []
            if uid in self.u_idx:
                cur_feature.append((self.u_idx[uid], 1))
            if iid in self.i_idx:
                cur_feature.append((self.i_idx[iid] + self.len_uinfo, 1))

             # user and item exist in training data
            if uid in self.u_idx and iid in self.i_idx:
                X_arr.append(cur_feature)
                Y.append(rating)
            elif uid in self.u_idx and iid not in self.i_idx:
                X_item_cold_arr.append(cur_feature)
                Y_item_cold.append(rating)
        X = X_arr
        X_item_cold = X_item_cold_arr
        Y = numpy.asarray(Y, dtype=numpy.float32)
        Y_item_cold = numpy.asarray(Y_item_cold, dtype=numpy.float32)
        return X, Y, X_item_cold, Y_item_cold, feature_length

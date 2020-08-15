from preprocessing.transformer import Transformer
import numpy
from  scipy.sparse import lil_matrix

class SmoreTransformer(Transformer):
    def __init__(self, user_data_dict, item_data_dict, user_item_rating, u_idx, i_idx):
        self.train_users = set()
        self.train_items = set()
        for i, info in enumerate(user_item_rating):
            self.train_users.add(info[0])
            self.train_items.add(info[1])
        self.u_idx = u_idx
        self.i_idx = i_idx


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
            uinfo = [0] * self.len_uinfo
            iinfo = [0] * self.len_iinfo
            if uid in user_data_dict:
                uinfo = user_data_dict[uid]
            if iid in item_data_dict:
                iinfo = item_data_dict[iid]
            cur_feature = []
            if uid in self.u_idx:
                cur_feature.append((self.u_idx[uid], 1))
            if iid in self.i_idx:
                cur_feature.append((self.i_idx[iid] + self.len_uinfo, 1))

            for j, info_e in enumerate(uinfo + iinfo):
                if info_e > 0:
                    cur_feature.append((j + self.len_uinfo + self.len_iinfo, info_e))

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

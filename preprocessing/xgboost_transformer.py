from preprocessing.transformer import Transformer
import copy
import numpy
from  scipy.sparse import lil_matrix

class XGBoostTransformer(Transformer):
    def __init__(self, user_data_dict, item_data_dict, user_item_rating, u_idx, i_idx):
        self.train_users = set()
        self.train_items = set()
        for ui_info in user_item_rating:
            self.train_users.add(ui_info[0])
            self.train_items.add(ui_info[1])

        self.u_idx = u_idx
        self.i_idx = i_idx
        self.user_nb = len(self.u_idx)
        self.item_nb = len(self.i_idx)

        # get length of user feature vector
        self.len_uinfo = 0
        if len(user_data_dict.values()) > 0:
            self.len_uinfo = len(list(user_data_dict.values())[0])

        # get length of item feature vector
        self.len_iinfo = 0
        if len(item_data_dict.values()) > 0:
            self.len_iinfo = len(list(item_data_dict.values())[0])


    def to_sparse(self, X_arr, feature_length):
        X = lil_matrix((len(X_arr), feature_length)).astype('float32')
        for i, x in enumerate(X_arr):
            for e in x:
                X[i, e[0]] = e[1]
        return X

    def get_feature_vectors(self, user_data_dict, item_data_dict, user_item_rating):
        # total feature length
        feature_length = len(self.u_idx) + len(self.i_idx) + self.len_uinfo + self.len_iinfo

        # user and item exist in training data
        Y = []
        X_arr = []

        # item cold start data
        Y_item_cold = []
        X_item_cold_arr = []
        i_pop = {}
        # item popularity data
        for info in user_item_rating:
            iid = info[1]
            if iid not in i_pop:
                i_pop[iid] = 0
            rating = info[2]
            if rating > 0:
                i_pop[iid] += 1

        i = 0
        for info in user_item_rating:
            uid = info[0]
            iid = info[1]
            rating = info[2]
            uinfo = [0] * self.len_uinfo
            iinfo = [0] * self.len_iinfo
            if uid in user_data_dict:
                uinfo = user_data_dict[uid].tolist()
            if iid in item_data_dict:
                iinfo = item_data_dict[iid].tolist()
                if len(iinfo) == self.len_iinfo:
                    iinfo.append(i_pop[iid])
            cur_feature = []
            for j, info_e in enumerate(uinfo + iinfo):
                if info_e > 0:
                    cur_feature.append((j, info_e))

            # user and item exist in training data
            if uid in self.train_users and iid in self.train_items:
                X_arr.append(cur_feature)
                Y.append(rating)
            elif uid in self.train_users and iid not in self.train_items:
                X_item_cold_arr.append(cur_feature)
                Y_item_cold.append(rating)

        # X_item_cold = self.to_sparse(X_item_cold_arr, feature_length)
        Y = numpy.asarray(Y, dtype=numpy.float32)
        Y_item_cold = numpy.asarray(Y_item_cold, dtype=numpy.float32)
        return X_arr, Y, X_item_cold_arr, Y_item_cold, feature_length




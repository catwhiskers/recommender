from preprocessing.transformer import Transformer
import copy
import numpy
from  scipy.sparse import lil_matrix

class FactorizationMachineTransformer(Transformer):
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
        feature_length = len(self.u_idx) + len(self.i_idx) + self.len_uinfo + self.len_iinfo
        Y = []
        X = lil_matrix((len(user_item_rating), feature_length)).astype('float32')
        processed = []

        for i, info in enumerate(user_item_rating):
            uid = info[0]
            iid = info[1]
            if uid not in self.u_idx or iid not in self.i_idx:
                continue
            rating = info[2]
            uinfo = [0]*self.len_uinfo
            iinfo = [0]*self.len_iinfo
            if uid in user_data_dict:
                uinfo = user_data_dict[uid]
            if iid in item_data_dict:
                iinfo = item_data_dict[iid]
            X[i,self.u_idx[uid]] = 1
            X[i,self.i_idx[iid]+self.len_uinfo] = 1
            for j, info_e in enumerate(uinfo+iinfo):
                if info_e > 0:
                    X[i, j+self.len_uinfo + self.len_iinfo] = info_e
            Y.append(rating)
            processed.append([uid, iid])
        Y = numpy.asarray(Y, dtype=numpy.float32)
        return X, Y, feature_length, processed




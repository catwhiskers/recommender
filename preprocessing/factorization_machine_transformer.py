from preprocessing.transformer import Transformer
import copy
import numpy
from  scipy.sparse import lil_matrix

class FactorizationMachineTransformer(Transformer):
    def get_feature_vectors(self, user_data_dict, item_data_dict, user_item_rating):
        uks = sorted(list(user_data_dict.keys()))
        u_idx = {}
        for i, u in enumerate(uks):
            u_idx[u] = i

        i_idx = {}
        iks = sorted(list(item_data_dict.keys()))
        for i, item in enumerate(iks):
            i_idx[item] = i



        Y = []
        len_uinfo = len(list(user_data_dict.values())[0])
        len_iinfo = len(list(item_data_dict.values())[0])
        feature_length = len(u_idx) + len(i_idx) + len_uinfo + len_iinfo
        X = lil_matrix((len(user_item_rating), feature_length)).astype('float32')
        for i, info in enumerate(user_item_rating):
            uid = info[0]
            iid = info[1]
            rating = info[2]
            uinfo = user_data_dict[uid]
            iinfo = item_data_dict[iid]
            X[i,u_idx[uid]] = 1
            X[i,i_idx[iid]+len_uinfo] = 1
            for j, info_e in enumerate(uinfo+iinfo):
                if info_e > 0:
                    X[i, j+len_uinfo + len_iinfo] = info_e

            Y.append(rating)
        Y = numpy.asarray(Y, dtype=numpy.float32)
        return X, Y, feature_length


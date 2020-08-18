from abc import ABC, abstractmethod
import numpy

class Transformer(ABC):
    @abstractmethod
    def get_feature_vectors(self):
        pass
    def get_data_information(self, user_data_dict, item_data_dict, user_item_rating):
        self.train_users = set()
        self.train_items = set()
        for ui_info in user_item_rating:
            self.train_users.add(ui_info[0])
            self.train_items.add(ui_info[1])

        self.u_idx = {}
        for i, uid in enumerate(sorted(list(self.train_users))):
            self.u_idx[uid] = i
        self.user_nb = len(self.u_idx)
        self.i_idx = {}
        for i, iid in enumerate(sorted(list(self.train_items))):
            self.i_idx[iid] = i
        self.item_nb = len(self.i_idx)

        # get length of user feature vector
        self.len_uinfo = 0
        if len(user_data_dict.values()) > 0:
            self.len_uinfo = len(list(user_data_dict.values())[0])

        # get length of item feature vector
        self.len_iinfo = 0
        if len(item_data_dict.values()) > 0:
            self.len_iinfo = len(list(item_data_dict.values())[0])

    def get_feature_length(self, include_user_item_identify=True):
        feature_length =  self.len_uinfo + self.len_iinfo
        if include_user_item_identify:
            feature_length += (self.user_nb + self.item_nb)
        return feature_length

    def get_raw_vectors(self, user_data_dict, item_data_dict, user_item_rating, include_user_item_identify=True):
        feature_length = self.get_feature_length(include_user_item_identify)
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
            if include_user_item_identify:
                if uid in self.u_idx:
                    cur_feature.append((self.u_idx[uid], 1))
                if iid in self.i_idx:
                    cur_feature.append((self.i_idx[iid] + self.user_nb, 1))

            setoff = 0
            if include_user_item_identify:
                setoff = self.user_nb + self.item_nb
            for j, info_e in enumerate(uinfo + iinfo):
                if abs(info_e) > 0.0001:
                    cur_feature.append((j + setoff, info_e))

            # user and item exist in training data
            if uid in self.train_users and iid in self.train_items:
                X_arr.append(cur_feature)
                Y.append(rating)
            elif uid in self.train_users and iid not in self.train_items:
                X_item_cold_arr.append(cur_feature)
                Y_item_cold.append(rating)

        Y = numpy.asarray(Y, dtype=numpy.float32)
        Y_item_cold = numpy.asarray(Y_item_cold, dtype=numpy.float32)
        return X_arr, Y, X_item_cold_arr, Y_item_cold, feature_length

import re
from preprocessing.datareader import AbstractDataReader

class SmoreDataReader(AbstractDataReader):
    def __init__(self, u_idx, i_idx, file_path):
        self.user_idx = u_idx
        self.item_idx = i_idx
        nb_users = len(u_idx)

        self.user_inv_idx = {}
        self.item_inv_idx = {}

        self.user_vectors = {}
        self.item_vectors = {}
        for k, v in u_idx.items():
            self.user_inv_idx[v] = k

        for k, v in i_idx.items():
            self.item_inv_idx[v] = k

        f = open(file_path, 'r')
        lidx = 0
        dim = 0
        for l in f.readlines():
            toks = l.split(' ')
            if lidx == 0:
                self.ui_nb = int(toks[0])
                self.dim = int(toks[1])
            else:
                idx = int(toks[0])
                array = []
                for j in range(1, len(toks)):
                    array.append(float(toks[j]))
                if idx < nb_users:
                    self.user_vectors[self.user_inv_idx[idx]] = array
                else:
                    self.item_vectors[self.item_inv_idx[idx - nb_users]] = array
            lidx += 1


    def read_user_data(self):
        return self.user_vectors

    def read_item_data(self):
        return self.item_vectors

    def read_dim(self):
        return self.dim

    def read_node_number(self):
        return self.ui_nb

    def read_user_item_rating(self):
        pass


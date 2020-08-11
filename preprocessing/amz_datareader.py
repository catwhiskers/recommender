import re
from preprocessing.datareader import AbstractDataReader
# import json
import pandas

class AMZDataReader(AbstractDataReader):

    def read_user_data(self, file_path:str):
        return {}


    def read_item_data(self, file_path:str):
        return {}


    def read_user_item_rating(self, file_path: str):
        df =  pandas.read_csv(file_path)
        uir = []
        for index, row in df.iterrows():
            iid = row['product_id']
            uid = row['customer_id']
            score = float(row['star_rating'])
            rtime = row['review_date']
            uir.append([uid, iid, score, rtime])
        uir = sorted(uir, key=lambda x: x[3])
        return uir

class AMZComprehendDataReader(AMZDataReader):
    def read_item_data(self, file_path:str):
        f = open(file_path, 'r')
        topic_num = 0
        idx = 0
        item_dict = {}
        result = {}
        for l in f.readlines():
            if idx == 0:
                topic_num = int(l)
            else:
                toks = l.split(',')
                iid = toks[0]
                topic = int(toks[1])
                prob = float(toks[2])
                if iid not in item_dict:
                    item_dict[iid] = []
                item_dict[iid].append((topic, prob))
            idx += 1
        for k, v in item_dict.items():
            arr = [0]*topic_num
            for tinfo in v:
                arr[tinfo[0]] = tinfo[1]
            result[k] = arr
        return   result








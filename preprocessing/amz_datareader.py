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


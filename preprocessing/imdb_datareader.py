import re
from preprocessing.datareader import AbstractDataReader

class IMDBDataReader(AbstractDataReader):
    def read_user_data(self, file_path:str):
        user_file = open(file_path, encoding = "ISO-8859-1", mode='r')
        dict = {}
        occ_dict = {}
        for l in user_file.readlines():
            toks = l.split('|')
            uid = toks[0]
            age = int(toks[1])
            gender = toks[2]
            gi = [1, 0]
            if gender == 'F':
                gi = [0, 1]
            occupation = toks[3]
            if occupation not in occ_dict:
                occ_dict[occupation] = len(occ_dict)
            oi = occ_dict[occupation]
            features_raw = [age]
            features_raw += gi
            features_raw.append(oi)
            dict[uid] = features_raw

        for k, v in dict.items():
            oid = v[-1]
            arr = [0]*len(occ_dict)
            arr[oid] = 1
            nv = v[:-1]+arr
            dict[k] = nv
        return dict

    def read_item_data(self, file_path:str):
        item_file = open(file_path, encoding = "ISO-8859-1", mode='r')
        dict = {}
        movie_year_p = re.compile('.*\((\d+)\)')

        for l in item_file.readlines():
            toks = l.split('|')
            iid = toks[0]
            title = toks[1]
            genres = toks[5:]
            m = re.search(movie_year_p, title)
            movie_year = -1
            try:
                movie_year = int(m.group(1))
            except:
                pass
            features = [movie_year]
            for g in genres:
                features.append(int(g.strip()))
            dict[iid] = features
        return dict

    def read_user_item_rating(self, file_path: str):
        rating_file = open(file_path, encoding = "ISO-8859-1", mode='r')
        uir = []
        for l in rating_file.readlines():
            toks = l.split('\t')
            uid = toks[0]
            iid = toks[1]
            rating = int(toks[2])
            rtime = int(toks[3].strip())
            score = 0
            if rating > 3:
                score = 1
            uir.append([uid, iid, score, rtime])
        uir = sorted(uir, key=lambda x: x[3])
        return uir


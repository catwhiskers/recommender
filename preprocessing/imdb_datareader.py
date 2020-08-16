import re
from preprocessing.datareader import AbstractDataReader
import pandas as pd


class PopularityReader(AbstractDataReader):
    def read_item_data(self, user_item_rating):
        i_pop = {}
        # item popularity data
        for info in user_item_rating:
            iid = info[1]
            if iid not in i_pop:
                i_pop[iid] = 0
            rating = info[2]
            if rating > 0:
                i_pop[iid] += 1
        return i_pop

    def read_user_data(self, file:str):
        pass

    def read_user_item_rating(self, file:str):
        pass


class IMDBDataReader(AbstractDataReader):
    def read_user_data(self, file_path:str):
        feature_dict = {}
        user_df = pd.read_csv(file_path, names=['uid', 'age', 'gender', 'occupation', 'zipcode'], sep='|')
        gender = pd.get_dummies(user_df.gender, prefix='gender')
        occupation = pd.get_dummies(user_df.occupation, prefix='occup')
        user_df = pd.concat([user_df, gender], axis=1)
        user_df = pd.concat([user_df, occupation], axis=1)
        user_df = user_df.drop(['gender', 'occupation', 'zipcode'], axis=1)
        user_data = user_df.values.tolist()

        for uinfo in user_data:
            uid = str(uinfo[0])
            features = uinfo[1:]
            feature_dict[uid] = features

        return feature_dict

    def read_item_data(self, file_path:str):
        genres = ['unknown', 'Action', 'Adventure', 'Animation', 'Childrens', 'Comedy', 'Crime', \
                  'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', \
                  'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

        item_df = pd.read_csv(file_path,
                              names=['iid', 'title', 'release_date', 'video_release_date', 'imdb url'] + genres,
                              sep='|', encoding="ISO-8859-1")


        import re
        def get_year(title):
            movie_year_p = re.compile('.*\((\d+)\)')
            m = re.search(movie_year_p, title)
            movie_year = -1
            try:
                movie_year = int(m.group(1))
            except:
                pass
            return movie_year

        item_df['year'] = item_df.apply(lambda x: get_year(x['title']), axis=1)
        item_df = item_df.drop(['title', 'release_date', 'video_release_date', 'imdb url'], axis=1)
        item_data = item_df.values.tolist()

        feature_dict = {}
        for item_info in item_data:
            iid = str(item_info[0])
            i_feature_vector = item_info[1:]
            feature_dict[iid] = i_feature_vector
        return feature_dict

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


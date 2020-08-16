from preprocessing.imdb_datareader import IMDBDataReader
from preprocessing.imdb_datareader import PopularityReader
from preprocessing.xgboost_transformer import XGBoostTransformer
from preprocessing.factorization_machine_transformer import FactorizationMachineTransformer
from preprocessing.smore_transformer import SmoreDataTransformer
user_path = '/Users/yianc/Downloads/ml-100k/u.user'
item_path = '/Users/yianc/Downloads/ml-100k/u.item'
user_item = '/Users/yianc/Downloads/ml-100k/u.data'
reader = IMDBDataReader()
user_item  = reader.read_user_item_rating(user_item)

users  = reader.read_user_data(user_path)
items = reader.read_item_data(item_path)

popreader = PopularityReader()
pop_info = popreader.read_item_data(user_item)
print('item feature length', len(list(items.values())[0]))
for k, v in items.items():
    if k in pop_info:
        v.append(pop_info[k])
    else:
        v.append(0)
# users = {}
# items = {}


train_user_item = user_item[:int(len(user_item)*0.8)]
# print(train_user_item[:10])
test_user_item = user_item[int(len(user_item)*0.8):]

print('item feature length', len(list(items.values())[0]))
# print(test_user_item[:10])
transformer = XGBoostTransformer(users, items, train_user_item)
print('item feature length', len(list(items.values())[0]))

X1, Y1, X1c, Y1c, feature_len = transformer.get_feature_vectors(users, items, test_user_item)
print("xgboost")
print("u_idx")
print(transformer.u_idx)
print("i_idx")
print(transformer.i_idx)
print(test_user_item[:10])
print(transformer.user_nb)
print(transformer.item_nb)
print(transformer.len_uinfo)
print(transformer.len_iinfo)
print(X1.shape)
print(X1[:10])

#
transformerfm = FactorizationMachineTransformer(users, items, train_user_item)
X1, Y1, X1c, Y1c, feature_len = transformerfm.get_feature_vectors(users, items, test_user_item)

# print(X1c[0:100])

transformerfm = SmoreDataTransformer({}, {}, train_user_item)
X1, Y1, X1c, Y1c, feature_len = transformerfm.get_feature_vectors({}, {}, test_user_item)
print("smore")
print("u_idx")
print(transformer.u_idx)
print("i_idx")
print(transformer.i_idx)
print(test_user_item[:10])
print(transformer.user_nb)
print(transformer.item_nb)
print(transformer.len_uinfo)
print(transformer.len_iinfo)
print(X1[:10])

# # print(X1[:10])
# # print(len(Y1))
# # X2, Y2, X2c, Y2x, feature_len = transformer.get_feature_vectors(users, items, test_user_item)
# # print(X2[:10])
# # print(len(Y2))
#
# import sagemaker_utils
# from sagemaker_utils.query_serializer import serialize as fmserialize
#
# sagemaker_utils.query_serializer.nFeatures = feature_len
# result = fmserialize(X1[:10])
# print(result)
#
# file = open('algorithms/rep.txt', 'r')
from preprocessing.smore_datareader import SmoreDataReader
reader = SmoreDataReader(transformer.u_idx, transformer.i_idx, 'rep_dw.txt')
user_data = reader.read_user_data()
print(user_data)
#
#
#
# # user_item = '/Users/yianc/Downloads/amz-review-apparel.csv'
# # reader = AMZDataReader()
# # user_item  = reader.read_user_item_rating(user_item)
# # print('finish reading user_item')
# # print(user_item)
# # items = reader.read_item_data(user_item)
# # users = reader.read_item_data(user_item)
# # transformer = FactorizationMachineTransformer()
# # X_train, X_test, Y_train, Y_test, feature_len = transformer.get_feature_vectors(users, items, user_item)
# # print('finish')
# # print(X_train)
# # print(len(Y_train))
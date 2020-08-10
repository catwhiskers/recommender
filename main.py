from preprocessing.amz_datareader import AMZDataReader
from preprocessing.imdb_datareader import IMDBDataReader
from preprocessing.factorization_machine_transformer import  FactorizationMachineTransformer
user_path = '/Users/yianc/Downloads/ml-100k/u.user'
item_path = '/Users/yianc/Downloads/ml-100k/u.item'
user_item = '/Users/yianc/Downloads/ml-100k/u.data'
reader = IMDBDataReader()
user_item  = reader.read_user_item_rating(user_item)
print(user_item)
users = reader.read_user_data(user_path)
items = reader.read_item_data(item_path)

train_user_item = user_item[:int(len(user_item)*0.8)]
print(train_user_item[:10])
test_user_item = user_item[int(len(user_item)*0.8):]
print(test_user_item[:10])
transformer = FactorizationMachineTransformer(users, items, train_user_item)
X1, Y1, feature_len, abandoned = transformer.get_feature_vectors(users, items, train_user_item)
print(X1[:10])
print(len(Y1))
X2, Y2, feature_len, abandoned = transformer.get_feature_vectors(users, items, test_user_item)
print(X2[:10])
print(len(Y2))



# user_item = '/Users/yianc/Downloads/amz-review-apparel.csv'
# reader = AMZDataReader()
# user_item  = reader.read_user_item_rating(user_item)
# print('finish reading user_item')
# print(user_item)
# items = reader.read_item_data(user_item)
# users = reader.read_item_data(user_item)
# transformer = FactorizationMachineTransformer()
# X_train, X_test, Y_train, Y_test, feature_len = transformer.get_feature_vectors(users, items, user_item)
# print('finish')
# print(X_train)
# print(len(Y_train))
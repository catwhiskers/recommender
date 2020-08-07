from preprocessing.imdb_datareader import IMDBDataReader
from preprocessing.factorization_machine_transformer import  FactorizationMachineTransformer
user_path = '/Users/yianc/Downloads/ml-100k/u.user'
item_path = '/Users/yianc/Downloads/ml-100k/u.item'
user_item = '/Users/yianc/Downloads/ml-100k/u.data'
reader = IMDBDataReader()
user_item  = reader.read_user_item_rating(user_item)
users = reader.read_user_data(user_path)
items = reader.read_item_data(item_path)
transformer = FactorizationMachineTransformer()
X, Y = transformer.get_feature_vectors(users, items, user_item)
print(X)
print(len(Y))
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd
import random
import pickle
path = './'
random.seed(2022)
movies_genres = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama',
                 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-F', 'Thriller',
                 'War', 'Western']
movies_col = ['GroupID', 'MovieID', 'Title', 'Year'] + movies_genres
user_col = ['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code']
rating_col = ['UserID', 'MovieID', 'Rating', 'Timestamp']
final_col = ['UserID', 'GroupID', 'MovieID', 'Rating'] + ['Gender', 'Age', 'Occupation', 'Zip-code'] + ['Year'] + movies_genres
movie_feature = pd.read_csv('{}adsFeature.csv'.format(path + 'movie_1m/'))
movie_feature = movie_feature.fillna(0)
user_data = list()
with open('{}users.dat'.format(path + 'movie_1m/')) as f:
    for i, record in enumerate(f):
        user_dict = {}
        record = record.strip().split('::')
        user_dict = dict(zip(user_col, record))
        user_dict['Gender'] = 0 if user_dict['Gender'] == 'M' else 1
        user_data.append(user_dict)
user_feature = pd.DataFrame(user_data, columns=user_col)
user_feature.UserID = user_feature.UserID.astype('float64')
user_feature.to_csv('{}userFeature.csv'.format(path + 'movie_1m/'), index=False)
user_feature = pd.read_csv('{}userFeature.csv'.format(path + 'movie_1m/'), low_memory=False)
rating_data = list()
with open('{}ratings.dat'.format(path + 'movie_1m/')) as f:
    for i, record in enumerate(f):
        user_dict = {}
        record = record.strip().split('::')
        user_dict = dict(zip(rating_col, record))
        user_dict['Rating'] = 1 if int(user_dict["Rating"]) > 3 else 0
        rating_data.append(user_dict)
rating_record = pd.DataFrame(rating_data, columns=rating_col)
rating_record = rating_record.astype('float64')
rating_record.to_csv('{}ratingRecord.csv'.format(path + 'movie_1m/'), index=False)
rating_record = pd.read_csv('{}ratingRecord.csv'.format(path + 'movie_1m/'))
all_data = pd.merge(rating_record, user_feature, on='UserID', how='left')
all_data = pd.merge(all_data, movie_feature, on='MovieID', how='left')
all_data.to_csv('{}all_data.csv'.format(path + 'movie_1m/'), index=False)
all_data = pd.read_csv('{}all_data.csv'.format(path + 'movie_1m/'))
columns = ['Year'] + movies_genres + ['Gender', 'Age', 'Occupation', 'Zip-code']
all_data['Zip-code'] = all_data['Zip-code'].astype(str)
for feature in columns:
    try:
        all_data[feature] = LabelEncoder().fit_transform(all_data[feature].apply(int))
    except:
        all_data[feature] = LabelEncoder().fit_transform(all_data[feature])
for feature in columns:
    print('{}: {},'.format(feature, max(all_data[feature]) + 1))
MovieIDSet = set(all_data.MovieID)
train_data = None
for MovieID in MovieIDSet:
    subData = all_data[all_data.MovieID == MovieID]
    lenth = subData.shape[0]
    sub = subData.sort_values(by='Timestamp', axis=0, ascending=True, inplace=False)
    sub = sub.iloc[:int(lenth * 0.8)]
    train_data = pd.concat([train_data, sub])
test_data = pd.concat([all_data, train_data, train_data]).drop_duplicates(keep=False)
train_data = pd.concat([all_data, test_data, test_data]).drop_duplicates(keep=False)
def save_pkl(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)
train_data = train_data[final_col]
test_data = test_data[final_col]
num_seed = train_data[train_data.Rating == 1]
num_nonseed = train_data[train_data.Rating == 0]
file = open('./record_file', 'a+')
file.write('train_data_length: {}\n test_data_length: {}\n'.format(train_data.shape[0], test_data.shape[0]))
file.write('num_seed_length: {}\n num_nonseed_length: {}\n'.format(num_seed.shape[0], num_nonseed.shape[0]))
for feature in columns:  
    record = '{}: {},'.format(feature, max(all_data[feature]) + 1)
    file.write(record)
    file.write('\n')
file.close()
save_pkl(train_data, '{}train_stage.pkl'.format(path + 'processed_data/'))
save_pkl(test_data, '{}test_stage.pkl'.format(path + 'processed_data/'))
print('all data processed finish')

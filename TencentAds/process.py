import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
import random
import pickle
import datetime
root_path = './'
random.seed(2021)
def save_pkl(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)
if os.path.exists('{}userFeature.csv'.format(root_path + "lookalike_data/")):
    print('===the userFeature.csv is exists===')
    user_feature = pd.read_csv('{}userFeature.csv'.format(root_path + "lookalike_data/"))
else:
    print('===the userFeature.csv is not exists,creating...===')
    userFeature_data = []
    with open('{}userFeature.data'.format(root_path + "lookalike_data/"), 'r') as f:
        for i, line in enumerate(f):
            line = line.strip().split('|')
            userFeature_dict = {}
            for each in line:
                each_list = each.split(' ')
                userFeature_dict[each_list[0]] = ' '.join(each_list[1:])
            userFeature_data.append(userFeature_dict)
            if i % 100000 == 0:
                print(i)
        user_feature = pd.DataFrame(userFeature_data)
        user_feature.to_csv('{}userFeature.csv'.format(root_path + "lookalike_data/"), index=False)
        del userFeature_data
user_feature.uid = user_feature.uid.astype('float64')
user_feature = user_feature.fillna('-1')
ID_col = ['aid']
item_col = ['advertiserId', 'campaignId', 'creativeId', 'creativeSize', 'adCategoryId', 'productId', 'productType']  # 7
static_context_col = ['age', 'gender', 'education', 'carrier', 'LBS', 'consumptionAbility', 'house']  # 7
dynamic_context_col = ['ct', 'interest1', 'interest2', 'interest3', 'interest4', 'interest5',
                       'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3']   # 12
columns = ID_col + ['label'] + item_col + static_context_col + dynamic_context_col  # 28
dict_dynamic_length = {
    'ct': 4,
    'interest1': 16,
    'interest2': 16,
    'interest3': 16,
    'interest4': 16,
    'interest5': 16,
    'kw1': 5,
    'kw2': 5,
    'kw3': 5,
    'topic1': 5,
    'topic2': 5,
    'topic3': 5
}
start_time = datetime.datetime.now()
dict_dynamic_set = {
    'ct': {-1},
    'interest1': {-1},
    'interest2': {-1},
    'interest3': {-1},
    'interest4': {-1},
    'interest5': {-1},
    'kw1': {-1},
    'kw2': {-1},
    'kw3': {-1},
    'topic1': {-1},
    'topic2': {-1},
    'topic3': {-1}
}
num = 0
for fea_name in dynamic_context_col:
    user_feature[fea_name + '_length'] = 0
def extract_dynamic_feature(x):
    global num
    for fea_name in dynamic_context_col:
        mark = False
        if x[fea_name] == '-1':
            mark = True
        tmp = x[fea_name].split(' ')[:16]
        if mark == True:
            x[fea_name + '_length'] = 0
        else:
            x[fea_name + '_length'] = len(tmp)
        tmp = np.array(tmp).astype('int')
        x[fea_name] = tmp
    num += 1
    if num % 50000 == 0:
        print('processed ID: {}'.format(num))
        end_time = datetime.datetime.now()
        print('time:', (end_time - start_time).seconds)
    return x
user_feature = user_feature.apply(func=extract_dynamic_feature, axis=1)
print('---------writing data....')
save_pkl(user_feature, './processed_data/user_feature1.pkl')
print('feature1')
for fea_name in dynamic_context_col:
    tmp = pad_sequences(user_feature[fea_name], maxlen=dict_dynamic_length[fea_name], padding='post', truncating='post',
                        value=-1)
    user_feature[fea_name] = tmp.tolist()
    print(fea_name + ' padding finished')
print(user_feature)
print('---------writing data....')
save_pkl(user_feature, './processed_data/user_feature2.pkl')
print('feature2')
for fea_name in dynamic_context_col:
    tmp_list = []
    for item in user_feature[fea_name]:
        tmp_list.extend(item)
    dict_dynamic_set[fea_name] = set(tmp_list)
    print(fea_name + ' set finished')
dict_map = {}
for fea_name in dynamic_context_col:
    dict_map[fea_name] = dict(zip(dict_dynamic_set[fea_name], range(len(dict_dynamic_set[fea_name]))))
    print('{}: {},'.format(fea_name, len(dict_dynamic_set[fea_name])))
def map_dict_func(x):
    for fea_name in dynamic_context_col:
        x[fea_name] = np.array([dict_map[fea_name][i] for i in x[fea_name]])
    return x
user_feature = user_feature.apply(func=map_dict_func, axis=1)
file = open('./record_file', 'a+')
for fea_name in dynamic_context_col:
    record = '{}: {},\n'.format(fea_name, len(dict_dynamic_set[fea_name]))
    file.write(record)
file.close()
del dict_dynamic_set
print('---------writing data....')
save_pkl(user_feature, './processed_data/user_feature3.pkl')
print('feature3')
print('======the user data proceed finish =======')
print('------concat data....')
ad_feature = pd.read_csv('{}adFeature.csv'.format(root_path + "lookalike_data/")).fillna('-1')
train_data = pd.read_csv('{}train.csv'.format(root_path + "lookalike_data/"))
test_data = pd.read_csv('{}test1_truth.csv'.format(root_path + "lookalike_data/"), header=None,
                        names=['aid', 'uid', 'label'])
train_data.loc[train_data['label'] == -1, 'label'] = 0
test_data.loc[test_data['label'] == -1, 'label'] = 0
train_data = pd.merge(train_data, ad_feature, on='aid', how='left')
train_data = pd.merge(train_data, user_feature, on='uid', how='left')
test_data = pd.merge(test_data, ad_feature, on='aid', how='left')
test_data = pd.merge(test_data, user_feature, on='uid', how='left')
del user_feature
train_data_length = train_data.shape[0]
test_data_length = test_data.shape[0]
columns = ID_col + item_col + static_context_col
print(train_data_length)
all_data = pd.concat([train_data, test_data]).fillna('-1')
for feature in columns:
    try:
        all_data[feature] = LabelEncoder().fit_transform(all_data[feature].apply(int))
    except:
        all_data[feature] = LabelEncoder().fit_transform(all_data[feature])
for feature in columns:
    print('{}: {},'.format(feature, max(all_data[feature]) + 1))
file = open('./record file', 'a+')
file.write('train_data_length: {}; test_data_length: {} \n'.format(train_data_length, test_data_length))
for feature in columns:
    record = '{}: {},'.format(feature, max(all_data[feature]) + 1)
    file.write(record)
    file.write('\n')
    print(record)
file.close()
train_data = all_data[: train_data_length]
test_data = all_data[train_data_length:]
del all_data
train_aid_seed_counts = train_data[train_data['label'] == 1].aid.value_counts()
test_aid_seed_counts = test_data[test_data['label'] == 1].aid.value_counts()
print("the sum of aid: {}".format(len(train_aid_seed_counts)))
train_aid_set = set(train_aid_seed_counts.index)
test_aid_set = set(test_aid_seed_counts.index)
print('if train_aid_set==test_aid_set? {}'.format(train_aid_set == test_aid_set))
cold_aid_set = set(train_aid_seed_counts[train_aid_seed_counts <= 1050].index)
hot_aid_set = train_aid_set - cold_aid_set
file = open('./record file', 'a+')
file.write('the aid list and record number in train:\n')
for aid, value in zip(train_aid_seed_counts.index, train_aid_seed_counts.values):
    file.write('aid: {}  number: {}\n'.format(aid, value))
file.write('\n')
file.write("the sum of aid: {} \n".format(len(train_aid_seed_counts)))
file.write('if train_aid_set==test_aid_set? {}\n'.format(train_aid_set == test_aid_set))
file.write('the cold_aid_set:\n')
for aid in cold_aid_set:
    file.write('{} '.format(aid))
file.write('\n')
file.write('the sum of cold aid is: {}\n'.format(len(cold_aid_set)))
file.write('the hot_aid_set:\n')
for aid in hot_aid_set:
    file.write('{} '.format(aid))
file.write('\n')
file.write('the sum of hot aid is: {}\n'.format(len(hot_aid_set)))
file.close()
print('sampling offline & online data....')
offline_list = []
for aid in set(train_aid_set):
    seed_record = train_data.loc[train_data['aid'] == aid].loc[train_data.loc[train_data['aid'] == aid]['label'] == 1]
    nonseed_record = train_data.loc[train_data['aid'] == aid].loc[train_data.loc[train_data['aid'] == aid]['label'] == 0]
    offline_seed = random.sample(set(seed_record.index), int(len(seed_record) / 2))
    offline_nonseed = random.sample(set(nonseed_record.index), int(len(nonseed_record) / 2))
    offline_list.extend(offline_seed)
    offline_list.extend(offline_nonseed)
online_list = list(set(train_data.index).difference(set(offline_list)))
offline_data = train_data.iloc[offline_list]
print('writing offline data to local file....')
save_pkl(offline_data, './processed_data/offline_train_stage.pkl')
del offline_data, offline_list
online_data = train_data.iloc[online_list]
del train_data
print('writing online data to local file....')
save_pkl(online_data, './processed_data/online_train_stage.pkl')
online_cold_data = online_data.loc[(online_data['aid'].isin(cold_aid_set))]
online_hot_data = online_data.loc[(online_data['aid'].isin(hot_aid_set))]
del online_data, online_list
print('writing online cold data to local file....')
save_pkl(online_cold_data, './processed_data/online_cold_data.pkl')
del online_cold_data
print('writing online hot data to local file....')
save_pkl(online_hot_data, './processed_data/online_hot_data.pkl')
del online_hot_data
test_cold_data = test_data.loc[(test_data['aid'].isin(cold_aid_set))]
print('writing test cold data to local file....')
save_pkl(test_cold_data, './processed_data/test_cold_data.pkl')
del test_cold_data
test_hot_data = test_data.loc[(test_data['aid'].isin(hot_aid_set))]
print('writing test cold data to local file....')
save_pkl(test_hot_data, './processed_data/test_hot_data.pkl')
del test_hot_data
print('all data processed finish')

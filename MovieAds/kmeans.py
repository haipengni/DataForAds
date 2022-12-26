import random
import pandas as pd
import numpy as np
random.seed(2022)
path = './movie_1m/'
class AdsKmCluster():
    def __init__(self, dataset, k):
        self.dataset = dataset
        self.k = k
        self.aid_dict = {}
        self.aid_center = {}
    def distance(self, u, v):
        m = u - v
        return sum(m * m) ** 0.5
    def initcenters(self):
        num, dim = self.dataset.shape
        index = random.sample(range(num), self.k)
        for aid in range(self.k):
            self.aid_center[aid] = self.dataset[index[aid]]
    def kmeans(self):
        num = self.dataset.shape[0]
        centerchange = True
        self.initcenters()
        while centerchange:
            centerchange = False
            for i in range(self.k):
                self.aid_dict[i] = []
            for i in range(num):
                mindis = 100000.0
                minindex = 0
                for j in range(self.k):
                    dis = self.distance(self.dataset[i], self.aid_center[j])
                    if dis < mindis:
                        mindis = dis
                        minindex = j
                self.aid_dict[minindex].append(i)
            newcenters = {}
            for tem in range(self.k):
                if len(self.aid_dict[tem]) != 0:
                    newcenters[tem] = self.dataset[self.aid_dict[tem]].mean(0)
                    if (newcenters[tem] - self.aid_center[tem]).any():
                        self.aid_center[tem] = newcenters[tem]
                        centerchange = True
        return self.aid_dict, self.aid_center
movies_genres = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama',
                 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-F', 'Thriller',
                 'War', 'Western']
movies_col = ['MovieID', 'Title', 'Year'] + movies_genres
final_col = ['GroupID'] + movies_col
movie_data = list()
with open('{}movies.dat'.format(path)) as f:
    for i, record in enumerate(f):
        movie_dict = {}
        record = record.strip().split('::')
        movie_dict['MovieID'] = int(record[0])
        srecord = record[1][::-1]
        movie_dict['Title'] = srecord[7:][::-1]
        movie_dict['Year'] = int(srecord[1:5][::-1])
        trecord = record[2].split('|')
        for fea in trecord:
            movie_dict[fea] = 1
        movie_data.append(movie_dict)
movie_feature = pd.DataFrame(movie_data, columns=movies_col)
movie_feature = movie_feature.fillna(0)
movie_feature.MovieID = movie_feature.MovieID.astype('float64')
movie_feature.to_csv('{}movieFeature.csv'.format(path), index=False)
movie_feature = pd.read_csv('{}movieFeature.csv'.format(path)).fillna(0)
print(len(set(movie_feature.MovieID)))
minY = movie_feature.Year.min()
maxY = movie_feature.Year.max()
def regularY(x):
    for fea in ['Year']:
        x['YearR'] = (x[fea] - minY) / (maxY - minY)
    return x
movie_feature = movie_feature.apply(func=regularY, axis=1)
cluster_genres = ['YearR'] + movies_genres
dataset = np.array(movie_feature[cluster_genres])
group_k = 50
aid_cluster, center = AdsKmCluster(dataset, group_k).kmeans()
aid_result_dict = {}
for k in range(group_k):
    for j in aid_cluster[k]:
        aid_result_dict[j] = [k, movie_feature.iloc[j].MovieID]
aid_result_data = pd.DataFrame.from_dict(aid_result_dict, orient='index', columns=['GroupID', 'MovieID'])
all_data = pd.merge(movie_feature, aid_result_data, on='MovieID', how='left')
all_data = all_data[final_col]
all_data.to_csv('{}adsFeature.csv'.format(path), index=False)


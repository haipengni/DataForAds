# MovieLens

MovieLens datasets https://grouplens.org/datasets/movielens

The MovieLens datasets are collected by GroupLens Research from the MovieLens web site ([https://movielens.org](https://movielens.org/)) where movie rating data are made available. The datasets have been widely used in various research on recommender systems.

In this paper,  the MovieLens 1M Dataset (https://grouplens.org/datasets/movielens/1m/)  was selected. 



## Data Description

The public MovieLens 1M Dataset contains 6,040 users, each of them consists of user ID, gender, age, occupation, and zip code. And holds 3,883 movies, each movie including movie ID, title, and genres. And it was rated by user with a score that among of 5 scale, and recorded timestamp for the rating behavior.

 In this study, in order to fit the audience targeting, we firstly according to the movie genres and the normalized years, the years were extracted from the movie title, to cluster all the movies into 50 groups thought k-means method. Each group was regarded as an ad group, and target audiences were found for each ad group. Meanwhile, in order to make the sample data suitable for CTR prediction task, we converted the rating data into a binary classification data.

### The Processed Data

*how to get the processed data for the study?*

Step 1: we run the kmeans.py in the pycharm  terminal

```
python kmeans.py
```

After this operation, all the movies into 50 groups, Each group was regarded as an ad group, and target audiences were found for each ad group.

Step 2: we run the process.py in the pycharm  terminal

```
python process.py
```

After this operation,  we obtained the training set and testing set. In this dataset we consider training set as both offline and online data, the offline data and online data are obtained to simulate the whole audience targeting process.



*Statistics about the MovieLens dataset  for the study are shown in the table.*

| Ads/Movies | Seeds  | Non-seeds | Candidates |
| ---------- | ------ | --------- | ---------- |
| 50/3883    | 461206 | 337438    | 201565     |

Due to capacity limitations in github, we release the processed data in baidu  web disk,  the origin data and  processed data are available for [download here](https://pan.baidu.com/s/1E9G9cPV9Bl06o_WmGmlfaA?pwd=uhc0).




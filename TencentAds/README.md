# Tencent Lookalike

Lookalike datasets (https://algo.qq.com/archive.html?)

## Data Description

The public dataset for Tencent Ads competitions in 2018 is based on the advertisers providing more than one hundred seed sets, which contains a large number of user characteristics, and aims to expand potential audiences for these campaigns. To ensure the security of service data, all data is desensitized. The whole dataset is divided into training set and test set. Each advertisement has eight categorical features: ad ID, advertiser ID, campaign ID, creative ID, creative size ID, ad category, product ID and product type. Each user contains 19 features: including age, gender, education, carrier, consumption ability, geographical location, house, type of Internet access, five groups of interest categories, three groups of topics, three groups of keywords.

### The Processed Data

*how to get the processed data for the study?*

 We run the process.py in the pycharm  terminal

```
python process.py
```

After this operation,  we obtained the training set and testing set. In this dataset, the offline data and online data are obtained to simulate the whole audience targeting process.



*Statistics about the lookalike dataset  for the study are shown in the table.*

| Ads  | Seeds  | Non-seeds | Candidates |
| ---- | ------ | --------- | ---------- |
| 173  | 421961 | 8376853   | 2265989    |

Due to capacity limitations in github warehouse, we release the processed data in baidu  web disk,  the origin data and  processed data are available for [download here](https://pan.baidu.com/s/10OGRxPaMutpXDK_vGjM1YA?pwd=cbi2).




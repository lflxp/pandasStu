import pandas as pd
# https://colab.research.google.com/notebooks/mlcc/intro_to_pandas.ipynb?hl=zh-cn#scrollTo=0gCEX99Hb8LR 
print pd.Series(['San Francisco', 'San Jose', 'Sacramento'])

city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])
name = pd.Series(['lixueping','liufangli','lijiacheng','liuwanru'])

cites = pd.DataFrame({ 'City name': city_names, 'Population': population ,'Great people':name})
print cites

cali = pd.read_csv("./c.csv",sep=",")
print cali.describe()

print cali.head()

# print cali.hist('housing_median_age')

print type(cites['City name'])
print cites['City name']
print cites['City name'][0]
print cites['City name'][0:2]
print population/1000

import numpy as np 
print np.log(population)

cites['Area square miles'] = pd.Series([46.87, 176.53, 97.92])
cites['Population density'] = cites['Population'] / cites['Area square miles']

print cites

print population.apply(lambda val: val>1000000)

print cites.index

# print cites.reindex(np.random.permutation(cites.index))
print np.random.permutation(cites.index)


print cites.reindex([0,3,1,2])


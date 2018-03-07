import pandas as pd 
print pd.Series(['San Francisco', 'San Jose', 'Sacramento'])

city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])
name = pd.Series(['lixueping','liufangli','lijiacheng','liuwanru'])

print pd.DataFrame({ 'City name': city_names, 'Population': population ,'Great people':name})


cali = pd.read_csv("./c.csv",sep=",")
print cali.describe()

print cali.head()
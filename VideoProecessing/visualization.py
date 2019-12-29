import pandas as pd

#https://stackoverflow.com/questions/34158103/sort-a-file-by-first-or-second-or-else-column-in-python

df=pd.read_csv('data.csv',  names=['a','b','c'])
# df=df.sort_values(["frame"], axis=0)
# df.to_csv (r'/Users/wangmeijie/ALLImportantProjects/Flame+MaskRCNN/data/sortedData.csv', index = None, header=True)

df['a'] = df['a'].str.replace(r'\D', '')


df = df.sort_values('a')
print (df)

df.to_csv('/Users/wangmeijie/ALLImportantProjects/Flame+MaskRCNN/data/sortedData.csv', encoding='utf-8', index=False)
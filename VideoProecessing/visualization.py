import pandas as pd
import numpy as np

#https://stackoverflow.com/questions/34158103/sort-a-file-by-first-or-second-or-else-column-in-python

df=pd.read_csv('data.csv',  names=['I','S','P'])
# df=df.sort_values(["frame"], axis=0)
# df.to_csv (r'/Users/wangmeijie/ALLImportantProjects/Flame+MaskRCNN/data/sortedData.csv', index = None, header=True)

df['a'] = df['a'].str.replace(r'\D', '')

df.a = pd.to_numeric(df.a, errors='coerce')
df = df.sort_values('a', ascending=True)

print (df)

df.to_csv('/Users/wangmeijie/ALLImportantProjects/Flame+MaskRCNN/data/sortedData.csv', encoding='utf-8', index=False)
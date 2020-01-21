import pandas as pd
import numpy as np

#https://stackoverflow.com/questions/34158103/sort-a-file-by-first-or-second-or-else-column-in-python

# Area

# df=pd.read_csv('data.csv',  names=['I','S','P'])
# df['I'] = df['I'].str.replace(r'\D', '')
# df.I = pd.to_numeric(df.I, errors='coerce')
# df = df.sort_values('I', ascending=True)
# print (df)
# df.to_csv('/Users/wangmeijie/ALLImportantProjects/FlameDetectionAPP/WebApplication/static/data/sortedData.csv', encoding='utf-8', index=False)

#Height
# dh=pd.read_csv('height.csv',  names=['N','H'])
#
# # df.N = pd.to_numeric(df.N, errors='coerce')
# # df = df.sort_values('I', ascending=True)
# print (dh)
# dh.to_csv('/Users/wangmeijie/ALLImportantProjects/FlameDetectionAPP/WebApplication/static/data/heightData.csv', encoding='utf-8', index=False)


dD=pd.read_csv('Data1.csv',  names=['I','H','S','P'])
print (dD)
dD.to_csv('/Users/wangmeijie/ALLImportantProjects/FlameDetectionAPP/WebApplication/static/data/Data1.csv', encoding='utf-8', index=False)
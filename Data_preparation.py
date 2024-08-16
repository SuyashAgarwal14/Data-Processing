import pandas as pd
import numpy as np

#Handling missing data
data=pd.Series([1,3.5,7,np.nan,0,None])                     #nan means it has no value not even zero/missing value/None value
check=data.isna()                                           #returns true for none values
check=data.notna()                                          #returns false for none values
check=data.dropna()                                         #drops none value data
check=data.notna()                                          #gives all non none values

data=pd.DataFrame([[1.,6.5,3.],[1.,np.nan,np.nan],
                  [np.nan,np.nan,np.nan],[np.nan,6.5,3.]])
check=data.dropna()                                         #drops the row that has any none value in it
check=data.dropna(how='all')                                #drops the row that has all none value in it
check=data.dropna(axis=0,how='all')                         #drops according to axis and condition

df=pd.DataFrame(np.random.standard_normal((2,3)))           #draw samples from a standard Normal distribution (mean=0, stdev=1)
df=data.dropna(axis=0,thresh=2)                             #the minimum number of non-NAN values needed


#Filling missing data
df=data.fillna(0.5)                                         #fills all nan value with some particular values
df=data.fillna({0:0.6,1:0.2,2:0.7})                         #fills nan value with some particular values column wise
df=data.fillna(method='ffill')                              #forward fill- fills the last none value in all none values of column
df=data.fillna(method='ffill',limit=1)                      #fills the last none value upto a particular limit of none values 


#Remove duplicate items
data= pd.DataFrame({"k1":["one","two"]*3+["two"],
                   "k2":[1,1,2,3,3,4,4]})
df=data.duplicated()                                        #checks for duplicate data returns boolean
df=data.drop_duplicates()                                   #removes duplicate data
df=data.drop_duplicates(subset=['k1'])                      #removes duplicate data from particular column
df=data.drop_duplicates(subset=['k1','k2'],keep='last')     #removes data that has duplicated values in both columns keeping the last of the duplicated

#Data Wrangling
df=pd.Series(np.random.uniform(size=9),index=[['a','a','a','b','b','c','c','d','d'],[1,2,3,1,3,1,2,2,3]])     #double indexed data can use more indexes
data=df.index                                               #gives a list of tuples containg multi index
data=df[1]                                                  #gives value at particular position
data=df['b']                                                #gives values at particular first index 
data=df['b'][3]                                             #gives value at particular index
data=df['b':'d']
data=df[['b','d']]
data=df.unstack()                                           #to convert into data frame format

frame=pd.DataFrame(np.arange(12).reshape((4,3)),index=[["a","a","b","b"],[1,2,1,2]],
                   columns=[["Ohio","Ohio","Colorado"],["Green","Red","Green"]])        #data frame with multiple columns
data=frame.index.nlevels                                    #to check level of indexing



#Merging data frames
#pandas.merge-   Connect rows in DataFrames based on one or more keys
df1=pd.DataFrame({"key":["b","b","a","c","a",'a',"b"],"data1":pd.Series(range(7))})
df2=pd.DataFrame({"key":["a","b","d"],"data2":pd.Series(range(3))})
data=pd.merge(df1,df2)                                         #data with similar key gets merged
data=pd.merge(df1,df2,how='outer')                             #use union of keys from both frames sort keys lexicographically with similar keys

df3=pd.DataFrame({"lkey":["b","b","a","c","a",'a',"b"],"data1":pd.Series(range(7))})
df4=pd.DataFrame({"rkey":["a","b","d"],"data2":pd.Series(range(3))})
data=pd.merge(df3,df4,left_on='lkey',right_on='rkey')          #merge data with different keys and defining positon of the key
data=pd.merge(df3,df4,left_on='lkey',right_on='rkey',how='outer')       

df5= pd.DataFrame({"key":["b","b","a","c","a","b"],"data1":pd.Series(range(6),dtype="Int64")})
df6=pd.DataFrame({"key":["a","b","a","b","d"],"data2":pd.Series(range(5),dtype="Int64")})
data=pd.merge(df5,df6,how='left')       
data=pd.merge(df5,df6,how='inner')                             #use intersection of keys from both frames preserve order

left1=pd.DataFrame({"key":["a","b","a","a","b","c"],"value":pd.Series(range(6),dtype="Int64")})
right1=pd.DataFrame({"group_val":[3.5,7]}, index=["a","b"])
data=pd.merge(left1,right1,left_on='key',right_index=True)     #Use the index from the right DataFrame as the join key

#Conactenating- Cncatenate or stack objects together along an axis
s1= pd.Series([0,1], index=["a","b"],dtype="Int64")
s2=pd.Series([2,3,4],index=["c","d","e"],dtype="Int64")
s3=pd.Series([5,6],index=["f","g"],dtype="Int64")
data=pd.concat([s1,s2,s3],axis=1)

#combine_first-  Splice together overlapping data to fill in miing values in one object with values from another
a=pd.Series([np.nan,2.5,0.0,3.5,4.5,np.nan],index=["f","e","d","c","b","a"])
b=pd.Series([0.,np.nan,2.,np.nan,np.nan,5.],index=["a","b","c","d","e","f"])
np.where(pd.isna(a),b,a)
a.combine_first(b)

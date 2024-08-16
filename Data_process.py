"""
Data Preprocessing
Data Preprocessing invloves cleaning and engineering data in a way that it can be used as input to several important data science tasks such as data visualization, machine learning, deep learning and data analytics.
Some of the most common data preparation tasks include feature scaling, handling missing values, categorical variable encoding, data discretization.

Feature Scaling
A dataset can have different attributes. The attributes can have different magnitudes, variances, standard deviation, mean value etc.
For instance, salary can be in thousands, whereas age is normally a two-digit number.The difference in the scale or magnitude of attributes can actually affect statistical models.
For instance, variables with bigger ranges dominate those woth smaller ranges for linear models.
Feature Scaling is applied on numeric data only.

Standardization
Standardization is the process of centering a variable at zero and standardizing the data variance to 1.
To standardize a dataset, you simply have to subtract each data point from the mean of all the data points and divide the result by the standard deviation of the data.

"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns

titanic_data=pd.read_csv("E:\\Programs\\Python\\Data processing\\titanic.csv")
titanic_data= titanic_data[["Age","Fare","Pclass"]]
scaler=StandardScaler()                                                     #it makes mean = 0 and scales the data to unit variance
sacler=scaler.fit(titanic_data)                                             #Compute the mean and std for a given feature to be used for later scaling
data=scaler.transform(titanic_data)
data=pd.DataFrame(data,columns=titanic_data.columns)
sns.kdeplot(data['Age'])                                                    #plots probability distribution function graph
plt.show()


"""
Min/Max Scaling
In min/max scaling you subtract each value by the minimum value and then divide the result by the difference between minimum and maximum value in the dataset
"""
from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()
scaler=scaler.fit(titanic_data)
data=scaler.transform(titanic_data)
data= pd.DataFrame(data, columns=titanic_data.columns)
sns.kdeplot(data['Age'])
plt.show()
 
"""
Handling Missing Data
Missing values are those observations in the dataset that do not contain any value.
Missing values can totally change data patterns and therefore it is extremely important to understand why missing values occur in the dataset and how to handle them.

Handling Missing Numerical Data
To handle missing numerical data, we can use statistical techniques. 
The use of statistical techniques or algorithms to replace missing values with statistically generated values is called imputation.
One of the most common ways of handling missing values in a categorical column is to replace the missing values with the most frequenly occuring values i.e the mode of the column
"""

data=titanic_data.isnull().mean()                                           #isnull() only it will return us True or False for each and every cell
#isnull().mean()- it gives us the probability of null values

median=titanic_data.Age.median()
mean=titanic_data.Age.mean()
titanic_data["Median_Age"]=titanic_data.Age.fillna(median)
titanic_data['Mean_Age']=titanic_data.Age.fillna(mean)
titanic_data['Mean_Age']=np.round(titanic_data['Mean_Age'],1)

titanic_data=pd.read_csv("E:\\Programs\\Python\\Data processing\\train.csv")
titanic_data= titanic_data[["embark_town","age","fare"]]
titanic_data.embark_town.value_counts().sort_values(ascending=False).plot.bar()

titanic_data.embark_town.mode()
titanic_data.embark_town.fillna('Southampton',inplace=True)

plt.xlabel("Embark Town")
plt.ylabel("Number of Passengers") 
plt.show()


"""
Categorical Data Encoding
Models based on statistical algorithms such as machine learning and deep learning work with numbers.
A dataset can contain numerical, categorical, datetime and mixed variables.
A mechanism is needed to convert categorical data to its numeric counterpart so that the data can be used to build statistical models.
The techniques used to convert categorical data to its numeric data are called categorical data encoding schemes.
"""

"""
One Hot Encoding
One Hot Encoding is one of the most commonly used categorical encoding schemes.
In one hot encoding for each unique value in the categorical column a new column is added.
Integer 1/True is added to the column that corresponds to original label and all the remaining column are filled with 0s/False
"""

"""
The get_dummies can't handle the unknown category during the transformation natively.You have to apply some techniques to handle it. But it is not efficient.
On the other hand, OneHotEncoder will natively handle unknown categories. All you need to do is set the parameter handle_unknown='ignore' to OneHotEncoder.
"""

titanic_data=pd.read_csv("E:\\Programs\\Python\\Data processing\\train.csv")
titanic_data= titanic_data[["sex","class","embark_town"]]
temp= pd.get_dummies(titanic_data['sex'])
pd.concat([titanic_data["sex"],pd.get_dummies(titanic_data['sex'])],axis=1).head()
temp=pd.get_dummies(titanic_data["embark_town"])
"""
Convert categorical variable into dummy/indicator variables.
Each variable is converted in as many 0/1 variables as there are different values. 
Columns in the output are each named after a value; if the input is a DataFrame, the name of the original variable is prepended to the value.
"""

"""
Label Encoding
In label encoding, labels are replaced by integers. This is why label encoding is also called as integer encoding.
Converts categorical variables into numerical format
"""
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()
le.fit(titanic_data['class'])
titanic_data['le_class']=le.transform(titanic_data['class'])

"""
Data Discretization
The process of converting continuous numeric values such as price, age and weight into discrete intervals is called discretization or binning.
Discretization is particularly helpful in cases where you have a skewed distribution of data.

Equal Width Discretization
The most common type of discretization approach is fixed width discretization.
"""

import warnings
warnings.filterwarnings('ignore')
data=pd.read_csv("E:\\Programs\\Python\\Data processing\\diamonds.csv")
sns.distplot(data['price'])                                                 #depicts the variation in the data distribution
plt.show()

#The histogram for price column shows that the data is positively skewed.
price_range=data['price'].max()-data['price'].min()
lower_interval=int(np.floor(data['price'].min()))
upper_interval=int(np.ceil(data['price'].max()))
interval_length=int(np.round(price_range/10))
total_bins=[i for i in range(lower_interval,upper_interval+interval_length,interval_length)]
bin_labels= ['Bin_no_'+str(i) for i in range(1,len(total_bins))]
data['price_bins']=pd.cut(x=data['price'],bins=total_bins,labels=bin_labels, include_lowest=True)
#cut()-  segment and sort data values into bins. This function is also useful for going from a continuous variable to a categorical variable.
data.groupby('price_bins')['price'].count().plot.bar()
plt.xticks(rotation=45)
plt.show()
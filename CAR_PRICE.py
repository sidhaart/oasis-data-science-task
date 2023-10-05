 
# #### #Task 3 - CAR PRICE PREDICTION WITH MACHINE LEARNING
# #### Problem Statement:
# * Analyse how various factors like features of the car, horsepower, mileage etc affect the price of the car 
# * Use Machine Learning Techniques for Predicting the car price using Python Programming

# In[1]:


#importing necessary libraries 
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
#importing libraries for visualisation
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns


# In[2]:


#importing Data
data_frame = pd.read_csv('C:/Users/sinun/OneDrive/Documents/oasis infobyte/car_price_prediction/CarPrice.csv')


# ####  Performing descriptive analysis. Understand the variables and their corresponding values. 

# In[3]:


# Understanding the dimensions of data
data_frame.shape


# In[4]:


# Understanding the Data Variables
data_frame.info()


# In[5]:


#Identify columns in Dataset
data_frame.columns


# In[6]:


# Show the top 5 Rows of data
data_frame.head()


# In[7]:


# Performing Descriptive Analysis
data_frame.describe().T


# In[8]:


# Checking for null values
data_frame.isnull().sum()


# In[9]:


#Dropping unwanted Columns from data
data_frame.drop(columns=['car_ID'], inplace=True )


# In[10]:


# Identifing Categorical(Non-Numerical) Columns in Dataset
data_frame_cat=data_frame.select_dtypes(exclude=['float64','int64'])
data_frame_cat.columns


# In[11]:


# Identifing Numerical Columns in Dataset
data_frame_num=data_frame.select_dtypes(include=['float64','int64'])
data_frame_num.columns


# In[12]:


# Find Unique values in each Categorical column and their count
cat_cols=['CarName','fueltype', 'aspiration', 'doornumber', 'carbody',
       'drivewheel', 'enginelocation', 'enginetype', 'cylindernumber',
       'fuelsystem']
def num_count():
    for col in cat_cols:
        print('Name of the variable :', col)
        print(data_frame[col].value_counts(), '\n\n')
num_count()
    


# #### Data Visualization

# ##### *  Data Visualization helps to show how the different factors affect the Price Variable

# #### Heat Map

# In[13]:


# find correlation between variables in data set for plotting heatmap
df_corr=data_frame.corr()


# In[14]:


plt.figure(figsize=(20,15))
sns.heatmap(df_corr,annot=True,cmap="BuPu")
plt.show()


#  * Variables Enginesize, curbweight, horsepower have high correlation values (above 0.8) with the target Price variable
#  * Factors such as carwidth, carlength, highwaympg and citympg are also having good correlation values (above 0.68) with the target Price variable
#  * Peakrpm, compressionratio, stroke, symboling have very low correlation with Price variable

# ####  Scatter Plot 

# In[15]:


#SCATTER PLOT ENGINE SIZE vs Price
plt.figure(figsize=(6,4))
sns.scatterplot(data=data_frame,x=data_frame['enginesize'],y=data_frame['price'])


# In[16]:


#SCATTER PLOT CURBWEIGHT vs Price
plt.figure(figsize=(6,4))
sns.scatterplot(data=data_frame,x=data_frame['curbweight'],y=data_frame['price'])


# In[17]:


#SCATTER PLOT horsepower vs Price
plt.figure(figsize=(6,4))
sns.scatterplot(data=data_frame,x=data_frame['horsepower'],y=data_frame['price'])


# ##### * It is seen that Enginesize, curbweight, horsepowe mostly follows a linear relationship with the Price variable.

# In[18]:


# For labelling the categorical columns,One Hot encoding is permormed 
data_frame= pd.get_dummies(data_frame, columns = ['fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel', 'cylindernumber', 'enginelocation', 'enginetype','fuelsystem' ])
data_frame.head(5)


# #### Building the Model

# In[19]:


from sklearn.model_selection import train_test_split


# In[20]:


#First step in building the model is to identify the Feature(Input) variables and Target (Output) variable
features = data_frame.drop(['CarName','price'], axis=1)
target = data_frame['price']


# #####  * Splitting data for training and testing the model

# In[21]:


# Splitting data for training the model and testing the model
# train size taken as 0.8
X_train, X_test, y_train, y_test = train_test_split(features, target, train_size = .8)
# Dimensions of Train and Test Data sets
print('Train set of features: ', X_train.shape)
print('Test set of features: ', X_test.shape)
print('Target for train: ', y_train.shape)
print('Target for test: ', y_test.shape)


# ### Learn the model on train data

# In[22]:


from sklearn.linear_model import LinearRegression


# In[23]:


# Linear Regression Model ( a Supervised Machine learning Algorithm)
# LR models impose a linear function between predictor and response variables
my_model = LinearRegression()


# In[24]:


# Fitting the model in train data set ie the Linear Regression Model should learn from the on Train Data
my_model.fit(X_train, y_train)


# #### Predicting the Car Price

# In[25]:


# Predicting the car price from Feature Test values
y_pred = my_model.predict(X_test)
y_pred


# #### Test the model

# In[26]:


from sklearn.metrics import mean_squared_error


# ##### Mean Squared Error

# In[27]:


# Compare the predicted values with the true values
mean_squared_error(y_pred, y_test)


# ##### Coefficient of Determination or R Squared Value (r2)

# In[28]:


from sklearn.metrics import r2_score


# In[29]:


# find Coefficient of Determination or R Squared Value (r2)
r2_score(y_test,y_pred)


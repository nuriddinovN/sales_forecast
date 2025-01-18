import os
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from  xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# <h1>Import data</h1>

# In[134]:


sales_data=pd.read_csv("train.csv")
print(sales_data.head())
sales_data.info()


# <h1>Drop store and item col </h1>

# In[135]:


sales_data = sales_data.drop(['store', 'item'], axis=1, errors='ignore')
sales_data.info()
sales_data.head()


# In[136]:


# Ensure the 'date' column is of datetime type
sales_data['date'] = pd.to_datetime(sales_data['date'])

# Convert the 'date' column to a monthly period
sales_data['date'] = sales_data['date'].dt.to_period("M")

# Group by the monthly period and sum the values
monthly_sales = sales_data.groupby('date').sum().reset_index()

# Convert the 'date' column back to a timestamp
monthly_sales['date'] = monthly_sales['date'].dt.to_timestamp()

# Display the first 10 rows of the result
monthly_sales.head(10)


# <h1>Visualization</h1>

# In[137]:


plt.figure(figsize=(15,7))
plt.plot(monthly_sales['date'],monthly_sales['sales'])
plt.xlabel("date")
plt.ylabel("sales")
plt.title("Monthly custumer sales")
plt.show()


# In[138]:


monthly_sales['sales_diff']=monthly_sales["sales"].diff()
monthly_sales=monthly_sales.dropna()
monthly_sales.head()


# <h1>Prepare supervised data</h1>

# In[139]:


suvervised_data=monthly_sales.drop(["date","sales"],axis=1)
print(f"Supervised data shape: \n{suvervised_data.shape}")
for i in range(1,13):
    col_name="month-"+str(i)
    suvervised_data[col_name]=suvervised_data['sales_diff'].shift(i)
suvervised_data=suvervised_data.dropna().reset_index(drop=True) 
suvervised_data.head()


# <h1>Training and Test set</h1>

# In[140]:


train_data=suvervised_data[:-12]
test_data=suvervised_data[-12:]
print(f'Training set shape: {train_data.shape}\nTest set shape: {test_data.shape}')


# <h1>Data preprocessing</h1>

# In[141]:


scaler=MinMaxScaler(feature_range=(-1,1))
scaler.fit(train_data)
train_data=scaler.transform(train_data)
test_data=scaler.transform(test_data)
#check
print(train_data[1])


# In[142]:


x_train,y_train=train_data[:,1:],train_data[:,0:1]
x_test,y_test=test_data[:,1:],test_data[:,0:1]
y_train=y_train.ravel()
y_test=y_test.ravel()


# In[143]:


print(f"x-train set shape:{x_train.shape}\ny-train set shape:{y_train.shape}\nx-test set shape:{x_test.shape}\ny-test set shape:{y_test.shape}")


# <h1>Make prediction data frame</h1>

# In[144]:


sales_dates=monthly_sales['date'][-12:].reset_index(drop=True)
predict_df=pd.DataFrame(sales_dates)


# In[145]:


actual_sales=monthly_sales["sales"][-13:].to_list()
print(actual_sales)


# <h1>Linear regresion model</h1>

# In[146]:


lr_model=LinearRegression()
lr_model.fit(x_train,y_train)
lr_predict=lr_model.predict(x_test)


# In[147]:


lr_predict=lr_predict.reshape(-1,1)
lr_predict_test=np.concatenate([lr_predict,x_test],axis=1)

lr_predict_test=scaler.inverse_transform(lr_predict_test)


# In[148]:


result=[]
for i in range(0,len(lr_predict_test)):
    result.append(lr_predict_test[i][0]+actual_sales[i])
lr_predict_series=pd.Series(result,name="Linear Prediction")
predict_df=predict_df.merge(lr_predict_series,left_index=True,right_index=True)
print(predict_df)


# <h1>Errors</h1>

# In[149]:


#mean squared error
sqr_err=np.sqrt(mean_squared_error(predict_df["Linear Prediction"],monthly_sales['sales'][-12:]))
#mean absolute error
abs_err=mean_absolute_error(predict_df['Linear Prediction'],monthly_sales["sales"][-12:])
#r2_score
r2=re_score=(predict_df['Linear Prediction'],monthly_sales['sales'][-12:])


# In[150]:


print(f'mean squared error: {sqr_err}\n absolute error: {abs_err}\nr2 score: \n{r2}')


# In[151]:


plt.figure(figsize=(15,7))
plt.plot(monthly_sales['date'],monthly_sales['sales'])
plt.plot(predict_df['date'],predict_df['Linear Prediction'])
plt.title('Custumer sales forecast using LR model')
plt.xlabel("Date")
plt.ylabel('Sales')
plt.legend(["actual sales",'predicted sales'])
plt.show()


# In[ ]:





# In[ ]:





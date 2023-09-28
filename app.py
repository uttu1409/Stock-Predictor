from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
import datetime
import math

start = st.sidebar.date_input('Start Date', datetime.date(2018, 1, 1))
end = st.sidebar.date_input('End Date', datetime.date(2020, 1, 1))


st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
df = data.DataReader(user_input, 'yahoo', start, end)

# Describing Data

st.subheader('Data from 2018 - 2021')
st.write(df.describe())

# Visuallization
st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
plt.xlabel('Time', fontsize=18)
plt.ylabel('Price', fontsize=18)
st.pyplot(fig)

#100MA-- 100 Moving Average
st.subheader('Closing Price vs Time chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100)
plt.plot(df.Close)
plt.xlabel('Time', fontsize=18)
plt.ylabel('Price', fontsize=18)
st.pyplot(fig)


st.subheader('Closing Price vs Time chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close, 'b')
plt.xlabel('Time', fontsize=18)
plt.ylabel('Price', fontsize=18)
st.pyplot(fig)


# Spliting data into Training and testing

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

print(data_training.shape)
print(data_testing.shape)

scaler = MinMaxScaler(feature_range=(0, 1))

data_training_array = scaler.fit_transform(data_training)


# Load my model
model = load_model('keras_model.h5')


# Testing Part

past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Final Graph

st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time', fontsize=18)
plt.ylabel('Price', fontsize=18)
plt.legend()
st.pyplot(fig2)



st.subheader('Closing and Prediction Price')

data= df.filter(['Close'])
#Convert the dataframe to a numpy array
dataset =data.values
#Get the numbr of rows to train the model on
training_data_len = math.ceil(len(dataset)*.8)


#Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)



#Create the training data set
#create the scaled training data set
train_data = scaled_data[0:training_data_len , :]
#splt the data into x_train and y_train data sets
x_train = []
y_train = []

for i in range(60,len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])
    if i<=61:
        print(x_train)
        print(y_train)
        print()

#Convert the x_train and y_train to numpy arrays
x_train ,y_train=np.array(x_train), np.array(y_train)
    
#reshape the data
x_train=np.reshape(x_train,(x_train.shape[0], x_train.shape[1],1))


#Load my model
model = load_model('keras_model1.h5')


#Create the testing daa set
#create a new array containig scaled values from index 1543 to 2003
test_data = scaled_data[training_data_len-60:,:]
#create the data set x_test and y_test
x_test=[]
y_test= dataset[training_data_len:,:]
for i in range(60,len(test_data)):
    x_test.append(test_data[i-60:i,0])

x_test=np.array(x_test)

#Reshape the data
x_test = np.reshape(x_test,(x_test.shape[0], x_test.shape[1],1))

y_predicted = model.predict(x_test)
y_predicted = scaler.inverse_transform(y_predicted)

rmse = np.sqrt(np.mean( y_predicted - y_test)**2)



#plot the data
train =data[:training_data_len]
valid = data[training_data_len:]
valid['Prediction'] = y_predicted
#visualize the model
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date',fontsize=1)
plt.ylabel('Close Price INR (Rs)',fontsize=1)
plt.plot(train['Close'])
plt.plot(valid[['Close','Prediction']])
plt.legend(['Train','Val','Prediction'],loc='lower right')
plt.show()

# Show the valid and predicted price
valid


import pandas_datareader as data


df = data.DataReader(user_input, 'yahoo', start, end)

#Create a new dataframe
new_df = df.filter(['Close'])
#Get the last 60 days closing price values and convert the datframe to an array
last_60_days = new_df[-60:].values
#Scale the data to be values between 0 and 1
last_60_days_scaled = scaler.transform(last_60_days)
#Create an empty list 
X_test = []
#Append the past 60 days
X_test.append(last_60_days_scaled)
#Convert the X_test data set to numpy
X_test = np.array(X_test)
#reshape the data
X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1], 1))
#Get the predicted scale
pred_price = model.predict(X_test)
#Undo the scaling
pred_price = scaler.inverse_transform(pred_price)


st.subheader('Predicted Price')
st.text(end)
st.text(pred_price)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler


def stockpred(day):
# Reading dataset
   stock_data=pd.read_csv(r"C:\Users\kiit1\deploy\S&P500_HistoricalData_1656311331506.csv")

   std_1=stock_data.copy()
   std_1["Volume"]=std_1["Volume"].replace("--",np.nan)
   std_1 = std_1.dropna(axis=1)

   std_2 = std_1.copy()
   std_2["High"]=std_2["High"].replace(0, np.nan)
   std_2=std_2.dropna(axis=0)

   std_2["Date"]=pd.to_datetime(std_2["Date"],format="%m/%d/%Y")
   std_2["year"]=std_2["Date"].dt.year
   c=[2015,2016,2018,2019,2020]
   y=[]
   for i in c:
      s= std_2[std_2.year==i]
      y.append(s)

# bifuracting years w.r.t one which is not having outliers
   d=[2012,2013,2014,2017,2021,2022]
   k=[]
   for i in d:
      b = std_2[std_2.year==i]
      k.append(b)

# removing outliers
   def remove_outliers(col):
      sorted(col)
      Q1, Q3 = col.quantile([0.25,0.75])
      IQR= Q3- Q1
      lower_range = Q1-(1.5*IQR)
      upper_range= Q3 + (1.5*IQR)
      return lower_range,upper_range
   for i in y:
     low_point,high_point= remove_outliers(i["Close/Last"])
     i["Close/Last"]= np.where(i["Close/Last"]>high_point,high_point,i["Close/Last"])
     i["Close/Last"]= np.where(i["Close/Last"]<low_point,low_point,i["Close/Last"])
   new_s_p500=pd.concat([k[0],k[1],k[2],y[0],y[1],k[3],y[2],y[3],y[4],k[4],k[5]],axis=0)
   new_s_p500=new_s_p500.sort_values("Date", ascending=True)

# normalizing the close/last price column
   from sklearn.preprocessing import MinMaxScaler
   scaler=MinMaxScaler(feature_range=(0,1))
   df1=scaler.fit_transform(np.array(new_s_p500["Close/Last"]).reshape(-1,1))

##splitting dataset into train and test split
   training_size=int(len(df1)*0.7)
   test_size=len(df1)-training_size
   train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]


# convert an array of values into a dataset matrix
   def create_dataset(dataset, time_step=1):
	   dataX, dataY = [], []
	   for i in range(len(dataset)-time_step-1):
		    a = dataset[i:(i+time_step), 0]   
		    dataX.append(a)
		    dataY.append(dataset[i + time_step, 0])
	   return np.array(dataX), np.array(dataY)

# reshape into X=t,t+1,t+2,t+3 and Y=t+4
   time_step = 15
   X_train, y_train = create_dataset(train_data, time_step)
   X_test, ytest = create_dataset(test_data, time_step)
#print(X_train.shape), print(y_train.shape)
#print(X_test.shape), print(ytest.shape)

# reshape input to be [samples, time steps, features] which is required for LSTM
   X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
   X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)


### Create the Stacked LSTM model

   from tensorflow.keras.models import Sequential
   from tensorflow.keras.layers import Dense
   from tensorflow.keras.layers import LSTM

   model=Sequential()
   model.add(LSTM(50,return_sequences=True,input_shape=(15,1)))
   model.add(LSTM(50,return_sequences=True))
   model.add(LSTM(50))
   model.add(Dense(1))
   model.compile(loss='mean_squared_error',optimizer='adam')


   model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=50,batch_size=15,verbose=1)
   new_s_p500=new_s_p500.sort_values("Date", ascending=False)
   def date(day):
     k=new_s_p500[new_s_p500["Date"]==day].index.values
     k=k[0]
     x=new_s_p500.iloc[k:k+15,:]
     return x[["Close/Last"]]
   y=date(day)
   l=np.array(y)
   l.reshape(1,-1)
   x_input=scaler.fit_transform(l)
   temp_input=list(x_input)
   temp_input=temp_input[0].tolist()

# demonstrate prediction for next 10 days
   from numpy import array

   lst_output=[]
   n_steps=15
   i=0
   while(i<1):
    
    if(len(temp_input)>15):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
      #   print(x_input)
        yhat = model.predict(x_input, verbose=0)
      #   print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
      #   print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
      #   print(yhat[0])
        temp_input.extend(yhat[0].tolist())
      #   print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
   return scaler.inverse_transform(lst_output)

"""# print(lst_output)

   day_new=np.arange(1,16)
   day_pred=np.arange(16,23)

len(df1)

plt.figure(figsize=(16,8))
plt.plot(day_new,scaler.inverse_transform(df1[2500:]),"b",label="last 15days trens")
plt.plot(day_pred,scaler.inverse_transform(lst_output),"-r",label="predicted 7days")
plt.xlabel("test and future prediction days")
plt.ylabel("Close price")
plt.legend(loc="upper left")

df3=df1.tolist()
df3.extend(lst_output)
plt.plot(scaler.inverse_transform(df3[2490:]))

df3=scaler.inverse_transform(df3).tolist()
plt.figure(figsize=(20,8))
plt.xlabel("total observations")
plt.ylabel("Close/Last price till next 7 days")
plt.plot(df3)
"""

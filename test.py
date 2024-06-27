import pandas as pd
import numpy as np
df=pd.read_csv("add (1).csv")
import pickle
from sklearn.model_selection import train_test_split
#for trainig purpose use train for test
X=df[['x','y']]
Y=df['sum']
X_train, X_test ,Y_train ,Y_test=train_test_split(
X,Y,test_size=0.33,random_state=8)

from sklearn.linear_model import LinearRegression

result=LinearRegression()
result.fit(X_train,Y_train)
#fit method is linearregression class method model
pickle.dump(result,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
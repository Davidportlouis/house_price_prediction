import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

#Data Loading
df = pd.read_csv('./dataset/streeteasy.csv')
# print(df.shape)

#Data Preprocessing
#The features with positive correlation were extracted and made as one of the features columns The target feature was extracted from the dataframe and made as the target columns
X = []
#Appending the positive corr into X 
for col in df:
        if df['rent'].corr(df[col]) > 0 and df['rent'].corr(df[col]) < 1 :
                X.append(col)


y = df['rent']

# print(X.shape)


x_train, x_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=6)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

model = LinearRegression()
model.fit(x_train, y_train)
y_predict = model.predict(x_test)
pickle.dump(model, open('model.pkl', 'wb'))

# print(model.coef_)

sample_test = [[1, 1, 620, 16, 1, 98,0, 1, 0, 0, 1, 1, 0]]

print(f"Rent Predicted: {model.predict(sample_test)}")

plt.scatter(y_test, y_predict)
plt.plot(range(20000), range(20000))
plt.xlabel("Predicted Score")
plt.ylabel("Actual Score")
plt.show()

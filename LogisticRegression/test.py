import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt



#import dataset 
data = pd.read_csv("LogisticRegression/cancer.csv")
data.drop(['Unnamed: 32',"id"], axis=1, inplace=True)
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
y = data.diagnosis.values
x = data.drop(['diagnosis'], axis=1)



# train test split
from sklearn.model_selection import train_test_split
from sklearn import linear_model

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)



logreg = linear_model.LogisticRegression(random_state = 42,max_iter= 150)
print("test accuracy: {} ".format(logreg.fit(x_train, y_train).score(x_test, y_test)))
print("train accuracy: {} ".format(logreg.fit(x_train, y_train).score(x_train, y_train)))

print(logreg.coef_)




from logisticRegression import LogisticRegression



model = LogisticRegression(x_train, y_train, classifierTreshhold = 0.5, costfunctionName = 'Sigmoid', scalerName = 'StdZ', optimizerName= 'Adam' , lambdaRegularization= 1, ifDetail = 'true')
model.trainModel()
print(model.weight)
model.test(x_test, y_test)


#import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report,mean_squared_error,r2_score,confusion_matrix
from xgboost import XGBClassifier

#load the dataset
data=pd.read_csv("F:\\NITTTR sumeer training\\Project\\Dataset.csv")

#Check for null values
#print(data.isnull().sum())

#Convert the data type into datetime
data['date'] = pd.to_datetime(data['date'])

#Check the unique values in columns of dataset
#print(data['weather'].nunique())

df = data.drop('date',axis=1)
X = df.iloc[:,:-1].values               #Independent variables
Y = df.iloc[:,-1].values.reshape(-1,1)  #Target Column


LE=LabelEncoder()
Y[:,0] = LE.fit_transform(Y[:,0])
Y = Y.astype('int')

#Feature Scaling
sc1=StandardScaler()
X = sc1.fit_transform(X)

#Splitting the data for Training and Testing
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.3,random_state=2)


Y_train= Y_train.ravel()
xgb = XGBClassifier(n_estimators=100, max_depth=4,eta=0.1, subsample=1, colsample_bytree=1)
xgb.fit(X_train,Y_train)
Y_pred = xgb.predict(X_test)

accu = accuracy_score(Y_test, Y_pred)
print('Accuracy (XGB Classifier):',accu)
r2_score = r2_score(Y_test, Y_pred)
print("R2 score (XGB Classifier): ",r2_score )
rmse = sqrt(mean_squared_error(Y_test, Y_pred))
print('RMSE (XGB Classifier): ',rmse)
print('Precision:', precision_score(Y_test, y_pred=Y_pred,average='micro'))
print('Recall:',recall_score(Y_test, y_pred=Y_pred,average='micro'))
print('F1 Score:',f1_score(Y_test, y_pred=Y_pred,average='micro'))
print('Report:',classification_report(Y_test, y_pred=Y_pred))


# XGBoost (different learning rate)
learning_rate_range = np.arange(0.01, 1, 0.05)
test_XG = [] 
train_XG = []
for lr in learning_rate_range:
    xgb_classifier = XGBClassifier(eta = lr)
    xgb_classifier.fit(X_train, Y_train)
    train_XG.append(xgb_classifier.score(X_train, Y_train))
    test_XG.append(xgb_classifier.score(X_test, Y_test))

fig = plt.figure(figsize=(10, 7))
plt.plot(learning_rate_range, train_XG, c='orange', label='Train')
plt.plot(learning_rate_range, test_XG, c='m', label='Test')
plt.xlabel('Learning rate')
plt.xticks(learning_rate_range)
plt.ylabel('Accuracy score')
plt.ylim(0.75, 1.1)
plt.legend(prop={'size': 12}, loc=2)
plt.title('Accuracy score vs. Learning rate of XGBoost', size=14)
plt.show()



confusion_mat = confusion_matrix(Y_test,Y_pred)
sns.heatmap(confusion_mat,cbar=True,annot=True,cmap = "PiYG",vmin=0, vmax=150,fmt ='g',xticklabels=['drizzle' ,'fog','rain' ,'snow','sun'  ],yticklabels=['drizzle' ,'fog','rain' ,'snow','sun'  ])
plt.show()

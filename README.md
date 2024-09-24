# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Import Necessary Libraries and Load Data

Split Dataset into Training and Testing Sets

Train the Model Using Stochastic Gradient Descent (SGD)

Make Predictions and Evaluate Accuracy

Generate Confusion Matrix

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: PRIYADHARSHINI S
RegisterNumber:  212223240129

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix

iris=load_iris()

df=pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target']= iris.target
print(df.head())

X=df.drop('target',axis=1)
y=df['target']

X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2, random_state=42)

sgd_clf=SGDClassifier(max_iter=1000, tol=1e-3)

sgd_clf.fit(X_train,y_train)

y_pred =sgd_clf.predict(X_test)

accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy:.3f}")

cm=confusion_matrix(y_test, y_pred)
print("confusion Matrix:")
print(cm)
*/
```

## Output:

![Screenshot 2024-09-24 172400](https://github.com/user-attachments/assets/bbec5d7a-bd91-4ed2-a804-d832f1b4d21a)

![Screenshot 2024-09-24 172408](https://github.com/user-attachments/assets/35b69c63-c7e4-4cfa-9464-650470e200b5)

![Screenshot 2024-09-24 172415](https://github.com/user-attachments/assets/8bb140d0-28a7-429d-907d-f2a0708566c0)


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.

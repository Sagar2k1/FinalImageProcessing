from sklearn.model_selection import train_test_split
import cv2
import pandas as pd
import numpy as np


df = (pd.read_csv('traffic_sign.csv',header=0))[['file','label']]
X = df['file']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

def processing(X, y):
    arr = []
    for i in X:
        img = cv2.imread(i)
        img = cv2.resize(img, (180,180))
        arr.append(img)
    arr = np.array(arr)
    arr = arr.reshape(arr.shape[0],-1)
    arr = arr.astype(np.float32)
    X = arr
    y = y.values
    y = y.reshape(y.shape[0],-1).astype(np.float32)
    return X, y

X_train, y_train = processing(X_train, y_train)
X_test, y_test = processing(X_test, y_test)

print(X_train.shape)
print(y_train.shape)
knn = cv2.ml.KNearest_create()
knn.train(X_train, cv2.ml.ROW_SAMPLE, y_train)
ret, res, neighbours, distance = knn.findNearest(X_test, 5) 
knn.save('ML_OpenCV_Model')
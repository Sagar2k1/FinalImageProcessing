import numpy as np
import pandas as pd
import keras
import cv2
import os

path = '95c0e06a-dff4-4af1-a44f-c6f7c181e4fd'
dataset = os.listdir(path)
list = {
    'file': [],
    'type': [],
    'label': []
}

for label in os.listdir(path):
    for file in os.listdir(os.path.join(path,label)):
        file_name = os.path.splitext(os.path.splitext(file)[0])[0]
        typesign = file_name.split('_')[-1]
        list['file'].append(path+'/'+label+'/'+file)
        list['label'].append(label)
        list['type'].append(typesign)

df = pd.DataFrame(list)
df.to_csv('traffic_sign.csv',index=0)



import os

from PIL import Image
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score

def file_list(start_dir):
    """Generate file list in directory"""
    file_list = []
    for root, dirs, files in os.walk(start_dir):
        for f in files:
            if f[0] != '.':
                file_list.append(f)
    return file_list

X = []
y = []

positive_list = file_list('./train_game')
negative_list = file_list('./train_non-game')

for f in positive_list[0:-100]:
    im = np.array(Image.open(os.path.join('./train_game', f)))
    result = []
    for i in range(0, 3):
        result.extend(im[:,:,i].flatten().tolist())
    X.append(result)
    y.append(1)

for f in negative_list[0:-100]:
    im = np.array(Image.open(os.path.join('./train_non-game', f)))
    result = []
    for i in range(0, 3):
        result.extend(im[:,:,i].flatten().tolist())
    X.append(result)
    y.append(0)

model = LogisticRegression()
model = model.fit(X, y)

validate = []
for f in negative_list[len(negative_list)-100:len(negative_list)]:
    im = np.array(Image.open(os.path.join('./train_non-game', f)))
    result = []
    for i in range(0, 3):
        result.extend(im[:,:,i].flatten().tolist())
    validate.append(result)

print(model.predict_proba(validate))
print(model.predict(validate))

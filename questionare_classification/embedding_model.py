from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import string
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from create_vectors import create_vector

data = 'LabelledData (1).txt'
df = pd.read_table(data,  sep = ',,,',names=['text','label'],index_col=False)
df.label = df.label.str.strip()

classes = list(df.label.unique())
text = list(df.text.values)
X = [create_vector(i) for i in text]
encoder = LabelEncoder()
encoder.fit(classes)
Y = encoder.transform([i for i in list(df.label.values)])
labels = classes
logreg = LogisticRegression(multi_class='multinomial',solver='lbfgs')
logreg.fit(X, Y)
y_pred = logreg.predict(X)
print(classification_report(Y, y_pred,target_names=labels))

joblib.dump(logreg, 'model/trained_embedding_model.pkl') 

print("Saved model to disk")
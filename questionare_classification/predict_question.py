from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.externals import joblib
import warnings
import pandas as pd
warnings.filterwarnings("ignore", category=DeprecationWarning)
data = 'LabelledData (1).txt'
df = pd.read_table(data,  sep = ',,,',names=['text','label'],index_col=False)
df.label = df.label.str.strip()
from create_vectors import create_vector
def predict_question(x,logreg):
    encoder = LabelEncoder()
    classes = list(df.label.unique())
    encoder.fit(classes)
    if len(x)==1:
        vector = create_vector(x[0].lower())
        vector = [vector]
    else:
        vector = [create_vector(i.lower()) for i in x]
    preds = logreg.predict(vector)
    transformed_vector = encoder.inverse_transform(preds)
    return transformed_vector
if __name__ == '__main__':
    print("enter the questions you want to predict SEPERATED BY ','")
    inputs = [i for i in input().split(',')]
    clf = joblib.load('model/trained_embedding_model.pkl')
    print(predict_question(inputs, clf))
import streamlit as st
import numpy as np
from sklearn import datasets
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name=="Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()

    X = data.data
    y = data.target
    return X,y,data
def add_parameter_ui(clf_name):
    params = dict()
    if clf_name =="KNN":
        K = st.sidebar.slider("K",1,15)
        params["K"] = K
    elif clf_name =="SVM":
        C=st.sidebar.slider("C",0.01,10.0)
        params["C"] = C
    else:
        max_depth = st.sidebar.slider("max_depth",2,15)
        n_estimators = st.sidebar.slider("n_estimators",1,100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators  
    return params    
def get_classifier(clf_name, params):
    if clf_name =="KNN":
        clf=KNeighborsClassifier(n_neighbors=params['K'])
    elif clf_name=="SVM":
        clf= SVC(C=params["C"])
    else:
        clf = RandomForestClassifier(n_estimators = params["n_estimators"],
                                     max_depth = params["max_depth"],
                                     random_state=1234)
    return clf                              

st.title('Streamlit Example')
st.write("""
# Explore different classifier and datasets.
Which one is the best?
""")

dataset_name = st.sidebar.selectbox('Select Dataset', ('Iris','Breast Cancer', 'Wine'))

classifier_name = st.sidebar.selectbox("select Classifier",("KNN","SVM","Random Forest"))

X,y,data = get_dataset(dataset_name)

df = pd.DataFrame(data=data.data, columns=data.feature_names, index = data.target)
st.dataframe(df)
st.write("shape of dataset", X.shape)
st.write("number of classes", len(np.unique(y)))
params = add_parameter_ui(classifier_name)


contents = st.text_input('Enter some text')

st.write("# "+contents)
                                 
clf = get_classifier(classifier_name, params)

X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size=0.2, random_state=1234)
clf.fit(X_train, Y_train)
y_pred=clf.predict(X_test)
acc = accuracy_score(Y_test, y_pred)
st.write(f"classifier = {classifier_name}")
st.write(f"accuracy = {acc}")

import matplotlib.pyplot as plt

fig = plt.figure()
plt.scatter(X_train[:,0], X_train[:,1])
st.pyplot(fig)
# st.button("BUTTON!")

# st.text_input('Enter some text')

# st.color_picker('Pick a color')

# st.file_uploader('File uploader')
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,recall_score,precision_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot  as plt
from xgboost import XGBClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier

st.set_option('deprecation.showPyplotGlobalUse', False)
st.header("Bài tập PCA")
"Sinh viên thực hiện: Hồ Hồng Hà"
"MSSV: 20520480"
st.subheader("Dataset")
df = pd.DataFrame()
value = pd.DataFrame()
data = pd.DataFrame()
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as  and write to local disk:
    df=pd.read_csv(uploaded_file)
    st.write(df)
    value = df['Class']
    value = value.to_numpy()
    data = df.drop(columns= ['Class'],axis=0)

    features = data.columns.to_numpy() 
    input_features = []
    st.subheader("Input Features")
    cols = st.columns(4)
    for i in range(len(features)):
        cbox = cols[int(i/len(features)*4)].checkbox(features[i])
        if cbox:
            input_features.append(features[i])

    encs = []
    X = np.array([])
    enc_idx = -1  
    for feature in input_features:
        x = df[feature].to_numpy().reshape(-1, 1)
        if (df.dtypes[feature] == 'object'):
            encs.append(StandardScaler())
            enc_idx += 1
            x = encs[enc_idx].fit_transform(x).toarray()
        if len(X)==0:
            X = x
        else:
            X = np.concatenate((X, x), axis=1)
    input_features = X

value[value==2] = 0
value[value==4] = 1
st.subheader("Output: Class")

Category2 = st.columns([5,1,3,15])
kfold = Category2[0].checkbox("KFold")

if kfold:
    Category2[1].write("K")
    k_value = Category2[2].text_input("K",label_visibility="collapsed")
button = st.button("Run")
if button:
    model = LogisticRegression()
    acc_score_final = pd.DataFrame(columns = ['Recall','Precision','F1_Score'])
    if kfold:
        model = LogisticRegression()
        acc_score = pd.DataFrame(columns = ['Recall','Precision','F1_Score'])
        st.write("Training using KFold with n_splits = ",k_value)
        kf = KFold(n_splits=int(k_value))
        for train_index, test_index in kf.split(input_features,value):
            model1 = LogisticRegression()
            data_train, data_test = input_features[train_index], input_features[test_index]
            value_train, value_test = value[train_index], value[test_index]
           
            model1.fit(data_train,value_train)
            values_pred = model1.predict(data_test)
            recall = recall_score(value_test, values_pred, average='weighted') 
            precision = precision_score(value_test, values_pred, average='weighted')
            f1 = f1_score(value_test, values_pred, average='weighted')
            acc_score.loc[len(acc_score.index)] = [recall,precision,f1]

            model2 = svm.SVC()
            acc_score1 = pd.DataFrame(columns = ['Recall','Precision','F1_Score'])
            model2.fit(data_train,value_train)
            values_pred = model2.predict(data_test)
            recall = recall_score(value_test, values_pred, average='weighted') 
            precision = precision_score(value_test, values_pred, average='weighted')
            f1 = f1_score(value_test, values_pred, average='weighted')
            acc_score1.loc[len(acc_score1.index)] = [recall,precision,f1]  

            model3 = DecisionTreeClassifier()
            acc_score2 = pd.DataFrame(columns = ['Recall','Precision','F1_Score'])
            model3.fit(data_train,value_train)
            values_pred = model3.predict(data_test)
            recall = recall_score(value_test, values_pred, average='weighted') 
            precision = precision_score(value_test, values_pred, average='weighted')
            f1 = f1_score(value_test, values_pred, average='weighted')
            acc_score2.loc[len(acc_score2.index)] = [recall,precision,f1]  

            acc_score3 = pd.DataFrame(columns = ['Recall','Precision','F1_Score'])
            model4 = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic')
            model4.fit(data_train,value_train)
            values_pred = model4.predict(data_test)
            recall = recall_score(value_test, values_pred, average='weighted') 
            precision = precision_score(value_test, values_pred, average='weighted')
            f1 = f1_score(value_test, values_pred, average='weighted')
            acc_score3.loc[len(acc_score3.index)] = [recall,precision,f1]  

        f1_sc = acc_score['F1_Score'].mean()
        pr_sc = acc_score['Precision'].mean()
        re_sc = acc_score['Recall'].mean()
        acc_score_final.loc[len(acc_score_final.index)] = [f1_sc,pr_sc,re_sc]

        f1_sc = acc_score1['F1_Score'].mean()
        pr_sc = acc_score1['Precision'].mean()
        re_sc = acc_score1['Recall'].mean()
        acc_score_final.loc[len(acc_score_final.index)] = [f1_sc,pr_sc,re_sc]

        f1_sc = acc_score2['F1_Score'].mean()
        pr_sc = acc_score2['Precision'].mean()
        re_sc = acc_score2['Recall'].mean()
        acc_score_final.loc[len(acc_score_final.index)] = [f1_sc,pr_sc,re_sc]

        f1_sc = acc_score3['F1_Score'].mean()
        pr_sc = acc_score3['Precision'].mean()
        re_sc = acc_score3['Recall'].mean()
        acc_score_final.loc[len(acc_score_final.index)] = [f1_sc,pr_sc,re_sc]
    
    st.write(acc_score_final)
    labels = ['LogisticRegression', 'SVM', 'DecisionTreeClassifier', 'XGBClassifier']
    fig, ax = plt.subplots()
    x = np.arange(len(labels))
    ax = acc_score_final[['Recall','Precision','F1_Score']].plot.bar(stacked=False)
    ax.set_title('Chart')
    ax.set_ylabel('Score')
    ax.set_xticks(x, labels)

    st.pyplot()

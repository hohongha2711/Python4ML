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

st.set_option('deprecation.showPyplotGlobalUse', False)
st.header("Bài tập tìm số chiều tối ưu để độ chính xác F1 score cao nhất với phương pháp KFold=5")
"Sinh viên thực hiện: Hồ Hồng Hà"
"MSSV: 20520480"
st.subheader("Dataset")
df = pd.DataFrame()
value = pd.DataFrame()
data = pd.DataFrame()
num_feature = 0
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as  and write to local disk:
    df=pd.read_csv(uploaded_file)
    st.write(df)
    value = df['Wine']
    value = value.to_numpy()
    data = df.drop(columns= ['Wine'],axis=0)
    
    features = data.columns.to_numpy()
    input_features = data.values
    num_feature = len(features)
    sc=StandardScaler()
    input_features = sc.fit_transform(input_features)

st.subheader("Output: Wine")
Category2 = st.columns([1,3,20])
Category2[0].write("K")
k_value = Category2[1].text_input("K",label_visibility="collapsed")
button = st.button("Run")
if button:
    num_features = []
    acc_score_avg = pd.DataFrame(columns = ['Recall','Precision','F1_Score'])
    for i in range(1,num_feature):
        pca = PCA(n_components=int(i))
        pca.fit(input_features)
        input_feature = pca.transform(input_features)
        model = LogisticRegression()
        recall = 0
        precision = 0
        f1 = 0
        num_features.append(i)
        #acc_score = pd.DataFrame(columns = ['Recall','Precision','F1_Score']
        kf = KFold(n_splits=int(k_value))
        for train_index, test_index in kf.split(input_feature,value):
            data_train, data_test = input_feature[train_index], input_feature[test_index]
            value_train, value_test = value[train_index], value[test_index]
            pca.fit(data_train)
            data_train = pca.transform(data_train)
            data_test = pca.transform(data_test)
            model.fit(data_train,value_train)
            values_pred = model.predict(data_test)
            recall += recall_score(value_test, values_pred, average='weighted') 
            precision += precision_score(value_test, values_pred, average='weighted')
            f1 += f1_score(value_test, values_pred, average='weighted')
            
            #acc_score.loc[len(acc_score.index)] = [recall,precision,f1]
        
        acc_score_avg.loc[len(acc_score_avg.index)] = [recall/int(k_value),precision/int(k_value),f1/int(k_value)]
    st.write(acc_score_avg)
    st.write("Max F1-Score = {} with Num_Feature = {}".format(acc_score_avg['F1_Score'].max(),acc_score_avg['F1_Score'].idxmax()+1))
    fig, ax = plt.subplots()
    ax = acc_score_avg[['Recall','Precision','F1_Score']].plot.bar(stacked=False)
    plt.title('Chart')
    plt.ylabel('Score')
    st.pyplot()

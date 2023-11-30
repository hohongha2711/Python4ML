import streamlit as st
import pandas as pd
import numpy as np
#import matplotlib.pyplot  as plt
#from collections.abc import MutableMapping

st.set_option('deprecation.showPyplotGlobalUse', False)
st.header("Logistic Regression")
"Sinh viên thực hiện: Hồ Hồng Hà"
"MSSV: 20520480"
'''st.subheader("Dataset")
df = pd.DataFrame()
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as  and write to local disk:
    df=pd.read_csv(uploaded_file)
    st.write(df)

st.subheader("Input feature")
Category = st.columns(4)
op1 = Category[0].checkbox("Age")
op2 = Category[1].checkbox("EstimatedSalary")

st.subheader("Output: Purchased")
st.subheader("Train/Test split")

Category1 = st.columns([2,2,10])
Category1[0].write("Test_size")
t = Category1[1].text_input("Train/Test split",label_visibility="collapsed")


Category2 = st.columns([5,1,3,15])
kfold = Category2[0].checkbox("KFold")

if kfold:
    Category2[1].write("K")
    k_value = Category2[2].text_input("K",label_visibility="collapsed")
button = st.button("Run")
if button:
    data_feature =df.drop(columns='Purchased',axis=0)
    values = df['Purchased'] 
    if not op1:
        data_feature = data_feature.drop(columns='Age',axis=0) 
    if not op2:
        data_feature = data_feature.drop(columns='EstimatedSalary',axis=0)
    s_scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(s_scaler.fit_transform(data_feature),columns=data_feature.columns)
    model = LogisticRegression()
    acc_score = pd.DataFrame(columns = ['Recall','Precision','F1_Score'])
    if kfold:
        st.write("Training using KFold with n_splits = ",k_value)
        kf = KFold(n_splits=int(k_value),shuffle=True)
        for train_index , test_index in kf.split(df_scaled):
            df_scaled_train , df_scaled_test = df_scaled.iloc[train_index,:],df_scaled.iloc[test_index,:]
            values_train , values_test = values.iloc[train_index] , values.iloc[test_index]
            model.fit(df_scaled_train,values_train)
            values_pred = model.predict(df_scaled_test)
            recall = recall_score(values_test, values_pred, average='weighted') 
            precision = precision_score(values_test, values_pred, average='weighted')
            f1 = f1_score(values_test, values_pred, average='weighted')
            acc_score.loc[len(acc_score.index)] = [recall,precision,f1]
    else:
        st.write("Training using Tran_test_split with Test_size = ",t)
        df_scaled_train,df_scaled_test,values_train,values_test = train_test_split(df_scaled, values, test_size=float(t))
        model.fit(df_scaled_train,values_train)
        values_pred = model.predict(df_scaled_test)
        recall = recall_score(values_test, values_pred, average='weighted') 
        precision = precision_score(values_test, values_pred, average='weighted')
        f1 = f1_score(values_test, values_pred, average='weighted')
        acc_score.loc[len(acc_score.index)] = [recall,precision,f1]
    st.write(acc_score)
    fig, ax = plt.subplots()
    ax = acc_score[['Recall','Precision','F1_Score']].plot.bar(stacked=False)
    ax.set_title('Chart')
    ax.set_ylabel('Score')
    st.pyplot()


            '''
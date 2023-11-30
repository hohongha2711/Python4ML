import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.preprocessing import OneHotEncoder

st.set_option('deprecation.showPyplotGlobalUse', False)
st.header("Linear Regression")
"Sinh viên thực hiện: Hồ Hồng Hà"
"MSSV: 20520480"
st.subheader("Dataset")
df = pd.DataFrame()
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as  and write to local disk:
    df=pd.read_csv(uploaded_file)
    st.write(df)

st.subheader("Input feature")
Category = st.columns(4)
op1 = Category[0].checkbox("R&D Spend")
op2 = Category[1].checkbox("Administration")
op3 = Category[2].checkbox("Marketing Spend")
op4 = Category[3].checkbox("State")

st.subheader("Output: Profit")
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
    error = pd.DataFrame(columns=["MSE","MAE"])
    data_feature = pd.DataFrame(df)
    data_feature = data_feature.drop(columns=['Profit','State'],axis=0)
    value = df['Profit']
    if not op1:
        data_feature = data_feature.drop(columns='R&D Spend',axis=0) 
    if not op2:
        data_feature = data_feature.drop(columns='Administration',axis=0) 
    if not op3:
        data_feature = data_feature.drop(columns='Marketing Spend',axis=0) 
    if op4:
        state = df['State'].to_numpy().reshape(-1,1)
        onehot_encoder = OneHotEncoder()
        state = onehot_encoder.fit_transform(state).toarray()
        data_feature = np.hstack([data_feature, state])

    data_feature = pd.DataFrame(data_feature)
    st.write()
    if kfold:
        st.write("Training using KFold with n_splits = ",k_value)
        kf = KFold(n_splits=int(k_value))
        for train_index, test_index in kf.split(data_feature,value):
            data_train, data_test = data_feature.iloc[train_index], data_feature.iloc[test_index]
            value_train, value_test = value.iloc[train_index], value.iloc[test_index]
            reg = LinearRegression().fit(data_train, value_train)
            value_pred = reg.predict(data_test)
            MAE = mean_absolute_error(value_test,value_pred)
            MSE = mean_squared_error(value_test,value_pred)
            error.loc[len(error.index)] = [MSE,MAE]

        st.write(error)
        
        st.write("mean_squared_error chart")
        st.bar_chart(error["MSE"])

        st.write("mean_absolute_error chart")
        st.bar_chart(error["MAE"])      
    else:
        st.write("Training using Tran_test_split with Test_size = ",t)
        data_train, data_test, value_train, value_test = train_test_split(data_feature, value, test_size=float(t))
        reg = LinearRegression().fit(data_train, value_train)
        value_pred = reg.predict(data_test)
        MAE = mean_absolute_error(value_test,value_pred)
        MSE = mean_squared_error(value_test,value_pred)
        error.loc[len(error.index)] = [MSE,MAE]
        st.write(error)
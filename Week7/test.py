from turtle import up
from unittest import result
import streamlit as st
from PIL import Image
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
st.markdown(
""" 
# Đây là bài tutorial
## 1. Giới thiệu streamlit
### 1.2. Giới thiệu chung
### 1.2. Cài đặt
## 2. Các thành phần cơ bản của giao diện
""")


uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as  and write to local disk:
    bytes_data = uploaded_file.getvalue()
    #st.write(bytes_data)
    img_path = 'data/'+uploaded_file.name
    with open(img_path, "wb") as f:
        f.write(bytes_data)
    img = cv2.imread(img_path, 0)
    #Filter ảnh
    filter = np.array(
        [[-1,0,1],
        [-2,0,2],
        [-1,0,1]])

    result1 = cv2.filter2D(img,-1,filter)
    filter = np.array(
        [[1,0,-1],
        [2,0,-2],
        [1,0,-1]])

    result2 = cv2.filter2D(img,-1,filter)

    filter = np.array(
        [[-1,-2,-1],
        [0,0,0],
        [1,2,1]])

    result3 = cv2.filter2D(img,-1,filter)

    filter = np.array(
        [[1,2,1],
        [0,0,0],
        [-1,-2,-1]])

    result4 = cv2.filter2D(img,-1,filter)

    result = np.add(result1,np.add(result2,np.add(result3,result4)))

    col1,col2 = st.columns(2)
    with col1:
        st.title("Ảnh gốc")
        st.image(img)
    with col2:
        st.title("Ảnh sau filter")
        st.image(result)
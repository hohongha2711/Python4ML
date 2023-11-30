import streamlit as st
from PIL import Image


st.markdown(
""" 
# Đây là bài tutorial
## 1. Giới thiệu streamlit
### 1.2. Giới thiệu chung
### 1.2. Cài đặt
## 2. Các thành phần cơ bản của giao diện
""")

'''a_value = st.text_input("Nhập a: ")
b_value = st.text_input("Nhập b: ")

operator = st.selectbox("Chọn phép toán", ['Cộng', 'Trừ', 'Nhân', 'Chia'])

button = st.button("Tính")

if button:
    if operator == 'Cộng':
        st.text_input("Kết quả: ",float(a_value) + float(b_value))
    elif operator == 'Trừ':
        st.text_input("Kết quả: ",float(a_value) - float(b_value))
    elif operator == 'Nhân':
        st.text_input("Kết quả: ",float(a_value) * float(b_value))
    elif operator == 'Chia':
        st.text_input("Kết quả: ",float(a_value) / float(b_value))'''

with col1:
   st.header("A cat")
   st.image("https://static.streamlit.io/examples/cat.jpg")
   st.header("A dog")
   st.image("https://static.streamlit.io/examples/dog.jpg")
   st.header("An owl")
   st.image("https://static.streamlit.io/examples/owl.jpg")

with col2:
   st.header("A dog")
   st.image("https://static.streamlit.io/examples/dog.jpg")
   st.header("A cat")
   st.image("https://static.streamlit.io/examples/cat.jpg")
   st.header("An owl")
   st.image("https://static.streamlit.io/examples/owl.jpg")
with col3:
   st.header("An owl")
   st.image("https://static.streamlit.io/examples/owl.jpg")
   st.header("A cat")
   st.image("https://static.streamlit.io/examples/cat.jpg")
   st.header("A dog")
   st.image("https://static.streamlit.io/examples/dog.jpg")

   cat_tab, dog_tab, owl_tab = st.tabs(["Cat", "Dog", "Owl"])

with cat_tab:
   st.header("A cat")
   st.image("https://static.streamlit.io/examples/cat.jpg", width=200)

with dog_tab:
   st.header("A dog")
   st.image("https://static.streamlit.io/examples/dog.jpg", width=200)

with owl_tab:
   st.header("An owl")
   st.image("https://static.streamlit.io/examples/owl.jpg", width=200)

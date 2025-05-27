import pickle
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# ----------------------- Load Model -----------------------
model = pickle.load(open('model_prediksi_harga_mobil.sav', 'rb'))

# ----------------------- Sidebar -----------------------
st.sidebar.title("Navigasi")
menu = st.sidebar.radio("Pilih Halaman:", ["Beranda", "Dataset", "Visualisasi", "Prediksi"])

# ----------------------- Beranda -----------------------
if menu == "Beranda":
    st.title("🚗 Prediksi Harga Mobil")
    st.image("IMG_6684.JPG", use_column_width=True)
    st.markdown("""
    Selamat datang di aplikasi **Prediksi Harga Mobil**!  
    Aplikasi ini memanfaatkan model **Machine Learning (Linear Regression)** untuk memprediksi harga mobil berdasarkan:
    - Highway-mpg
    - Curb Weight
    - Horsepower

    Dibuat menggunakan **Streamlit**.
    """)

# ----------------------- Dataset -----------------------
elif menu == "Dataset":
    st.title("📊 Dataset Mobil")

    df1 = pd.read_csv('CarPrice.csv')
    st.dataframe(df1.head(10))

    st.markdown(f"Jumlah data: **{df1.shape[0]}** baris, **{df1.shape[1]}** kolom")

# ----------------------- Visualisasi -----------------------
elif menu == "Visualisasi":
    st.title("📈 Visualisasi Data")

    df1 = pd.read_csv('CarPrice.csv')

    st.markdown("### Pilih Jenis Grafik:")
    option = st.selectbox("Pilih:", ["Highway-mpg", "Curb Weight", "Horsepower"])

    if option == "Highway-mpg":
        chart = alt.Chart(df1).mark_line(color="green").encode(
            x='car_ID',
            y='highwaympg'
        ).properties(title="Grafik Highway-mpg")
        st.altair_chart(chart, use_container_width=True)

    elif option == "Curb Weight":
        chart = alt.Chart(df1).mark_line(color="orange").encode(
            x='car_ID',
            y='curbweight'
        ).properties(title="Grafik Curb Weight")
        st.altair_chart(chart, use_container_width=True)

    elif option == "Horsepower":
        chart = alt.Chart(df1).mark_line(color="red").encode(
            x='car_ID',
            y='horsepower'
        ).properties(title="Grafik Horsepower")
        st.altair_chart(chart, use_container_width=True)

# ----------------------- Prediksi -----------------------
elif menu == "Prediksi":
    st.title("🧮 Prediksi Harga Mobil")

    col1, col2, col3 = st.columns(3)
    with col1:
        highwaympg = st.number_input('Highway-mpg', min_value=0, max_value=100, value=30)
    with col2:
        curbweight = st.number_input('Curb Weight', min_value=0, max_value=6000, value=2500)
    with col3:
        horsepower = st.number_input('Horsepower', min_value=0, max_value=500, value=150)

    if st.button('Prediksi Harga'):
        car_prediction = model.predict([[highwaympg, curbweight, horsepower]])
        harga_mobil_float = float(car_prediction[0])
        harga_mobil_formatted = f"${harga_mobil_float:,.2f}"
        st.success(f"💰 Prediksi Harga Mobil: **{harga_mobil_formatted}**")

        st.balloons()

# ----------------------- Footer -----------------------
st.markdown("""
<hr>
<div style='text-align: center'>
    Dibuat oleh: <b>Richo Novian Saputra</b> | Praktikum Kecerdasan Buatan - Pertemuan 13
</div>
""", unsafe_allow_html=True)
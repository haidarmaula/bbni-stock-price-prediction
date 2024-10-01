import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Fungsi untuk memuat data
@st.cache_data
def load_data(file_path):
    """Memuat dataset dari file CSV."""
    df = pd.read_csv(file_path)
    return df

# Fungsi untuk menyiapkan data (preprocessing)
def preprocess_data(df):
    """Melakukan scaling dan membagi data menjadi fitur dan target."""
    columns = ['Open', 'High', 'Low', 'Volume']
    scaler = RobustScaler()
    df_scaled = scaler.fit_transform(df[columns])
    df_scaled = pd.DataFrame(df_scaled, columns=columns)
    return df_scaled, df['Close'], scaler

# Fungsi untuk melatih model regresi linier
def train_model(X, y):
    """Melatih model regresi linier dengan data pelatihan."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    return model, r2, X_test, y_test, y_pred

# Fungsi untuk memprediksi harga saham berdasarkan input pengguna
def predict_closing_price(model, scaler, open_price, high_price, low_price, volume):
    """Memprediksi harga penutupan saham berdasarkan input pengguna."""
    user_input = pd.DataFrame([[open_price, high_price, low_price, volume]], 
                              columns=['Open', 'High', 'Low', 'Volume'])
    user_input_scaled = scaler.transform(user_input)
    predicted_price = model.predict(user_input_scaled)
    return predicted_price[0]

# Fungsi untuk menampilkan halaman prediksi saham
def page_prediction(model, scaler):
    """Menampilkan halaman untuk input prediksi saham."""
    st.image("images/image.png")
    st.title("Prediksi Harga Saham BBNI 💸💰")
    st.write("""
    Masukkan harga saham untuk memprediksi harga penutupan berdasarkan model yang telah dilatih.
    """)

    # Form input untuk pengguna
    st.subheader("Masukkan Data Saham")
    open_price = st.number_input("Harga Pembukaan (Open)", min_value=0.0, value=1000.0)
    high_price = st.number_input("Harga Tertinggi (High)", min_value=0.0, value=1000.0)
    low_price = st.number_input("Harga Terendah (Low)", min_value=0.0, value=1000.0)
    volume = st.number_input("Volume Perdagangan", min_value=0.0, value=1000000.0)

    # Tombol untuk memprediksi harga
    if st.button("Prediksi Harga"):
        predicted_price = predict_closing_price(model, scaler, open_price, high_price, low_price, volume)
        st.subheader(f"Harga Penutupan Prediksi: Rp {predicted_price:,.2f}")

# Fungsi untuk menampilkan halaman evaluasi model
def page_evaluation(r2, X_test, y_test, y_pred):
    """Menampilkan halaman evaluasi performa model."""
    st.title("Evaluasi Performa Model")
    st.write("""
    Di halaman ini, Anda dapat melihat performa model regresi linier yang telah dilatih.
    """)

    # Menampilkan skor R-squared
    st.subheader(f"R-squared Score: {r2:.4f}")

    # Menampilkan perbandingan aktual vs prediksi
    st.subheader("Perbandingan Harga Aktual vs Harga Prediksi")
    df_result = pd.DataFrame({
        'Actual Price': y_test,
        'Predicted Price': y_pred
    }).reset_index(drop=True)

    st.write(df_result.head(10))

    # Visualisasi perbandingan harga aktual dan prediksi dengan Matplotlib
    st.subheader("Grafik Perbandingan Harga Aktual vs Prediksi")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(df_result['Actual Price'], label='Actual Price', color='blue', marker='o')
    ax.plot(df_result['Predicted Price'], label='Predicted Price', color='red', linestyle='--', marker='x')
    
    ax.set_title("Perbandingan Harga Aktual vs Harga Prediksi", fontsize=16)
    ax.set_xlabel("Index", fontsize=12)
    ax.set_ylabel("Harga (Rp)", fontsize=12)
    
    ax.legend(loc="upper left", fontsize=12)
    ax.grid(True)
    
    # Menampilkan plot di Streamlit
    st.pyplot(fig)

# Fungsi utama untuk navigasi halaman
def main():
    # Memuat dataset
    df = load_data('BBNI.JK.csv')

    # Menyiapkan data untuk pelatihan
    X, y, scaler = preprocess_data(df)

    # Melatih model
    model, r2, X_test, y_test, y_pred = train_model(X, y)

    # Sidebar untuk memilih halaman
    st.sidebar.title("Menu Dasbor")
    page = st.sidebar.selectbox("Pilih Halaman", ["Prediksi Harga Saham", "Evaluasi Model"])

    # Navigasi antara halaman prediksi dan evaluasi
    if page == "Prediksi Harga Saham":
        page_prediction(model, scaler)
    elif page == "Evaluasi Model":
        page_evaluation(r2, X_test, y_test, y_pred)

if __name__ == "__main__":
    main()

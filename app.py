# Nama : Muhammad Rizkhi
# NIM : 2020230005

import numpy as np
import pandas as pd
import pickle

age = float(input("Masukkan usia: "))
sex = int(input("Masukkan jenis kelamin (0 untuk laki-laki, 1 untuk perempuan): "))
bmi = float(input("Masukkan nilai BMI: "))
children = int(input("Masukkan jumlah anak: "))
smoker = int(input("Apakah perokok? (0 untuk tidak, 1 untuk ya): "))

# Proses input - dijadikan array dan di reshape
X = np.array([age, sex, bmi, children, smoker])
X = X.reshape(1, -1)

# Load model tersimpan - pickle
with open('model_uas.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Prediksi
charger_pred = loaded_model.predict(X)

# Menampilkan hasil prediksi
print("Prediksi Biaya Asuransi:", charger_pred)
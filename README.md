# Mengimpor pustaka
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Memuat dataset Iris
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['target'] = iris.target

# Memproses data
X = data.drop(columns=['target'])
y = data['target']

# Membagi data menjadi set pelatihan dan pengujian
X_latih, X_uji, y_latih, y_uji = train_test_split(X, y, test_size=0.3, random_state=42)

# Inisialisasi classifier Naive Bayes
nb = GaussianNB()

# Melatih model
nb.fit(X_latih, y_latih)

# Memprediksi pada set pengujian
y_pred = nb.predict(X_uji)

# Menghitung akurasi
akurasi = accuracy_score(y_uji, y_pred)
print(f'Akurasi: {akurasi}')

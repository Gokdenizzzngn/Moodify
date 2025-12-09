# 🎵 Moodify: Spotify Recommender System

## 📌 Proje Hakkında
Bu proje, Machine Learning Bootcamp kapsamında geliştirilmiştir. Kullanıcının seçtiği şarkının ses özelliklerini (acousticness, danceability, energy vb.) analiz ederek, ona matematiksel olarak en yakın şarkıları öneren bir "Content-Based Filtering" sistemidir.

## 🚀 Canlı Demo
Projenin çalışan halini buradan deneyebilirsiniz: [BURAYA_DEPLOY_LİNKİ_GELECEK] (gelemedi..)

## 📂 Veri Seti ve Yöntem
* **Veri Kaynağı:**  [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)
* **Kullanılan Algoritma:** K-Nearest Neighbors (KNN) - Cosine Similarity
* **Özellik Mühendisliği:** Genre Encoding, Audio Features Scaling (MinMax).

## 🛠 Kullanılan Teknolojiler
* Python 3.x
* Streamlit (Arayüz)
* Scikit-learn (Model)
* Pandas & Numpy (Veri İşleme)

## ⚙️ Kurulum (Local)
1. Repoyu klonlayın:
   `git clone https://github.com/Gokdenizzzngn/Moodify.git`
2. Gerekli paketleri yükleyin:
   `pip install -r requirements.txt`
3. Uygulamayı çalıştırın:
   `streamlit run app.py`

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from math import pi
import os
from huggingface_hub import hf_hub_download

# --- 1. AYARLAR ---
st.set_page_config(page_title="Moodify: Discovery", layout="wide")

# Sayfa Başlığı
st.markdown("<h1 style='text-align: center; color: #1DB954;'>🎧 Moodify</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Müzik Keşif ve Öneri Motoru</h4>", unsafe_allow_html=True)
st.markdown("---")


# --- AYARLAR ---
# Buraya kendi Hugging Face kullanıcı adını ve model ismini yaz
HF_REPO_ID = "Gokdenizz/spotify-recommender-models" 
HF_MODEL_FILENAME = "recommender_pipeline.pkl"




# --- 2. CLASS TANIMI ---
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

class SpotifyRecommender:
    def __init__(self, n_neighbors=11, metric='cosine'):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.model = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, algorithm='brute')
        self.scaler = MinMaxScaler()
        self.feature_matrix = None
        self.df = None
    def preprocess(self, df): pass 
    def recommend(self, song_name, n_recommendations=10): pass

# --- 3. VERİ YÜKLEME ---
@st.cache_data
def load_data():
    path = 'data/Spotify_dataset.csv'
    if os.path.exists(path):
        df = pd.read_csv(path)
        df = df.drop_duplicates(subset=['track_id']).reset_index(drop=True)
        return df
    return None

@st.cache_resource
def load_pipeline():
    try:

        # Modeli Hugging Face'den indirir ve cache'ler
        model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_MODEL_FILENAME)
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Model yüklenirken hata oluştu: {e}")

        return joblib.load('models/recommender_pipeline.pkl')
    except:

        return None

df = load_data()
recommender = load_pipeline()

if df is not None and recommender is not None:
    if recommender.df is None: recommender.df = df
    genre_col = 'track_genre' if 'track_genre' in df.columns else 'genre'

    # --- 4. SEÇİM PANELİ ---
    st.sidebar.header("🔍 Şarkı Seçimi")
    all_genres = sorted(df[genre_col].dropna().unique().astype(str))
    selected_genre = st.sidebar.selectbox("1. Tür Seçimi:", all_genres, index=0)

    if selected_genre:
        genre_songs_df = df[df[genre_col] == selected_genre].copy()
        genre_songs_df['display_name'] = genre_songs_df['track_name'] + " - " + genre_songs_df['artists']
        all_songs_sorted = sorted(genre_songs_df['display_name'].unique())
        selected_song_display = st.sidebar.selectbox("2. Şarkı Seçimi:", all_songs_sorted, index=0)

    # --- 5. HESAPLAMA ---
    if selected_song_display and st.sidebar.button("Analiz Et 🚀", type="primary"):
        song_row = genre_songs_df[genre_songs_df['display_name'] == selected_song_display].iloc[0]
        original_idx = song_row.name
        
        query_vector = recommender.feature_matrix[original_idx].reshape(1, -1)
        distances, indices = recommender.model.kneighbors(query_vector, n_neighbors=5000)
        
        same_genre_recs = []
        cross_genre_recs = []
        input_genre = song_row[genre_col]
        
        for i, neighbor_idx in enumerate(indices[0]):
            if neighbor_idx == original_idx: continue
            neighbor_song = df.iloc[neighbor_idx]
            dist_score = (1 - distances[0][i])
            rec_item = {'track_id': neighbor_song['track_id'], 'name': neighbor_song['track_name'], 'artist': neighbor_song['artists'], 'score': dist_score, 'genre': neighbor_song[genre_col]}
            if neighbor_song[genre_col] == input_genre:
                if len(same_genre_recs) < 10: same_genre_recs.append(rec_item)
            else:
                if len(cross_genre_recs) < 5: cross_genre_recs.append(rec_item)
            if len(same_genre_recs) >= 10 and len(cross_genre_recs) >= 5: break
        
        # --- GÖRSELLEŞTİRME ---
        
        # 1. ANA ŞARKI (ÖZEL TASARIM)
        # 3 Kolon: Boşluk - İçerik - Boşluk (Ortalamak için)
        _, c_mid, _ = st.columns([1.5, 2, 1.5]) 
        with c_mid:
            # HTML/CSS ile Özel Başlık Tasarımı
            st.markdown(f"""
            <div style="text-align: center; padding: 10px; border: 2px solid #1DB954; border-radius: 15px; margin-bottom: 20px;">
                <h3 style="color: #1DB954; margin:0;">{song_row['track_name']}</h2>
                <h5 style="margin:5px 0 0 0; opacity: 0.8;">🎤 {song_row['artists']}</h5>
            </div>
            """, unsafe_allow_html=True)
            
            # Kare Player (250px)
            components.iframe(f"https://open.spotify.com/embed/track/{song_row['track_id']}?utm_source=generator&theme=0", height=250)
            
            # Altına küçük not
            st.markdown("<p style='text-align: center; font-size: 12px; opacity: 0.6;'>▲ Referans Parçanız ▲</p>", unsafe_allow_html=True)

        st.markdown("---")

        # KART FONKSİYONU
        def display_compact(song_list, title, header_color):
            # Renkli Başlıklar
            st.markdown(f"<h3 style='color: {header_color}; border-bottom: 2px solid {header_color}; padding-bottom: 5px;'>{title}</h3>", unsafe_allow_html=True)
            
            if not song_list:
                st.warning("Veri yok.")
                return

            for song in song_list:
                with st.container(border=True):
                    c1, c2 = st.columns([2.5, 2.5]) # Oran ayarı
                    
                    with c1:
                        st.markdown(f"**{song['name']}**")
                        
                        # Detaylı Bilgi Bloğu
                        info_html = f"""
                        <div style="font-size: 16px; color: #555; line-height: 1.85;">
                        👤 Şarkıcı: <b>{song['artist']}</b><br>
                        🎵 Tür: <b>{song['genre']}<br>
                        📊 Uyum: <span style="color: {header_color}; font-weight: bold;">%{int(song['score']*100)}</span>
                        </div>
                        """
                        st.markdown(info_html, unsafe_allow_html=True)
                    
                    with c2:
                        embed_url = f"https://open.spotify.com/embed/track/{song['track_id']}?utm_source=generator&theme=0"
                        components.iframe(embed_url, height=90)

        # 2. ÖNERİ LİSTELERİ
        col_same, _, col_diff = st.columns([1.2, 0.5, 1.2])
        
        with col_same:
            display_compact(same_genre_recs, "🟢 Güvenli Liman (Aynı Türden Öneriler)", "#1DB954") # Yeşil
            
        with col_diff:
            display_compact(cross_genre_recs, "🚀 Yeni Ufuklar (Farklı Türden Öneriler)", "#FF6B6B") # Kırmızı/Turuncu

        # MİNİ GRAFİK
        st.markdown("---")
        _, col_g, _ = st.columns([2, 1, 2])
        with col_g:
            try:
                categories = ['Dance', 'Energy', 'Acoustic', 'Valence']
                features = recommender.feature_matrix[original_idx]
                values = [features[0], features[1], features[4], features[7]]
                values += values[:1]
                angles = [n / 4 * 2 * pi for n in range(4)]
                angles += angles[:1]
                fig, ax = plt.subplots(figsize=(2, 2), subplot_kw=dict(polar=True))
                ax.plot(angles, values, color='#1DB954', linewidth=1)
                ax.fill(angles, values, '#1DB954', alpha=0.3)
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(categories, fontsize=6)
                ax.set_yticklabels([])
                ax.spines['polar'].set_visible(False)
                st.pyplot(fig, use_container_width=False)
            except: pass
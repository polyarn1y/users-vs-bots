import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from model import train_model

MODEL_FILE = "model.pkl"
FEATURES_FILE = "feature_list.pkl"
metrics = None

st.sidebar.title("⚙️ Управление моделью")
if st.sidebar.button("🔁 Обучить модель заново"):
    with st.spinner("Обучение модели..."):
        model, feature_list, metrics = train_model()
    st.sidebar.success("✅ Модель обучена!")
else:
    if not os.path.exists(MODEL_FILE) or not os.path.exists(FEATURES_FILE):
        st.warning("Модель не найдена. Выполняется первичное обучение...")
        model, feature_list, metrics = train_model()
    else:
        model = joblib.load(MODEL_FILE)
        feature_list = joblib.load(FEATURES_FILE)
        
if metrics:
    st.sidebar.markdown("### 📊 Метрики модели")
    st.sidebar.write(f"**Accuracy:** {metrics['accuracy']:.3f}")
    st.sidebar.write(f"**ROC AUC:** {metrics['roc_auc']:.3f}")
    st.sidebar.write(f"**Precision:** {metrics['precision']:.3f}")
    st.sidebar.write(f"**Recall:** {metrics['recall']:.3f}")
    st.sidebar.write(f"**F1-score:** {metrics['f1']:.3f}")

    cm = metrics['confusion_matrix']
    st.sidebar.markdown("**Матрица ошибок**")
    st.sidebar.write(f"True Neg: {cm[0][0]}, False Pos: {cm[0][1]}")
    st.sidebar.write(f"False Neg: {cm[1][0]}, True Pos: {cm[1][1]}")
        
feature_translation = {
    "has_domain": "Имеет домен",
    "has_birth_date": "Указана дата рождения",
    "has_photo": "Имеет фотографию",
    "can_post_on_wall": "Может писать на стене",
    "can_send_message": "Может отправлять сообщения",
    "has_website": "Указан сайт",
    "gender": "Пол",
    "has_short_name": "Короткое имя",
    "has_first_name": "Указано имя",
    "has_last_name": "Указана фамилия",
    "has_nickname": "Есть никнейм",
    "has_mobile": "Указан мобильный телефон",
    "all_posts_visible": "Все посты видны",
    "audio_available": "Аудио доступно",
    "can_invite_to_group": "Можно приглашать в группы",
    "is_blacklisted": "В черном списке",
    "has_relatives": "Указаны родственники",
    "has_career": "Указана карьера",
    "marital_status": "Семейное положение",
    "city_freq": "Частота города",
    "occupation_type": "Тип занятия",
    "can_add_as_friend": "Можно добавить в друзья",
    "has_status": "Есть статус",
    "avg_views": "Среднее число просмотров",
    "avg_likes": "Среднее число лайков",
    "avg_keywords": "Среднее число ключевых слов",
    "avg_text_length": "Средняя длина текста",
    "posting_frequency_days": "Интервал между постами (в днях)",
    "reposts_ratio": "Доля репостов",
    "is_closed_profile": "Профиль закрыт",
    "has_military_service": "Служил в армии",
    "has_hometown": "Указан родной город",
    "has_universities": "Указаны вузы",
    "has_schools": "Указаны школы",
    "is_verified": "Профиль верифицирован",
    "is_confirmed": "Номер подтверждён",
    "posts_count": "Количество постов",
    "links_ratio": "Доля ссылок",
    "hashtags_ratio": "Доля хэштегов"
}

binary_features = [
    "has_mobile", "has_photo", "can_add_as_friend", "can_send_message",
    "has_status", "all_posts_visible", "is_closed_profile",
    "has_domain", "has_birth_date", "has_website", "can_post_on_wall",
    "has_short_name", "has_first_name", "has_last_name", "has_nickname",
    "audio_available", "can_invite_to_group", "is_blacklisted", "has_relatives",
    "has_career", "has_military_service", "has_hometown", "has_universities",
    "has_schools", "is_verified", "is_confirmed"
]

top_features = [
    "has_mobile", "has_photo", "can_add_as_friend", "can_send_message",
    "has_status", "all_posts_visible", "is_closed_profile",
    "occupation_type", "city_freq", "avg_views", "avg_likes",
    "avg_keywords", "avg_text_length", "posting_frequency_days", "reposts_ratio"
]

st.title("🧠 Детектор Ботов ВКонтакте")
st.write("Введите данные или выберите шаблон профиля для предсказания:")

col1, col2 = st.columns(2)
if col1.button("Сгенерировать бота"):
    st.session_state["input_df"] = pd.DataFrame([{f: 0 for f in feature_list}])
if col2.button("Сгенерировать человека"):
    st.session_state["input_df"] = pd.DataFrame([{
        f: 1 if f in binary_features else 50 for f in feature_list
    }])

if "input_df" not in st.session_state:
    st.session_state["input_df"] = pd.DataFrame([{f: 0 for f in feature_list}])
input_df = st.session_state["input_df"]

st.markdown("### 📝 Отредактируйте 15 важнейших признаков")
cols = st.columns(3)
for i, feat in enumerate(top_features):
    with cols[i % 3]:
        label = feature_translation.get(feat, feat)
        if feat in binary_features:
            input_df.at[0, feat] = st.checkbox(label, value=bool(input_df.at[0, feat]))
        elif feat == "reposts_ratio":
            input_df.at[0, feat] = st.slider(label, 0.0, 1.0, float(input_df.at[0, feat]), step=0.01)
        else:
            input_df.at[0, feat] = st.number_input(label, min_value=0.0, value=float(input_df.at[0, feat]), step=1.0)

with st.expander("📦 Показать остальные признаки"):
    remaining = [f for f in feature_list if f not in top_features]
    cols = st.columns(3)
    for i, feat in enumerate(remaining):
        with cols[i % 3]:
            label = feature_translation.get(feat, feat)
            if feat in binary_features:
                input_df.at[0, feat] = st.checkbox(label, value=bool(input_df.at[0, feat]))
            else:
                input_df.at[0, feat] = st.number_input(label, min_value=0.0, value=float(input_df.at[0, feat]), step=1.0)

if st.button("🔍 Предсказать"):
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]
    if proba > 0.3:
        prediction = 1
    if prediction == 1:
        st.error("🤖 БОТ")
    else:
        st.success("👤 ЧЕЛОВЕК")

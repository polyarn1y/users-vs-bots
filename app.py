import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# Настройка страницы
st.set_page_config(
    page_title="🤖 VK Bots vs Users Analysis",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Функция для загрузки данных с обработкой "Unknown"
@st.cache_data
def load_data():
    try:
        data = pd.read_csv('bots_vs_users.csv')

        # Заменяем "Unknown" на NaN
        data = data.replace('Unknown', np.nan)

        # Разделяем числовые и категориальные столбцы
        numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = data.select_dtypes(include=['object']).columns

        # Заполняем пропуски: числовые — средним, категориальные — модой
        for col in numerical_cols:
            data[col] = data[col].fillna(data[col].mean())
        for col in categorical_cols:
            data[col] = data[col].fillna(data[col].mode()[0])

        return data
    except FileNotFoundError:
        st.error("❌ Файл 'bots_vs_users.csv' не найден. Убедитесь, что файл находится в той же папке.")
        return None
    except Exception as e:
        st.error(f"❌ Ошибка при загрузке файла: {str(e)}")
        return None


# Функция для предобработки данных
@st.cache_data
def preprocess_data(data):
    if data is None:
        return None, None, None

    numerical_features = []
    categorical_features = []

    for col in data.columns:
        if col != 'target':
            if data[col].dtype in ['int64', 'float64']:
                numerical_features.append(col)
            else:
                categorical_features.append(col)

    model_data = data.copy()

    for col in categorical_features:
        if model_data[col].dtype == 'object':
            model_data[col] = pd.get_dummies(model_data[col], prefix=col).iloc[:, 0]
        else:
            model_data[col] = model_data[col].astype(int)

    return data, numerical_features, categorical_features


# Главная функция
def main():
    st.markdown("## 🤖 Панель анализа VK Bots vs Users")

    data = load_data()

    if data is None:
        st.stop()

    original_data, numerical_features, categorical_features = preprocess_data(data)

    if original_data is None:
        st.stop()

    # Боковая панель с навигацией через selectbox
    st.sidebar.markdown("## 🎛️ Навигация")

    analysis_section = st.sidebar.selectbox(
        "Выберите раздел анализа:",
        [
            "🏠 Обзор данных",
            "📈 Статистический анализ",
            "🔍 Исследование признаков",
            "📊 Корреляционный анализ",
            "🤖 Машинное обучение",
            "🎨 Интерактивные графики",
            "📋 Детальная информация"
        ]
    )

    # Компонент 2: Slider для фильтрации данных
    st.sidebar.markdown("### 🎚️ Фильтры данных")
    sample_size = st.sidebar.slider(
        "Размер выборки для анализа:",
        min_value=100,
        max_value=len(original_data),
        value=min(1000, len(original_data)),
        step=100
    )

    # Фильтрация данных
    filtered_data = original_data.sample(n=sample_size, random_state=42)

    # Компонент 3: Multiselect для выбора признаков
    if analysis_section in ["🔍 Исследование признаков", "📊 Корреляционный анализ"]:
        selected_features = st.sidebar.multiselect(
            "🎯 Выберите признаки для анализа:",
            options=numerical_features[:20],
            default=numerical_features[:5] if len(numerical_features) >= 5 else numerical_features
        )

    # Основной контент в зависимости от выбранного раздела
    if analysis_section == "🏠 Обзор данных":
        show_data_overview(filtered_data, numerical_features, categorical_features)

    elif analysis_section == "📈 Статистический анализ":
        show_statistical_analysis(filtered_data, numerical_features)

    elif analysis_section == "🔍 Исследование признаков":
        if 'selected_features' in locals():
            show_feature_exploration(filtered_data, selected_features)
        else:
            st.error("Выберите признаки в боковой панели")

    elif analysis_section == "📊 Корреляционный анализ":
        if 'selected_features' in locals():
            show_correlation_analysis(filtered_data, selected_features)
        else:
            st.error("Выберите признаки в боковой панели")

    elif analysis_section == "🤖 Машинное обучение":
        show_ml_analysis(original_data, numerical_features)

    elif analysis_section == "🎨 Интерактивные графики":
        show_interactive_plots(filtered_data, numerical_features, categorical_features)

    elif analysis_section == "📋 Детальная информация":
        show_detailed_info(filtered_data)

    # Дополнительная боковая панель с информацией
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ О датасете")
    st.sidebar.info(
        """
        **VK Bots vs Users Dataset**

        📊 Датасет содержит данные профилей из социальной сети VKontakte для различения реальных пользователей и ботов.

        🎯 **Целевая переменная:**
        - 0: Реальный пользователь
        - 1: Бот

        📈 **Размер:** 5,874 записи × 60 признаков

        ⚖️ **Баланс:** 50/50 (сбалансированный)
        """
    )

    # Кнопка для создания отчета
    if st.sidebar.button("📋 Создать отчет", type="secondary"):
        with st.spinner("📝 Создание отчета..."):
            report = generate_report(original_data, numerical_features, categorical_features)

            st.sidebar.success("✅ Отчет создан!")

            st.markdown("---")
            st.markdown(report)

            st.download_button(
                label="📥 Скачать отчет",
                data=report,
                file_name=f"vk_dataset_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )


def show_data_overview(data, numerical_features, categorical_features):
    st.markdown("### 📊 Обзор датасета")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📊 Общее количество записей", f"{len(data):,}")

    with col2:
        st.metric("🔢 Численных признаков", len(numerical_features))

    with col3:
        st.metric("📝 Категориальных признаков", len(categorical_features))

    with col4:
        missing_pct = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
        st.metric("❌ Пропущенных значений", f"{missing_pct:.1f}%")

    tab1, tab2, tab3 = st.tabs(["📋 Первые строки", "📊 Типы данных", "❌ Пропущенные значения"])

    with tab1:
        st.dataframe(data.head(10), use_container_width=True)

    with tab2:
        dtype_df = pd.DataFrame({
            'Колонка': data.columns,
            'Тип данных': data.dtypes.values,
            'Количество уникальных': [data[col].nunique() for col in data.columns]
        })
        st.dataframe(dtype_df, use_container_width=True)

    with tab3:
        missing_data = data.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)

        if len(missing_data) > 0:
            st.markdown(
                "**Описание:** Этот график показывает количество пропущенных значений для каждого признака в датасете.")
            fig = px.bar(
                x=missing_data.values,
                y=missing_data.index,
                orientation='h',
                title="Количество пропущенных значений по признакам",
                labels={'x': 'Количество пропусков', 'y': 'Признаки'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ В данных нет пропущенных значений!")


def show_statistical_analysis(data, numerical_features):
    st.markdown("### 📈 Статистический анализ")

    selected_feature = st.selectbox(
        "📊 Выберите признак для анализа:",
        numerical_features
    )

    if selected_feature:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                "**Описание:** Эта гистограмма показывает распределение значений выбранного признака в датасете.")
            fig = px.histogram(
                data,
                x=selected_feature,
                nbins=30,
                title=f"Распределение признака: {selected_feature}",
                color_discrete_sequence=['#4267B2']
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown(
                "**Описание:** Эта диаграмма размаха показывает разброс значений выбранного признака, включая медиану, квартили и выбросы.")
            fig = px.box(
                data,
                y=selected_feature,
                title=f"Диаграмма размаха для {selected_feature}",
                color_discrete_sequence=['#42A5F5']
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("📊 Описательная статистика")
        stats = data[selected_feature].describe()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Среднее", f"{stats['mean']:.2f}")
        with col2:
            st.metric("Медиана", f"{stats['50%']:.2f}")
        with col3:
            st.metric("Стд. отклонение", f"{stats['std']:.2f}")
        with col4:
            st.metric("Размах", f"{stats['max'] - stats['min']:.2f}")


def show_feature_exploration(data, selected_features):
    st.markdown("### 🔍 Исследование признаков")

    if not selected_features:
        st.warning("⚠️ Выберите признаки для анализа в боковой панели")
        return

    viz_type = st.radio(
        "📊 Тип визуализации:",
        ["Распределения", "Сравнение по классам", "Матрица диаграмм рассеяния"],
        horizontal=True
    )

    if viz_type == "Распределения":
        cols = st.columns(2)
        for i, feature in enumerate(selected_features[:4]):
            with cols[i % 2]:
                st.markdown(
                    f"**Описание:** Гистограмма показывает распределение значений признака {feature} в датасете.")
                fig = px.histogram(
                    data,
                    x=feature,
                    title=f"Распределение признака: {feature}",
                    nbins=30,
                    color_discrete_sequence=['#667eea']
                )
                st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Сравнение по классам":
        if 'target' in data.columns:
            cols = st.columns(2)
            for i, feature in enumerate(selected_features[:4]):
                with cols[i % 2]:
                    st.markdown(
                        f"**Описание:** Диаграмма размаха показывает распределение признака {feature} для пользователей (0) и ботов (1).")
                    fig = px.box(
                        data,
                        x='target',
                        y=feature,
                        title=f"{feature} по классам",
                        color='target',
                        color_discrete_sequence=['#4ECDC4', '#FF6B6B']
                    )
                    fig.update_layout(xaxis=dict(tickvals=[0, 1], ticktext=['Пользователь', 'Бот']))
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("❌ Целевая переменная не найдена")

    elif viz_type == "Матрица диаграмм рассеяния":
        if len(selected_features) >= 2:
            features_for_matrix = selected_features[:4]
            st.markdown(
                "**Описание:** Матрица диаграмм рассеяния показывает парные отношения между выбранными признаками.")
            fig = px.scatter_matrix(
                data[features_for_matrix].dropna(),
                title="Матрица диаграмм рассеяния",
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Выберите минимум 2 признака для матрицы")


def show_correlation_analysis(data, selected_features):
    st.markdown("### 📊 Корреляционный анализ")

    if not selected_features:
        st.warning("⚠️ Выберите признаки для анализа в боковой панели")
        return

    corr_data = data[selected_features].corr()

    show_values = st.checkbox("📝 Показать значения корреляции", value=True)
    st.markdown(
        "**Описание:** Тепловая карта показывает корреляцию между выбранными признаками, где цвет указывает на силу и направление связи (синий — отрицательная, красный — положительная).")
    fig = px.imshow(
        corr_data,
        title="Корреляционная матрица выбранных признаков",
        color_continuous_scale='RdBu_r',
        aspect='auto',
        text_auto=show_values
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔝 Сильные корреляции")

    corr_pairs = []
    for i in range(len(corr_data.columns)):
        for j in range(i + 1, len(corr_data.columns)):
            feature1 = corr_data.columns[i]
            feature2 = corr_data.columns[j]
            correlation = corr_data.iloc[i, j]
            corr_pairs.append((feature1, feature2, correlation))

    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    top_corr_df = pd.DataFrame(
        corr_pairs[:10],
        columns=['Признак 1', 'Признак 2', 'Корреляция']
    )
    top_corr_df['Корреляция'] = top_corr_df['Корреляция'].round(3)

    st.dataframe(top_corr_df, use_container_width=True)


def show_ml_analysis(data, numerical_features):
    st.markdown("### 🤖 Машинное обучение")

    if 'target' not in data.columns:
        st.error("❌ Целевая переменная 'target' не найдена")
        return

    ml_features = [col for col in numerical_features if col in data.columns]
    if len(ml_features) == 0:
        st.error("❌ Не найдено численных признаков для обучения модели")
        return

    col1, col2 = st.columns(2)
    with col1:
        test_size = st.number_input("🎯 Размер тестовой выборки", 0.1, 0.95, 0.2, 0.05)
    with col2:
        n_estimators = st.number_input("🌳 Количество деревьев", 10, 200, 100, 10)

    if st.button("🚀 Обучить модель", type="primary"):
        with st.spinner("⏳ Обучение модели..."):
            X = data[ml_features].fillna(0)
            y = data['target']

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train_scaled, y_train)

            y_pred = model.predict(X_test_scaled)

            st.success("✅ Модель успешно обучена!")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(
                    "**Описание:** Матрица ошибок показывает, как модель классифицировала пользователей и ботов (истинные vs предсказанные классы).")
                cm = confusion_matrix(y_test, y_pred)

                fig = px.imshow(
                    cm,
                    text_auto=True,
                    title="Матрица ошибок",
                    labels={'x': 'Предсказанный класс', 'y': 'Истинный класс'},
                    color_continuous_scale='Blues'
                )
                fig.update_layout(
                    xaxis=dict(tickvals=[0, 1], ticktext=['Пользователь', 'Бот']),
                    yaxis=dict(tickvals=[0, 1], ticktext=['Пользователь', 'Бот'])
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown(
                    "**Описание:** Горизонтальная столбчатая диаграмма показывает важность топ-10 признаков для модели.")
                feature_importance = pd.DataFrame({
                    'Признак': ml_features,
                    'Важность': model.feature_importances_
                }).sort_values('Важность', ascending=False).head(10)

                fig = px.bar(
                    feature_importance,
                    x='Важность',
                    y='Признак',
                    orientation='h',
                    title="Топ-10 важных признаков",
                    color='Важность',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)

            accuracy = model.score(X_test_scaled, y_test)
            st.subheader("📊 Метрики качества")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🎯 Точность", f"{accuracy:.3f}")
            with col2:
                precision = cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) > 0 else 0
                st.metric("🔍 Точность (Precision)", f"{precision:.3f}")
            with col3:
                recall = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
                st.metric("📈 Полнота (Recall)", f"{recall:.3f}")


def show_interactive_plots(data, numerical_features, categorical_features):
    st.markdown("### 🎨 Интерактивные графики")

    plot_type = st.selectbox(
        "📊 Выберите тип графика:",
        ["Точечный график", "3D Точечный график", "Скрипичная диаграмма", "Кольцевая диаграмма"]
    )

    if plot_type == "Точечный график":
        col1, col2 = st.columns(2)
        with col1:
            x_feature = st.selectbox("Ось X:", numerical_features, key="scatter_x")
        with col2:
            y_feature = st.selectbox("Ось Y:", numerical_features, key="scatter_y",
                                     index=1 if len(numerical_features) > 1 else 0)

        color_by = st.selectbox("Цвет по:", ['target'] + categorical_features[:5], key="scatter_color")
        st.markdown(
            f"**Описание:** Точечный график показывает взаимосвязь между {x_feature} и {y_feature}, с цветом в зависимости от {color_by}.")
        fig = px.scatter(
            data.dropna(subset=[x_feature, y_feature]),
            x=x_feature,
            y=y_feature,
            color=color_by,
            title=f"{x_feature} против {y_feature}",
            opacity=0.6,
            hover_data=[col for col in data.columns[:5]]
        )
        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "3D Точечный график":
        if len(numerical_features) >= 3:
            col1, col2, col3 = st.columns(3)
            with col1:
                x_feature = st.selectbox("Ось X:", numerical_features, key="3d_x")
            with col2:
                y_feature = st.selectbox("Ось Y:", numerical_features, key="3d_y", index=1)
            with col3:
                z_feature = st.selectbox("Ось Z:", numerical_features, key="3d_z", index=2)

            color_by = st.selectbox("Цвет по:", ['target'] + categorical_features[:5], key="3d_color")
            st.markdown(
                f"**Описание:** 3D график показывает взаимосвязь между {x_feature}, {y_feature} и {z_feature}, с цветом в зависимости от {color_by}.")
            fig = px.scatter_3d(
                data.dropna(subset=[x_feature, y_feature, z_feature]),
                x=x_feature,
                y=y_feature,
                z=z_feature,
                color=color_by,
                title=f"3D: {x_feature} против {y_feature} против {z_feature}",
                opacity=0.6
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Недостаточно численных признаков для 3D графика")

    elif plot_type == "Скрипичная диаграмма":
        feature = st.selectbox("Выберите признак:", numerical_features, key="violin_feature")
        st.markdown(
            f"**Описание:** Скрипичная диаграмма показывает распределение признака {feature} для пользователей (0) и ботов (1), включая медиану и плотность.")
        if 'target' in data.columns:
            fig = px.violin(
                data,
                y=feature,
                x='target',
                box=True,
                title=f"Скрипичная диаграмма: {feature}",
                color='target',
                color_discrete_sequence=['#4ECDC4', '#FF6B6B']
            )
            fig.update_layout(
                xaxis=dict(tickvals=[0, 1], ticktext=['Пользователь', 'Бот'])
            )
            st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Кольцевая диаграмма":
        if len(categorical_features) >= 2:
            cat1 = st.selectbox("Первый уровень:", categorical_features, key="sun_1")
            cat2 = st.selectbox("Второй уровень:", categorical_features, key="sun_2", index=1)
            st.markdown(
                f"**Описание:** Кольцевая диаграмма показывает иерархическое распределение данных по категориям {cat1} и {cat2}.")
            sunburst_data = data[[cat1, cat2, 'target']].dropna()

            fig = px.sunburst(
                sunburst_data,
                path=[cat1, cat2],
                values='target',
                title=f"Кольцевая диаграмма: {cat1} -> {cat2}"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Недостаточно категориальных признаков для кольцевой диаграммы")

def show_detailed_info(data):
    st.markdown("### 📋 Детальная информация")

    st.subheader("🔍 Поиск в данных")

    if 'target' in data.columns:
        target_filter = st.selectbox(
            "Фильтр по типу:",
            ["Все", "Только пользователи (0)", "Только боты (1)"]
        )

        if target_filter == "Только пользователи (0)":
            filtered_data = data[data['target'] == 0]
        elif target_filter == "Только боты (1)":
            filtered_data = data[data['target'] == 1]
        else:
            filtered_data = data
    else:
        filtered_data = data

    search_term = st.text_input("🔍 Поиск по всем колонкам:")

    if search_term:
        mask = pd.Series([False] * len(filtered_data))
        for col in filtered_data.columns:
            if filtered_data[col].dtype == 'object':
                mask |= filtered_data[col].astype(str).str.contains(search_term, case=False, na=False)

        if mask.any():
            filtered_data = filtered_data[mask]
            st.success(f"✅ Найдено {len(filtered_data)} записей")
        else:
            st.warning("⚠️ Ничего не найдено")

    st.info(f"📊 Показано записей: {len(filtered_data)} из {len(data)}")

    page_size = st.selectbox("📄 Записей на страницу:", [10, 25, 50, 100], index=1)

    total_pages = (len(filtered_data) - 1) // page_size + 1

    if total_pages > 1:
        page = st.number_input(
            f"Страница (1-{total_pages}):",
            min_value=1,
            max_value=total_pages,
            value=1
        )

        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size

        display_data = filtered_data.iloc[start_idx:end_idx]
    else:
        display_data = filtered_data

    st.dataframe(display_data, use_container_width=True, height=400)

    if len(display_data) > 0:
        st.subheader("📊 Статистика по отфильтрованным данным")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("📝 Записей", len(filtered_data))

        with col2:
            if 'target' in filtered_data.columns:
                bot_ratio = (filtered_data['target'] == 1).mean() * 100
                st.metric("🤖 % ботов", f"{bot_ratio:.1f}%")

        with col3:
            missing_ratio = (filtered_data.isnull().sum().sum() / (
                    len(filtered_data) * len(filtered_data.columns))) * 100
            st.metric("❌ % пропусков", f"{missing_ratio:.1f}%")

    st.subheader("💾 Экспорт данных")

    if st.button("📥 Скачать отфильтрованные данные (CSV)"):
        csv = filtered_data.to_csv(index=False)
        st.download_button(
            label="📥 Скачать CSV",
            data=csv,
            file_name=f"filtered_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )


def create_dataset_summary(data):
    summary = {
        'total_records': len(data),
        'total_features': len(data.columns),
        'missing_values': data.isnull().sum().sum(),
        'memory_usage': data.memory_usage(deep=True).sum() / 1024 ** 2,
        'numeric_features': len(data.select_dtypes(include=[np.number]).columns),
        'categorical_features': len(data.select_dtypes(include=['object']).columns)
    }

    if 'target' in data.columns:
        summary['class_balance'] = data['target'].value_counts().to_dict()

    return summary


def detect_outliers(data, feature, method='IQR'):
    if method == 'IQR':
        Q1 = data[feature].quantile(0.25)
        Q3 = data[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = data[(data[feature] < lower_bound) | (data[feature] > upper_bound)]
        return outliers

    elif method == 'Z-score':
        z_scores = np.abs((data[feature] - data[feature].mean()) / data[feature].std())
        outliers = data[z_scores > 3]
        return outliers

    return pd.DataFrame()


def generate_report(data, numerical_features, categorical_features):
    report = []

    report.append("# 📊 Автоматический отчет по датасету VK Bots vs Users")
    report.append(f"**📅 Дата создания отчета:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    report.append("## 🔍 Основные характеристики")
    report.append(f"- **Общее количество записей:** {len(data):,}")
    report.append(f"- **Количество признаков:** {len(data.columns)}")
    report.append(f"- **Численных признаков:** {len(numerical_features)}")
    report.append(f"- **Категориальных признаков:** {len(categorical_features)}")
    report.append("")

    missing_total = data.isnull().sum().sum()
    report.append("## 📋 Качество данных")
    report.append(f"- **Общее количество пропусков:** {missing_total:,}")
    report.append(f"- **Процент пропусков:** {(missing_total / (len(data) * len(data.columns)) * 100):.2f}%")

    if missing_total > 0:
        top_missing = data.isnull().sum().sort_values(ascending=False).head(5)
        report.append("- **Топ-5 признаков с пропусками:**")
        for col, count in top_missing.items():
            if count > 0:
                report.append(f"  - {col}: {count:,} ({count / len(data) * 100:.1f}%)")

    report.append("")

    if 'target' in data.columns:
        report.append("## 🎯 Анализ целевой переменной")
        target_counts = data['target'].value_counts()
        report.append(f"- **Пользователи (0):** {target_counts.get(0, 0):,}")
        report.append(f"- **Боты (1):** {target_counts.get(1, 0):,}")

        if len(target_counts) == 2:
            balance_ratio = min(target_counts.values) / max(target_counts.values)
            report.append(f"- **Коэффициент баланса:** {balance_ratio:.3f}")

        report.append("")

    report.append("## 💡 Рекомендации")

    if missing_total > len(data) * len(data.columns) * 0.1:
        report.append("- ⚠️ **Высокий процент пропусков** - рассмотрите стратегии обработки пропущенных данных")

    if 'target' in data.columns and len(data['target'].value_counts()) == 2:
        balance_ratio = min(data['target'].value_counts().values) / max(data['target'].value_counts().values)
        if balance_ratio < 0.8:
            report.append("- ⚖️ **Дисбаланс классов** - рассмотрите методы балансировки данных")

    if len(numerical_features) > 20:
        report.append("- 🔍 **Много численных признаков** - рассмотрите методы отбора признаков")

    report.append("- 📊 **Проведите корреляционный анализ** для выявления взаимосвязей между признаками")
    report.append("- 🤖 **Попробуйте различные алгоритмы машинного обучения** для классификации")

    return "\n".join(report)


if __name__ == "__main__":
    main()
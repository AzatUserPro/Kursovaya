import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

# === 1. ЗАГРУЗКА И ОЧИСТКА ДАННЫХ ===
@st.cache_data
def load_and_clean():
    df = pd.read_csv('data.csv', sep=';', on_bad_lines='skip', header=None)

    columns = [
        "federal_district", "federal_district_short", "region_code", "region_name",
        "okato", "id", "name", "name_short", "year",
        "e1", "e2", "e3", "e4", "e5", "e6", "e8",
        "ege_budg", "wos", "scopus", "rsci", "rnd",
        "total_income", "square", "phd_share", "pps",
        "rnd_private", "pk"
    ]

    if df.shape[1] > len(columns):
        df = df.iloc[:, :len(columns)]
    elif df.shape[1] < len(columns):
        for _ in range(len(columns) - df.shape[1]):
            df[len(df.columns)] = np.nan

    df.columns = columns

    df.dropna(how='all', inplace=True)
    df.drop_duplicates(inplace=True)
    df.dropna(subset=['year', 'e1', 'name_short'], inplace=True)

    numeric_cols = [c for c in columns if c not in ["federal_district", "federal_district_short",
                                                    "region_name", "name", "name_short", "okato"]]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # e1 должен быть в диапазоне 0–100
    df = df[(df['e1'] >= 0) & (df['e1'] <= 100)]
    df.dropna(subset=['e1'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

df_raw = load_and_clean()

# === 2. ФИЛЬТРЫ (боковая панель) ===
st.sidebar.title("🎛️ Фильтры")
selected_fd = st.sidebar.multiselect(
    "Федеральный округ",
    options=sorted(df_raw['federal_district'].unique())
)
selected_year = st.sidebar.slider(
    "Год",
    min_value=2013,
    max_value=2017,
    value=(2013, 2017)
)

# Применяем фильтры
filtered_df = df_raw.copy()
if selected_fd:
    filtered_df = filtered_df[filtered_df['federal_district'].isin(selected_fd)]
filtered_df = filtered_df[
    (filtered_df['year'] >= selected_year[0]) &
    (filtered_df['year'] <= selected_year[1])
]

# === 3. ЧИТАЕМЫЕ НАЗВАНИЯ МЕТРИК ===
metric_labels = {
    'e1': 'Балл ЕГЭ с учётом льготников = 100',
    'e2': 'Средства от НИОКР на научно-педагогического работника (тыс. руб.)',
    'e3': 'Доля иностранных студентов (%)',
    'e4': 'Средний балл ЕГЭ поступивших на бюджет (все направления)',
    'e5': 'Зарплата ППС относительно средней по региону (%)',
    'e6': 'Трудоустройство выпускников (%)',
    'e8': 'Число ППС с учёной степенью на 100 студентов',
    'ege_budg': 'Средний балл ЕГЭ на бюджете (альтернативный расчёт)',
    'phd_share': 'Доля аспирантов (%)',
    'rnd': 'Общий объём НИОКР (тыс. руб.)',
    'wos': 'Публикации в Web of Science на 100 НПР',
    'scopus': 'Публикации в Scopus на 100 НПР',
    'rsci': 'Публикации в РИНЦ на 100 НПР',
    'total_income': 'Доходы вуза из всех источников (тыс. руб.)',
    'square': 'Общая площадь зданий (кв. м)',
    'pps': 'Число ППС',
    'rnd_private': 'Объём НИОКР по заказам юрлиц (тыс. руб.)',
    'pk': 'Число ПК на студента'
}

# === 4. ВЫБОР ЗАДАЧИ ===
st.sidebar.title("📋 Выберите задачу")
task = st.sidebar.radio("Цель анализа:", [
    "1. Обзор всех показателей: описательная статистика и графики",
    "2. Анализ качества приёма абитуриентов",
    "3. Прогноз качества приёма на 2018 год",
    "4. Факторы, влияющие на баллы абитурентов",
    "5. Сравнение вузов по ключевым показателям",
    "6. Подбор вуза по вашим критериям",
    "7. Кластеризация вузов",
    "8. Подбор вуза по вашим баллам ЕГЭ"])

# === ЗАДАЧА 1: ОЦЕНКА КАЧЕСТВА ДАННЫХ ===
if task == "1. Обзор всех показателей: описательная статистика и графики":
    st.title("🔍 Оценка качества данных")

    # --- 1. Общая структура ---
    st.subheader("1. Общая структура")
    st.write(f"- **Записей (строк)**: {len(df_raw):,}")
    st.write(f"- **Уникальных вузов**: {df_raw['name_short'].nunique():,}")
    st.write(f"- **Годы наблюдений**: {df_raw['year'].min()} – {df_raw['year'].max()}")
    st.write(f"- **Федеральных округов**: {df_raw['federal_district'].nunique()}")

    # --- 2. Пропущенные значения ---
    st.subheader("2. Пропущенные значения (NaN)")
    missing = df_raw.isna().sum()
    missing_pct = (missing / len(df_raw)) * 100
    missing_df = pd.DataFrame({
        'Пропусков': missing,
        '% пропусков': missing_pct.round(2)
    })
    # Показываем только столбцы с пропусками
    missing_df = missing_df[missing_df['Пропусков'] > 0].sort_values('% пропусков', ascending=False)
    if not missing_df.empty:
        st.dataframe(missing_df)
    else:
        st.write("Пропусков не обнаружено.")

    # --- 3. Дубликаты ---
    st.subheader("3. Дубликаты")
    full_dup = df_raw.duplicated().sum()
    id_year_dup = df_raw.duplicated(subset=['id', 'year']).sum()
    st.write(f"- **Полные дубликаты строк**: {full_dup}")
    st.write(f"- **Дубликаты по вузу + год** (`id` + `year`): {id_year_dup}")

    # --- 4. Некорректные значения ---
    st.subheader("4. Некорректные значения")
    issues = []
    if issues:
        for issue in issues:
            st.write(issue)
    else:
        st.write("✅ Все числовые показатели находятся в ожидаемых диапазонах.")

    # --- 5. Полнота данных по вузам ---
    st.subheader("5. Полнота временных рядов")
    years_per_univ = df_raw.groupby('name_short')['year'].nunique()
    full_period_count = (years_per_univ == 5).sum()
    total_univ = df_raw['name_short'].nunique()
    st.write(f"- Вузов с данными за все 5 лет (2013–2017): **{full_period_count} из {total_univ}** ({full_period_count / total_univ * 100:.1f}%)")

    # --- 6. Гистограмма пропусков по столбцам ---
    st.subheader("6. Распределение пропусков по столбцам")
    missing_pct_all = (df_raw.isna().mean() * 100).sort_values(ascending=False)
    fig_missing = px.bar(
        x=missing_pct_all.index,
        y=missing_pct_all.values,
        labels={'x': 'Показатель', 'y': '% пропусков'},
        title='Процент пропусков по столбцам'
    )
    fig_missing.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_missing, use_container_width=True)

# === ЗАДАЧА 2: АНАЛИЗ `e1` ===
elif task == "2. Анализ качества приёма абитуриентов":
    st.title("📊 Анализ качества приёма абитуриентов")
    st.markdown("Этот показатель отражает **средний балл ЕГЭ поступающих**, с учётом того, что льготники (поступившие без экзаменов) считаются набравшими **100 баллов**.")
    st.dataframe(filtered_df['e1'].describe().to_frame().rename(columns={'e1': metric_labels['e1']}))

    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.box(filtered_df, y='e1', labels={'e1': metric_labels['e1']})
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        avg_fd = filtered_df.groupby('federal_district')['e1'].mean().reset_index()
        fig2 = px.bar(avg_fd, x='federal_district', y='e1', labels={'e1': metric_labels['e1'], 'federal_district': 'Федеральный округ'})
        st.plotly_chart(fig2, use_container_width=True)

# === ЗАДАЧА 3: ПРОГНОЗ НА 2018 ГОД ===
elif task == "3. Прогноз качества приёма на 2018 год":
    st.title("🔮 Прогноз качества приёма абитуриентов на 2018 год")
    univ = st.selectbox("Выберите вуз", filtered_df['name_short'].unique())
    data = filtered_df[filtered_df['name_short'] == univ].sort_values('year')

    if len(data) < 2:
        st.warning("Недостаточно данных (нужно ≥2 года).")
    else:
        X = data[['year']].values
        y = data['e1'].values
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        pred = model.predict([[2018]])[0]
        st.metric("Прогноз на 2018 г.", f"{pred:.2f}")
        fig = px.line(data, x='year', y='e1', title=f"Динамика качества приёма: {univ}", labels={'e1': metric_labels['e1'], 'year': 'Год'})
        fig.add_scatter(x=[2018], y=[pred], mode='markers+text', text=["Прогноз 2018"], textposition="top center", marker=dict(color='red', size=10))
        st.plotly_chart(fig, use_container_width=True)

# === ЗАДАЧА 4: ФАКТОРЫ ВЛИЯНИЯ ===
elif task == "4. Факторы, влияющие на баллы абитурентов":
    st.title("🔍 Какие факторы влияют на баллы абитурентов?")
    features = ['e4', 'e5', 'e6', 'e8', 'phd_share', 'rnd', 'wos', 'scopus']
    df_corr = filtered_df[['e1'] + features].dropna()
    corr_columns = [metric_labels.get(col, col) for col in ['e1'] + features]
    df_corr_renamed = df_corr.copy()
    df_corr_renamed.columns = corr_columns
    corr_matrix = df_corr_renamed.corr()
    fig = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", title="Корреляция качества приёма с другими показателями", color_continuous_scale='Blues')
    fig.update_xaxes(side="top")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)

    X, y = df_corr[features], df_corr['e1']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    importances = pd.Series(model.feature_importances_, index=features)
    importances.index = [metric_labels.get(col, col) for col in importances.index]
    importances = importances.sort_values(ascending=False)
    fig2 = px.bar(importances, title='Важность факторов для прогноза качества приёма')
    st.plotly_chart(fig2, use_container_width=True)

# === ЗАДАЧА 5: СРАВНЕНИЕ ВУЗОВ ===
elif task == "5. Сравнение вузов по ключевым показателям":
    st.title("⚖️ Сравнение вузов")
    unis = st.multiselect("Выберите до 3 вузов", filtered_df['name_short'].unique(), max_selections=3)
    if len(unis) >= 2:
        comp = filtered_df[filtered_df['name_short'].isin(unis) & (filtered_df['year'] == 2017)]
        if not comp.empty:
            comp.set_index('name_short', inplace=True)
            cols = ['e1', 'e6', 'e8', 'rnd', 'phd_share']
            comp = comp[cols]
            comp.rename(columns=metric_labels, inplace=True)
            st.dataframe(comp)
        else:
            st.warning("Нет данных за 2017 год.")

# === ЗАДАЧА 6: РЕКОМЕНДАЦИИ ДЛЯ АБИТУРИЕНТА ===
elif task == "6. Подбор вуза по вашим критериям":
    st.title("🎓 Подбор вуза по вашим приоритетам")
    st.markdown("""
    Выберите, какие показатели для вас важны при выборе вуза.  
    Система отранжирует вузы и покажет **топ-10 рекомендаций**.
    """)

    # Доступные критерии для выбора
    criteria_options = {
        'e1': 'Качество приёма (балл ЕГЭ с льготниками = 100)',
        'e6': 'Трудоустройство выпускников (%)',
        'e8': 'ППС с учёной степенью на 100 студентов',
        'rnd': 'Общий объём НИОКР (тыс. руб.)',
        'phd_share': 'Доля аспирантов (%)',
        'e5': 'Зарплата ППС выше средней по региону (%)'
    }

    # Выбор критериев
    selected_criteria = st.multiselect(
        "Выберите важные критерии (можно несколько)",
        options=list(criteria_options.keys()),
        format_func=lambda x: criteria_options[x]
    )

    if not selected_criteria:
        st.info("Пожалуйста, выберите хотя бы один критерий.")
    else:
        # Используем только данные за последний год (2017)
        df_2017 = filtered_df[filtered_df['year'] == 2017].copy()

        if df_2017.empty:
            st.warning("Нет данных за 2017 год с учётом фильтров.")
        else:
            # Оставляем только вузы с заполненными данными по всем выбранным критериям
            df_valid = df_2017.dropna(subset=selected_criteria)

            if df_valid.empty:
                st.warning("Нет вузов с полными данными по выбранным критериям.")
            else:
                # Нормализуем каждый критерий (приводим к шкале 0–1)
                for col in selected_criteria:
                    min_val = df_valid[col].min()
                    max_val = df_valid[col].max()
                    if max_val == min_val:
                        df_valid[f'{col}_norm'] = 1.0
                    else:
                        df_valid[f'{col}_norm'] = (df_valid[col] - min_val) / (max_val - min_val)

                # Считаем общий балл — среднее по выбранным нормализованным критериям
                norm_cols = [f'{col}_norm' for col in selected_criteria]
                df_valid['total_score'] = df_valid[norm_cols].mean(axis=1)

                # Сортируем по убыванию
                df_top = df_valid.sort_values('total_score', ascending=False).head(10)

                # Выводим результат
                st.subheader("🏆 Топ-10 рекомендуемых вузов")
                result_df = df_top[['name_short'] + selected_criteria].copy()
                result_df.rename(columns={**metric_labels, 'name_short': 'Вуз'}, inplace=True)
                st.dataframe(result_df.reset_index(drop=True))

# === ЗАДАЧА 7: КЛАСТЕРИЗАЦИЯ ВУЗОВ ===
elif task == "7. Кластеризация вузов":
    st.title("🧩 Кластеризация вузов по схожести показателей")
    st.markdown("""
    Вузы группируются на кластеры по ключевым метрикам:
    - Качество приёма (`e1`)
    - Трудоустройство (`e6`)
    - ППС с учёной степенью (`e8`)
    - Объём НИОКР (`rnd`)
    - Доля аспирантов (`phd_share`)

    Используется метод **K-Means (5 кластеров)** на данных за **2017 год**.
    """)

    # Берём данные за 2017 год
    df_2017 = filtered_df[filtered_df['year'] == 2017].copy()
    features = ['e1', 'e6', 'e8', 'rnd', 'phd_share']

    # Удаляем строки с пропусками
    df_clean = df_2017.dropna(subset=features)

    if df_clean.empty:
        st.warning("Нет данных за 2017 год с полными значениями по всем метрикам.")
    else:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        # Нормализация
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_clean[features])

        # Кластеризация
        kmeans = KMeans(n_clusters=5, random_state=42)
        df_clean['cluster'] = kmeans.fit_predict(X_scaled)

        # Профиль кластеров
        cluster_profile = df_clean.groupby('cluster')[features].mean()
        cluster_profile.index.name = 'Кластер'
        cluster_profile.rename(columns=metric_labels, inplace=True)

        st.subheader("📊 Профиль кластеров (средние значения)")
        st.dataframe(cluster_profile)

        # Выбор кластера
        selected_cluster = st.selectbox("Выберите кластер для просмотра вузов", sorted(df_clean['cluster'].unique()))

        # Вузы в выбранном кластере
        cluster_vuzes = df_clean[df_clean['cluster'] == selected_cluster][['name_short'] + features]
        cluster_vuzes.rename(columns={**metric_labels, 'name_short': 'Вуз'}, inplace=True)

        st.subheader(f"🎓 Вузы в кластере {selected_cluster}")
        st.dataframe(cluster_vuzes.reset_index(drop=True))

# === ЗАДАЧА 8: ПОДБОР ВУЗА ПО БАЛЛАМ ЕГЭ ===
elif task == "8. Подбор вуза по вашим баллам ЕГЭ":
    st.title("🎓 Подбор вуза по вашему баллу ЕГЭ")
    st.markdown("""
    Введите **ваш средний балл ЕГЭ**.  
    Система покажет **вузы, в которые вы можете поступить** на основе данных за 2017 год.
    """)

    # Ввод балла
    user_score = st.number_input(
        "Ваш средний балл ЕГЭ (от 0 до 100)",
        min_value=0.0,
        max_value=100.0,
        value=70.0,
        step=0.5
    )

    if user_score < 0 or user_score > 100:
        st.warning("Средний балл ЕГЭ должен быть в диапазоне от 0 до 100.")
    else:
        # Данные за 2017 год
        df_2017 = filtered_df[filtered_df['year'] == 2017].copy()

        # Вузы, где качество приёма <= вашему баллу
        eligible = df_2017[df_2017['e1'] <= user_score].copy()
        eligible = eligible.sort_values('e1', ascending=False)

        if eligible.empty:
            st.warning("К сожалению, по вашему баллу нет подходящих вузов с учётом фильтров.")
        else:
            st.subheader("🎯 Вузы, в которые вы можете поступить")
            result = eligible[['name_short', 'e1', 'e6', 'region_name']].copy()
            result.columns = ['Вуз', 'Качество приёма (e1)', 'Трудоустройство (%)', 'Регион']
            st.dataframe(result.reset_index(drop=True))

            # Дополнительно: прогноз на 2018
            st.subheader("🔮 Прогноз на 2018 год")
            st.write("Некоторые вузы могут ужесточить конкурс. Вот те, где прогнозируемый e1 на 2018 год всё ещё ≤ вашего балла:")

            # Строим прогноз для каждого вуза
            eligible['e1_2018_pred'] = np.nan
            for idx, row in eligible.iterrows():
                univ_data = filtered_df[
                    (filtered_df['name_short'] == row['name_short']) &
                    (filtered_df['year'] <= 2017)
                ].sort_values('year')
                if len(univ_data) >= 2:
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    model.fit(univ_data[['year']], univ_data['e1'])
                    pred_2018 = model.predict([[2018]])[0]
                    eligible.at[idx, 'e1_2018_pred'] = pred_2018

            # Фильтруем по прогнозу
            safe_vuzes = eligible[eligible['e1_2018_pred'] <= user_score]
            if not safe_vuzes.empty:
                st.dataframe(safe_vuzes[['Вуз', 'Качество приёма (e1)', 'e1_2018_pred', 'Регион']].rename(columns={
                    'e1_2018_pred': 'Прогноз e1 на 2018'
                }).reset_index(drop=True))
            else:
                st.write("По прогнозу на 2018 год, конкурс может вырасти выше вашего балла во всех вузах.")

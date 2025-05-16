import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import streamlit.components.v1 as components

# Page config
st.set_page_config(page_title="화장품 수출 대시보드", layout="wide")

# Title
st.title("🌍 2024 화장품 수출 적합도 대시보드")

# Load data
@st.cache_data
def load_data():
    # 데이터 생성
    data = {
        'Country': [
            '중국', '미국', '일본', '베트남', '홍콩', '러시아', '폴란드', '대만', '태국', 
            '싱가포르', '캐나다', 'UAE', '호주', '영국', '필리핀', '인도네시아', '네덜란드', 
            '스페인', '말레이시아', '카자흐스탄'
        ],
        'Continent': [
            '아시아', '북미', '아시아', '아시아', '아시아', '유럽', '유럽', '아시아', '아시아',
            '아시아', '북미', '중동', '오세아니아', '유럽', '아시아', '아시아', '유럽',
            '유럽', '아시아', '아시아'
        ],
        'Total_Export_USD': [
            908247, 592074, 332213, 206933, 191093, 122240, 99657, 99367, 93613,
            77604, 76580, 66358, 60062, 50775, 49411, 45000, 42000, 38000, 35000, 30000
        ],
        'Average_Growth_Rate_Percent': [
            -12.0, 65.5, 23.1, 24.5, 22.2, -8.8, 141.5, 36.7, 22.2,
            22.2, 61.8, 72.3, 54.0, 22.2, 32.6, 85.4, 87.7, 36.0, 25.9, 50.6
        ],
        'Risk_Index': [
            4, 3, 3, 4, 3, 5, 3, 3, 4,
            2, 2, 4, 2, 2, 3, 4, 3, 3, 3, 3
        ],
        'Overdue_Rate_Percent': [
            25.0, 17.2, 17.4, 30.0, 20.0, 35.0, 15.0, 14.8, 25.0,
            10.0, 8.0, 20.0, 5.0, 7.0, 18.0, 28.0, 12.0, 15.0, 18.0, 22.0
        ]
    }
    
    df = pd.DataFrame(data)
    
    # Export_Suitability_Score 계산 (수출액, 성장률, 리스크 지수, 연체율 기반)
    df['Export_Suitability_Score'] = (
        df['Total_Export_USD'] * 0.4 + 
        df['Average_Growth_Rate_Percent'] * 100 * 0.3 - 
        df['Risk_Index'] * 1000 * 0.2 - 
        df['Overdue_Rate_Percent'] * 100 * 0.1
    )
    
    # 클러스터 분류
    df['Cluster'] = df.apply(lambda row: (
        '고성장-저위험' if row['Average_Growth_Rate_Percent'] >= 30 and row['Risk_Index'] <= 3 else
        '고성장-고위험' if row['Average_Growth_Rate_Percent'] >= 30 else
        '저성장-저위험' if row['Risk_Index'] <= 3 else '저성장-고위험'
    ), axis=1)
    
    return df

data = load_data()

# Sidebar filters
st.sidebar.header("필터")
continent_filter = st.sidebar.multiselect("대륙 선택", options=data['Continent'].unique(), default=data['Continent'].unique())
cluster_filter = st.sidebar.multiselect("클러스터 선택", options=data['Cluster'].unique(), default=data['Cluster'].unique())

filtered = data[(data['Continent'].isin(continent_filter)) & (data['Cluster'].isin(cluster_filter))]

# 차트 표시 설정
st.sidebar.header("차트 표시 설정")
show_top_countries = st.sidebar.checkbox("수출 적합도 상위 국가", value=True)
show_risk_growth = st.sidebar.checkbox("리스크 vs 성장률 클러스터", value=True)
show_regional = st.sidebar.checkbox("대륙별 평균 적합도 점수", value=True)
show_growth_rate = st.sidebar.checkbox("국가별 수출 성장률", value=True)

# 차트 표시 on/off 기능만 유지

# Section 1: Top N Countries
if show_top_countries:
    st.markdown("---")
    st.subheader("🏆 수출 적합도 점수 기준 상위 국가")
    top_n = st.slider("상위 국가 수 선택", 5, 20, 10)
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    sns.barplot(data=filtered.nlargest(top_n, 'Export_Suitability_Score'), x='Export_Suitability_Score', y='Country', palette='viridis', ax=ax1)
    ax1.set_title("수출 적합도 상위 국가")
    ax1.set_xlabel("수출 적합도 점수")
    ax1.set_ylabel("국가")
    st.pyplot(fig1)

# Section 2: Risk vs Growth Clustering
if show_risk_growth:
    st.markdown("---")
    st.subheader("📈 리스크 vs 성장률 클러스터")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    # 스캐터플롯에서 s 파라미터 제거하여 오류 방지
    scatter = sns.scatterplot(data=filtered, x='Risk_Index', y='Average_Growth_Rate_Percent', 
                     hue='Cluster', style='Cluster', ax=ax2)
    ax2.set_title("클러스터별 성장률 vs 리스크")
    ax2.set_xlabel("리스크 지수")
    ax2.set_ylabel("평균 성장률 (%)")
    ax2.grid(True)
    st.pyplot(fig2)

# Section 3: Regional Score Analysis
if show_regional:
    st.markdown("---")
    st.subheader("🌐 대륙별 평균 적합도 점수")
    region_score = filtered.groupby('Continent')['Export_Suitability_Score'].mean().sort_values()
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    sns.barplot(x=region_score.values, y=region_score.index, palette='coolwarm', ax=ax3)
    ax3.set_title("대륙별 평균 적합도")
    ax3.set_xlabel("평균 적합도 점수")
    ax3.set_ylabel("대륙")
    st.pyplot(fig3)

# Section 4: Growth Rate Analysis
if show_growth_rate:
    st.markdown("---")
    st.subheader("🚀 국가별 수출 성장률")
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    growth_data = filtered.sort_values('Average_Growth_Rate_Percent', ascending=False).head(10)
    sns.barplot(data=growth_data, x='Country', y='Average_Growth_Rate_Percent', palette='YlGn', ax=ax4)
    ax4.set_title("상위 10개국 수출 성장률")
    ax4.set_xlabel("국가")
    ax4.set_ylabel("성장률 (%)")
    plt.xticks(rotation=45)
    st.pyplot(fig4)

# Section 5: Predictive Modeling (항상 표시)
st.markdown("---")
st.subheader("🤖 수출 적합도 점수 예측 모델")

features = ['Total_Export_USD', 'Average_Growth_Rate_Percent', 'Risk_Index', 'Overdue_Rate_Percent']
X = filtered[features]
y = filtered['Export_Suitability_Score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)

col1, col2 = st.columns(2)
col1.metric("모델 R² 점수", f"{model.score(X_test, y_test):.2f}")
col2.metric("평균 예측 점수", f"{pred.mean():,.0f}")

# 사용자 입력 기반 예측
st.subheader("새로운 국가 수출 적합도 예측")
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

with col1:
    new_export = st.number_input("수출액 (천 USD)", min_value=0, max_value=1000000, value=100000)
with col2:
    new_growth = st.number_input("성장률 (%)", min_value=-100.0, max_value=200.0, value=30.0)
with col3:
    new_risk = st.number_input("리스크 지수 (1-5)", min_value=1, max_value=5, value=3)
with col4:
    new_overdue = st.number_input("연체율 (%)", min_value=0.0, max_value=100.0, value=15.0)

if st.button("예측하기"):
    new_data = pd.DataFrame({
        'Total_Export_USD': [new_export],
        'Average_Growth_Rate_Percent': [new_growth],
        'Risk_Index': [new_risk],
        'Overdue_Rate_Percent': [new_overdue]
    })
    prediction = model.predict(new_data)[0]
    st.success(f"예측된 수출 적합도 점수: {prediction:,.0f}")
    
    # 유사 국가 찾기
    data['Score_Diff'] = abs(data['Export_Suitability_Score'] - prediction)
    similar_countries = data.nsmallest(3, 'Score_Diff')
    st.info("유사한 적합도 점수를 가진 국가:")
    for i, row in similar_countries.iterrows():
        st.write(f"- {row['Country']}: {row['Export_Suitability_Score']:,.0f} (차이: {row['Score_Diff']:,.0f})")

# Footer
st.markdown("---")
st.caption("화장품 수출 인텔리전스 대시보드 © 2024 | Powered by Streamlit & sklearn")

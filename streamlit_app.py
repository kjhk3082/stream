import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# 페이지 설정
st.set_page_config(
    page_title="K-Beauty Export Optimizer (KBEO)",
    page_icon="💄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일링
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #ff69b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .risk-high { color: #ff4444; }
    .risk-medium { color: #ffaa00; }
    .risk-low { color: #44ff44; }
</style>
""", unsafe_allow_html=True)

# 데이터 로딩 함수
@st.cache_data
def load_export_data():
    """실제 화장품 수출 데이터 로딩"""
    # 실제 2024년 화장품 수출 데이터 (상위 20개국)
    data = {
        'Country': ['중국', '미국', '일본', '베트남', '홍콩', '러시아', '대만', '태국', 
                   '싱가포르', '아랍에미리트', '영국', '말레이시아', '폴란드', '인도네시아', 
                   '캐나다', '호주', '카자흐스탄', '필리핀', '네덜란드', '키르기스스탄'],
        'Export_Value': [2156.3, 1547.6, 840.4, 466.1, 511.1, 322.3, 218.8, 186.4, 
                        117.2, 158.3, 133.0, 112.1, 112.4, 118.9, 103.4, 96.0, 
                        83.7, 76.7, 74.6, 73.9],
        'Growth_Rate': [-9.3, 51.3, 26.0, 4.8, 16.6, 2.0, 31.5, 12.6, 
                       14.6, 87.2, 46.5, 26.2, 154.2, 73.8, 54.9, 56.4, 
                       32.7, 33.5, 33.5, 19.3],
        'Risk_Index': [4, 2, 1, 4, 3, 5, 2, 3, 2, 3, 2, 3, 3, 4, 2, 2, 4, 4, 1, 4],
        'PDR_Rate': [8.5, 3.2, 2.1, 12.3, 6.8, 18.9, 4.5, 8.7, 3.8, 7.2, 4.1, 9.1, 
                    6.5, 15.2, 3.9, 2.8, 14.7, 11.8, 2.3, 16.4],
        'Continent': ['Asia', 'North America', 'Asia', 'Asia', 'Asia', 'Europe', 'Asia', 'Asia',
                     'Asia', 'Asia', 'Europe', 'Asia', 'Europe', 'Asia', 'North America', 
                     'Oceania', 'Asia', 'Asia', 'Europe', 'Asia']
    }
    return pd.DataFrame(data)

# MinMax 정규화 함수
def minmax_normalize(series):
    """MinMax 정규화 수행"""
    return 100 * (series - series.min()) / (series.max() - series.min())

# 수출 적합도 점수 계산
def calculate_export_suitability(df, weights):
    """가중합 기반 수출 적합도 점수 계산"""
    df_copy = df.copy()
    
    # MinMax 정규화
    df_copy['Export_Score'] = minmax_normalize(df_copy['Export_Value'])
    df_copy['Growth_Score'] = minmax_normalize(df_copy['Growth_Rate'])
    df_copy['Safety_Score'] = minmax_normalize(6 - df_copy['Risk_Index'])  # 위험지수 역정규화
    df_copy['Payment_Score'] = minmax_normalize(100 - df_copy['PDR_Rate'])  # 연체율 역정규화
    
    # 가중합 계산
    df_copy['Suitability_Score'] = (
        df_copy['Export_Score'] * weights['export'] / 100 +
        df_copy['Growth_Score'] * weights['growth'] / 100 +
        df_copy['Safety_Score'] * weights['safety'] / 100 +
        df_copy['Payment_Score'] * weights['payment'] / 100
    )
    
    return df_copy

# K-means 군집분석
def perform_clustering(df, n_clusters=4):
    """K-means 군집분석 수행"""
    features = ['Export_Value', 'Growth_Rate', 'Risk_Index', 'PDR_Rate']
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[features])
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(scaled_features)
    
    # 클러스터 라벨링
    cluster_labels = {
        0: '고성장-저위험',
        1: '고성장-고위험', 
        2: '저성장-저위험',
        3: '저성장-고위험'
    }
    df['Cluster_Label'] = df['Cluster'].map(cluster_labels)
    
    return df, kmeans, scaler

# 메인 애플리케이션
def main():
    # 헤더
    st.markdown('<h1 class="main-header">💄 K-Beauty Export Optimizer (KBEO)</h1>', 
                unsafe_allow_html=True)
    st.markdown("### MinMax 정규화 기반 화장품 수출 최적화 전략 분석 플랫폼")
    
    # 데이터 로딩
    df = load_export_data()
    
    # 사이드바 설정
    st.sidebar.header("🎛️ 분석 설정")
    
    # 전략 선택
    strategy_options = {
        "수출중심": {"export": 60, "growth": 20, "safety": 15, "payment": 5},
        "성장중심": {"export": 20, "growth": 60, "safety": 15, "payment": 5},
        "안전중심": {"export": 20, "growth": 20, "safety": 50, "payment": 10},
        "밸런스": {"export": 30, "growth": 40, "safety": 20, "payment": 10},
        "사용자정의": None
    }
    
    selected_strategy = st.sidebar.selectbox("전략 선택", list(strategy_options.keys()))
    
    if selected_strategy == "사용자정의":
        st.sidebar.subheader("가중치 설정 (%)")
        export_weight = st.sidebar.slider("수출액 비중", 0, 100, 30)
        growth_weight = st.sidebar.slider("성장률 비중", 0, 100, 40)
        safety_weight = st.sidebar.slider("안전도 비중", 0, 100, 20)
        payment_weight = st.sidebar.slider("결제안전 비중", 0, 100, 10)
        
        total = export_weight + growth_weight + safety_weight + payment_weight
        if total != 100:
            st.sidebar.warning(f"가중치 합계: {total}% (100%가 되도록 조정하세요)")
        
        weights = {
            "export": export_weight,
            "growth": growth_weight, 
            "safety": safety_weight,
            "payment": payment_weight
        }
    else:
        weights = strategy_options[selected_strategy]
    
    # 대륙 필터
    continent_filter = st.sidebar.multiselect(
        "대륙 선택", 
        df['Continent'].unique(), 
        default=df['Continent'].unique()
    )
    
    # 상위 국가 수 선택
    top_n = st.sidebar.slider("분석 대상 국가 수", 5, 20, 15)
    
    # 데이터 필터링
    filtered_df = df[df['Continent'].isin(continent_filter)].head(top_n)
    
    # 수출 적합도 계산
    analyzed_df = calculate_export_suitability(filtered_df, weights)
    analyzed_df = analyzed_df.sort_values('Suitability_Score', ascending=False)
    
    # 군집분석 수행
    clustered_df, kmeans_model, scaler = perform_clustering(analyzed_df)
    
    # 탭 구성
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 종합 대시보드", "🎯 전략별 분석", "🔍 군집 분석", 
        "📈 성장성 분석", "⚠️ 리스크 분석", "🎮 시뮬레이션"
    ])
    
    with tab1:
        st.header("📊 종합 대시보드")
        
        # KPI 지표
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "최고 수출액", 
                f"${analyzed_df['Export_Value'].max():.1f}B",
                f"{analyzed_df.loc[analyzed_df['Export_Value'].idxmax(), 'Country']}"
            )
        
        with col2:
            st.metric(
                "최고 성장률", 
                f"{analyzed_df['Growth_Rate'].max():.1f}%",
                f"{analyzed_df.loc[analyzed_df['Growth_Rate'].idxmax(), 'Country']}"
            )
        
        with col3:
            st.metric(
                "최고 적합도", 
                f"{analyzed_df['Suitability_Score'].max():.1f}점",
                f"{analyzed_df.loc[analyzed_df['Suitability_Score'].idxmax(), 'Country']}"
            )
        
        with col4:
            avg_risk = analyzed_df['Risk_Index'].mean()
            risk_color = "🟢" if avg_risk <= 2 else "🟡" if avg_risk <= 3 else "🔴"
            st.metric(
                "평균 위험도", 
                f"{avg_risk:.1f} {risk_color}",
                f"전체 {len(analyzed_df)}개국"
            )
        
        # 상위 10개국 수출 적합도 차트
        st.subheader("🏆 상위 10개국 수출 적합도")
        top_10 = analyzed_df.head(10)
        
        fig_bar = px.bar(
            top_10, 
            x='Country', 
            y='Suitability_Score',
            color='Risk_Index',
            color_continuous_scale='RdYlGn_r',
            title=f"{selected_strategy} 전략 기준 수출 적합도"
        )
        fig_bar.update_layout(height=400)
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # 수출액 vs 성장률 산점도
        st.subheader("📈 수출액 vs 성장률 매트릭스")
        fig_scatter = px.scatter(
            analyzed_df,
            x='Export_Value',
            y='Growth_Rate', 
            size='Suitability_Score',
            color='Risk_Index',
            hover_name='Country',
            color_continuous_scale='RdYlGn_r',
            title="BCG 매트릭스 (수출액-성장률)",
            labels={
                'Export_Value': '수출액 (억달러)',
                'Growth_Rate': '성장률 (%)',
                'Risk_Index': '위험지수'
            }
        )
        
        # 사분면 구분선 추가
        median_export = analyzed_df['Export_Value'].median()
        median_growth = analyzed_df['Growth_Rate'].median()
        
        fig_scatter.add_hline(y=median_growth, line_dash="dash", line_color="gray")
        fig_scatter.add_vline(x=median_export, line_dash="dash", line_color="gray")
        
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with tab2:
        st.header("🎯 전략별 분석 결과")
        
        # 현재 전략 정보
        st.info(f"**선택된 전략: {selected_strategy}**\n"
                f"수출액: {weights['export']}%, 성장률: {weights['growth']}%, "
                f"안전도: {weights['safety']}%, 결제안전: {weights['payment']}%")
        
        # 전략별 상위 5개국 비교
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🥇 최우선 진출 대상")
            top_5 = analyzed_df.head(5)
            for i, (_, row) in enumerate(top_5.iterrows(), 1):
                risk_emoji = "🟢" if row['Risk_Index'] <= 2 else "🟡" if row['Risk_Index'] <= 3 else "🔴"
                st.write(f"{i}. **{row['Country']}** {risk_emoji}")
                st.write(f"   적합도: {row['Suitability_Score']:.1f}점 | "
                        f"수출액: ${row['Export_Value']:.1f}B | "
                        f"성장률: {row['Growth_Rate']:.1f}%")
        
        with col2:
            st.subheader("⚠️ 신중 검토 대상")
            bottom_5 = analyzed_df.tail(5)
            for i, (_, row) in enumerate(bottom_5.iterrows(), 1):
                risk_emoji = "🟢" if row['Risk_Index'] <= 2 else "🟡" if row['Risk_Index'] <= 3 else "🔴"
                st.write(f"{i}. **{row['Country']}** {risk_emoji}")
                st.write(f"   적합도: {row['Suitability_Score']:.1f}점 | "
                        f"위험도: {row['Risk_Index']} | "
                        f"연체율: {row['PDR_Rate']:.1f}%")
        
        # 레이더 차트 (상위 5개국)
        st.subheader("📡 상위 5개국 다차원 분석")
        top_5_countries = analyzed_df.head(5)
        
        categories = ['수출액', '성장률', '안전도', '결제안전도']
        
        fig_radar = go.Figure()
        
        for _, country in top_5_countries.iterrows():
            values = [
                country['Export_Score'],
                country['Growth_Score'],
                country['Safety_Score'], 
                country['Payment_Score']
            ]
            
            fig_radar.add_trace(go.Scatterpolar(
                r=values + [values[0]],  # 첫 번째 값을 마지막에 추가하여 폐곡선 만들기
                theta=categories + [categories[0]],
                fill='toself',
                name=country['Country'],
                opacity=0.6
            ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title="상위 5개국 종합 역량 비교"
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
    
    with tab3:
        st.header("🔍 K-means 군집 분석")
        
        # 군집별 특성 설명
        st.subheader("📋 군집별 전략적 시사점")
        
        cluster_descriptions = {
            '고성장-저위험': '🌟 **Star Markets**: 최우선 투자 대상, 공격적 확장 전략',
            '고성장-고위험': '❓ **Question Marks**: 선제적 진입, 리스크 관리 필수',
            '저성장-저위험': '💰 **Cash Cows**: 안정적 수익 창출, 현상 유지',
            '저성장-고위험': '🐕 **Dogs**: 전략적 철수 또는 최소 투자'
        }
        
        for cluster, description in cluster_descriptions.items():
            st.write(f"- {description}")
        
        # 군집별 국가 분포
        st.subheader("🗺️ 군집별 국가 분포")
        
        cluster_summary = clustered_df.groupby('Cluster_Label').agg({
            'Country': 'count',
            'Export_Value': 'mean',
            'Growth_Rate': 'mean', 
            'Risk_Index': 'mean',
            'Suitability_Score': 'mean'
        }).round(2)
        
        st.dataframe(cluster_summary)
        
        # 3D 군집 시각화
        st.subheader("🎲 3D 군집 분석")
        
        fig_3d = px.scatter_3d(
            clustered_df,
            x='Export_Value',
            y='Growth_Rate',
            z='Risk_Index',
            color='Cluster_Label',
            size='Suitability_Score',
            hover_name='Country',
            title="3차원 국가 포지셔닝"
        )
        
        st.plotly_chart(fig_3d, use_container_width=True)
    
    with tab4:
        st.header("📈 성장성 분석")
        
        # 성장률 히스토그램
        st.subheader("📊 성장률 분포")
        
        fig_hist = px.histogram(
            analyzed_df,
            x='Growth_Rate',
            nbins=10,
            title="국가별 성장률 분포",
            labels={'Growth_Rate': '성장률 (%)', 'count': '국가 수'}
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # 대륙별 성장률 박스플롯
        st.subheader("🌍 대륙별 성장률 비교")
        
        fig_box = px.box(
            analyzed_df,
            x='Continent',
            y='Growth_Rate',
            title="대륙별 성장률 분포"
        )
        st.plotly_chart(fig_box, use_container_width=True)
        
        # 성장률 상위/하위 국가
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🚀 고성장 시장 TOP 5")
            high_growth = analyzed_df.nlargest(5, 'Growth_Rate')
            for _, row in high_growth.iterrows():
                st.write(f"**{row['Country']}**: {row['Growth_Rate']:.1f}%")
        
        with col2:
            st.subheader("📉 저성장 시장 TOP 5")
            low_growth = analyzed_df.nsmallest(5, 'Growth_Rate')
            for _, row in low_growth.iterrows():
                st.write(f"**{row['Country']}**: {row['Growth_Rate']:.1f}%")
    
    with tab5:
        st.header("⚠️ 리스크 분석")
        
        # 위험도별 분류
        low_risk = analyzed_df[analyzed_df['Risk_Index'] <= 2]
        medium_risk = analyzed_df[(analyzed_df['Risk_Index'] > 2) & (analyzed_df['Risk_Index'] <= 3)]
        high_risk = analyzed_df[analyzed_df['Risk_Index'] > 3]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🟢 저위험 국가", len(low_risk), f"{len(low_risk)/len(analyzed_df)*100:.1f}%")
        with col2:
            st.metric("🟡 중위험 국가", len(medium_risk), f"{len(medium_risk)/len(analyzed_df)*100:.1f}%")
        with col3:
            st.metric("🔴 고위험 국가", len(high_risk), f"{len(high_risk)/len(analyzed_df)*100:.1f}%")
        
        # 위험도와 수출액 관계
        st.subheader("💰 위험도별 수출 현황")
        
        fig_risk = px.scatter(
            analyzed_df,
            x='Risk_Index',
            y='Export_Value',
            size='Growth_Rate',
            color='PDR_Rate',
            hover_name='Country',
            title="위험도 vs 수출액",
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig_risk, use_container_width=True)
        
        # 결제 위험 분석
        st.subheader("💳 결제 위험도 분석")
        
        fig_payment = px.bar(
            analyzed_df.sort_values('PDR_Rate', ascending=True),
            x='Country',
            y='PDR_Rate',
            color='Risk_Index',
            title="국가별 결제 연체율",
            color_continuous_scale='RdYlGn_r'
        )
        fig_payment.update_xaxes(tickangle=45)
        st.plotly_chart(fig_payment, use_container_width=True)
    
    with tab6:
        st.header("🎮 수출 적합도 시뮬레이션")
        
        st.write("가상 시나리오를 입력하여 수출 적합도를 예측해보세요.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📝 시나리오 입력")
            sim_country = st.text_input("국가명", "가상국가")
            sim_export = st.number_input("수출액 (억달러)", 0.0, 3000.0, 100.0)
            sim_growth = st.number_input("성장률 (%)", -50.0, 200.0, 20.0)
            sim_risk = st.slider("위험지수", 1, 5, 3)
            sim_pdr = st.number_input("연체율 (%)", 0.0, 50.0, 8.0)
        
        with col2:
            st.subheader("🎯 예측 결과")
            
            # 시뮬레이션 데이터 생성
            sim_data = pd.DataFrame({
                'Country': [sim_country],
                'Export_Value': [sim_export],
                'Growth_Rate': [sim_growth],
                'Risk_Index': [sim_risk],
                'PDR_Rate': [sim_pdr]
            })
            
            # 기존 데이터와 합쳐서 정규화
            combined_data = pd.concat([df, sim_data], ignore_index=True)
            sim_analyzed = calculate_export_suitability(combined_data, weights)
            sim_result = sim_analyzed.iloc[-1]
            
            # 결과 표시
            st.metric("수출 적합도", f"{sim_result['Suitability_Score']:.1f}점")
            
            # 적합도 등급
            score = sim_result['Suitability_Score']
            if score >= 80:
                grade = "🌟 최우수"
                color = "green"
            elif score >= 60:
                grade = "✅ 우수"
                color = "blue"
            elif score >= 40:
                grade = "⚠️ 보통"
                color = "orange"
            else:
                grade = "❌ 부적합"
                color = "red"
            
            st.markdown(f"**적합도 등급**: <span style='color:{color}'>{grade}</span>", 
                       unsafe_allow_html=True)
            
            # 개별 점수 표시
            st.write("**세부 점수:**")
            st.write(f"- 수출 점수: {sim_result['Export_Score']:.1f}점")
            st.write(f"- 성장 점수: {sim_result['Growth_Score']:.1f}점") 
            st.write(f"- 안전 점수: {sim_result['Safety_Score']:.1f}점")
            st.write(f"- 결제 점수: {sim_result['Payment_Score']:.1f}점")
        
        # 유사 국가 추천
        st.subheader("🔍 유사 국가 분석")
        
        # 입력값과 유사한 국가 찾기
        feature_weights = [0.3, 0.3, 0.2, 0.2]  # 각 특성의 중요도
        
        distances = []
        for _, row in analyzed_df.iterrows():
            distance = (
                feature_weights[0] * abs(row['Export_Value'] - sim_export) / analyzed_df['Export_Value'].std() +
                feature_weights[1] * abs(row['Growth_Rate'] - sim_growth) / analyzed_df['Growth_Rate'].std() +
                feature_weights[2] * abs(row['Risk_Index'] - sim_risk) / analyzed_df['Risk_Index'].std() +
                feature_weights[3] * abs(row['PDR_Rate'] - sim_pdr) / analyzed_df['PDR_Rate'].std()
            )
            distances.append(distance)
        
        analyzed_df['Similarity'] = distances
        similar_countries = analyzed_df.nsmallest(3, 'Similarity')
        
        st.write("**가장 유사한 3개국:**")
        for i, (_, row) in enumerate(similar_countries.iterrows(), 1):
            st.write(f"{i}. **{row['Country']}** - 적합도: {row['Suitability_Score']:.1f}점")

    # 푸터
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>K-Beauty Export Optimizer (KBEO) v1.0 | 
        Developed by 미생s 팀 | 
        Data: KITA, KOTRA, K-SURE</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

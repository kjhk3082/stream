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

# 페이지 설정 (반드시 첫 번째 명령어)
st.set_page_config(
    page_title="K-Beauty Export Optimizer (KBEO)",
    page_icon="🌟",  # 💄 아이콘을 🌟로 변경
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
    
    .stTab {
        font-size: 1.2rem;
        font-weight: bold;
    }
    
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# 데이터 로딩 함수
@st.cache_data
def load_export_data():
    """실제 화장품 수출 데이터 로딩"""
    # 2024년 화장품 수출 데이터 (상위 30개국)
    data = {
        'Country': [
            '중국', '미국', '일본', '베트남', '홍콩', '러시아', '대만', '태국', 
            '싱가포르', '아랍에미리트', '영국', '말레이시아', '폴란드', '인도네시아', 
            '캐나다', '호주', '카자흐스탄', '필리핀', '네덜란드', '키르기스스탄',
            '독일', '프랑스', '우크라이나', '미얀마', '인도', '몽골', 
            '사우디아라비아', '스페인', '브라질', '이라크'
        ],
        'Export_Value': [
            2156.3, 1547.6, 840.4, 466.1, 511.1, 322.3, 218.8, 186.4, 
            117.2, 158.3, 133.0, 112.1, 112.4, 118.9, 103.4, 96.0, 
            83.7, 76.7, 74.6, 73.9, 58.9, 67.6, 54.7, 48.5, 
            65.6, 36.9, 54.4, 30.7, 28.8, 27.7
        ],
        'Growth_Rate': [
            -9.3, 51.3, 26.0, 4.8, 16.6, 2.0, 31.5, 12.6, 
            14.6, 87.2, 46.5, 26.2, 154.2, 73.8, 54.9, 56.4, 
            32.7, 33.5, 33.5, 19.3, 45.9, 1.8, -3.5, 18.2,
            63.9, 18.7, 102.4, 39.8, 73.1, 121.9
        ],
        'Risk_Index': [
            4, 2, 1, 4, 3, 5, 2, 3, 2, 3, 2, 3, 3, 4, 2, 2, 
            4, 4, 1, 4, 2, 3, 5, 5, 4, 4, 3, 3, 4, 5
        ],
        'PDR_Rate': [
            8.5, 3.2, 2.1, 12.3, 6.8, 18.9, 4.5, 8.7, 3.8, 7.2, 
            4.1, 9.1, 6.5, 15.2, 3.9, 2.8, 14.7, 11.8, 2.3, 16.4,
            3.5, 4.2, 22.1, 19.8, 13.5, 12.9, 8.3, 5.1, 11.2, 17.6
        ],
        'OA_Ratio': [
            78.4, 82.8, 82.6, 67.3, 78.9, 94.4, 85.2, 74.8,
            83.6, 71.2, 68.3, 70.6, 68.9, 72.1, 92.0, 74.2,
            81.5, 69.7, 84.8, 73.4, 75.8, 72.3, 85.7, 88.2,
            74.6, 79.1, 67.1, 73.9, 85.7, 82.3
        ],
        'Continent': [
            'Asia', 'North America', 'Asia', 'Asia', 'Asia', 'Europe', 'Asia', 'Asia',
            'Asia', 'Asia', 'Europe', 'Asia', 'Europe', 'Asia', 'North America', 
            'Oceania', 'Asia', 'Asia', 'Europe', 'Asia', 'Europe', 'Europe',
            'Europe', 'Asia', 'Asia', 'Asia', 'Asia', 'Europe', 'South America', 'Asia'
        ]
    }
    return pd.DataFrame(data)

# 데이터 정리 함수
def clean_data(df):
    """데이터에서 NaN 값 처리"""
    df = df.copy()
    
    # NaN 값을 처리
    numeric_columns = ['Export_Value', 'Growth_Rate', 'Risk_Index', 'PDR_Rate', 'OA_Ratio']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())
    
    # 무한값 처리
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.median(numeric_only=True))
    
    return df

# MinMax 정규화 함수
def minmax_normalize(series):
    """MinMax 정규화 수행"""
    if series.max() == series.min():
        return pd.Series([50] * len(series), index=series.index)
    return 100 * (series - series.min()) / (series.max() - series.min())

# 수출 적합도 점수 계산
def calculate_export_suitability(df, weights):
    """가중합 기반 수출 적합도 점수 계산"""
    df_copy = clean_data(df)
    
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
    df_clean = clean_data(df)
    features = ['Export_Value', 'Growth_Rate', 'Risk_Index', 'PDR_Rate']
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_clean[features])
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df_clean['Cluster'] = kmeans.fit_predict(scaled_features)
    
    # 클러스터 라벨링
    cluster_labels = {
        0: '고성장-저위험',
        1: '고성장-고위험', 
        2: '저성장-저위험',
        3: '저성장-고위험'
    }
    df_clean['Cluster_Label'] = df_clean['Cluster'].map(cluster_labels)
    
    return df_clean, kmeans, scaler

# 백테스팅 함수
def perform_backtesting(df, weights, years=['2022', '2023', '2024']):
    """백테스팅 수행"""
    # 가상의 과거 성과 데이터 생성
    results = []
    for year in years:
        temp_df = clean_data(df)
        temp_df['Year'] = year
        
        # 연도별 성과 변동 시뮬레이션
        if year == '2022':
            temp_df['Growth_Rate'] = temp_df['Growth_Rate'] * 0.8
        elif year == '2023':
            temp_df['Growth_Rate'] = temp_df['Growth_Rate'] * 0.9
        
        analyzed = calculate_export_suitability(temp_df, weights)
        
        # 상위 10개국 선정
        top_10 = analyzed.nlargest(10, 'Suitability_Score')
        avg_growth = top_10['Growth_Rate'].mean()
        hit_rate = len(top_10[top_10['Growth_Rate'] > 0]) / len(top_10) * 100
        
        results.append({
            'Year': year,
            'Avg_Growth': avg_growth,
            'Hit_Rate': hit_rate,
            'Top_Countries': top_10['Country'].tolist()[:5]
        })
    
    return results

# 안전한 plotly 차트 생성 함수
def create_safe_scatter(df, x, y, size=None, color=None, hover_name=None, **kwargs):
    """NaN 값을 처리한 안전한 scatter plot 생성"""
    df_plot = df.copy()
    
    # 필수 컬럼 체크 및 NaN 처리
    required_cols = [x, y]
    if size:
        required_cols.append(size)
    if color:
        required_cols.append(color)
        
    for col in required_cols:
        if col in df_plot.columns:
            df_plot[col] = pd.to_numeric(df_plot[col], errors='coerce')
    
    # NaN이 있는 행 제거
    df_plot = df_plot.dropna(subset=required_cols)
    
    # 무한값 처리
    df_plot = df_plot.replace([np.inf, -np.inf], np.nan).dropna(subset=required_cols)
    
    if len(df_plot) == 0:
        # 빈 차트 반환
        fig = go.Figure()
        fig.add_annotation(text="데이터가 없습니다", x=0.5, y=0.5, showarrow=False)
        return fig
    
    try:
        fig = px.scatter(df_plot, x=x, y=y, size=size, color=color, hover_name=hover_name, **kwargs)
        return fig
    except Exception as e:
        # 에러 발생 시 기본 차트 반환
        fig = go.Figure()
        fig.add_annotation(text=f"차트 생성 오류: {str(e)[:50]}...", x=0.5, y=0.5, showarrow=False)
        return fig

# 메인 애플리케이션
def main():
    # 헤더 (아이콘 변경)
    st.markdown('<h1 class="main-header">🌟 K-Beauty Export Optimizer (KBEO)</h1>', 
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
            # 자동 조정
            factor = 100 / total if total > 0 else 1
            export_weight = int(export_weight * factor)
            growth_weight = int(growth_weight * factor)
            safety_weight = int(safety_weight * factor)
            payment_weight = 100 - export_weight - growth_weight - safety_weight
        
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
    top_n = st.sidebar.slider("분석 대상 국가 수", 5, 30, 20)
    
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
        
        # 동적 텍스트 적용: 분석 대상 국가 수에 맞춰 제목 변경
        actual_countries = len(analyzed_df)
        st.subheader(f"🏆 상위 {min(10, actual_countries)}개국 수출 적합도")
        top_display = analyzed_df.head(min(10, actual_countries))
        
        fig_bar = px.bar(
            top_display, 
            x='Country', 
            y='Suitability_Score',
            color='Risk_Index',
            color_continuous_scale='RdYlGn_r',
            title=f"{selected_strategy} 전략 기준 수출 적합도 (총 {actual_countries}개국 중 상위 {len(top_display)}개국)",
            labels={
                'Country': '국가',
                'Suitability_Score': '수출 적합도 점수',
                'Risk_Index': '위험지수'
            }
        )
        fig_bar.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # 수출액 vs 성장률 산점도 (BCG 매트릭스) - 안전한 버전
        st.subheader("📈 BCG 매트릭스 (수출액 vs 성장률)")
        
        fig_scatter = create_safe_scatter(
            analyzed_df,
            x='Export_Value',
            y='Growth_Rate', 
            size='Suitability_Score',
            color='Risk_Index',
            hover_name='Country',
            color_continuous_scale='RdYlGn_r',
            title="BCG 매트릭스 분석",
            labels={
                'Export_Value': '수출액 (억달러)',
                'Growth_Rate': '성장률 (%)',
                'Risk_Index': '위험지수',
                'Suitability_Score': '적합도 점수'
            }
        )
        
        # 사분면 구분선 추가
        if len(analyzed_df) > 0:
            median_export = analyzed_df['Export_Value'].median()
            median_growth = analyzed_df['Growth_Rate'].median()
            
            fig_scatter.add_hline(y=median_growth, line_dash="dash", line_color="gray", 
                                 annotation_text="성장률 중위값")
            fig_scatter.add_vline(x=median_export, line_dash="dash", line_color="gray",
                                 annotation_text="수출액 중위값")
        
        fig_scatter.update_layout(height=600)
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # 대륙별 수출 현황
        st.subheader("🌍 대륙별 수출 현황")
        
        continent_summary = analyzed_df.groupby('Continent').agg({
            'Export_Value': 'sum',
            'Growth_Rate': 'mean',
            'Suitability_Score': 'mean',
            'Country': 'count'
        }).round(2)
        continent_summary.columns = ['총수출액', '평균성장률', '평균적합도', '국가수']
        
        col1, col2 = st.columns(2)
        
        with col1:
            if len(continent_summary) > 0:
                fig_pie = px.pie(
                    values=continent_summary['총수출액'],
                    names=continent_summary.index,
                    title="대륙별 수출액 비중"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.dataframe(continent_summary, use_container_width=True)
    
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
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        for idx, (_, country) in enumerate(top_5_countries.iterrows()):
            values = [
                country['Export_Score'],
                country['Growth_Score'],
                country['Safety_Score'], 
                country['Payment_Score']
            ]
            
            fig_radar.add_trace(go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill='toself',
                name=country['Country'],
                opacity=0.6,
                line_color=colors[idx % len(colors)]
            ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title="상위 5개국 종합 역량 비교",
            height=500
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # 전략별 비교 분석
        st.subheader("📊 전략별 성과 비교")
        
        strategy_comparison = {}
        for strategy_name, strategy_weights in strategy_options.items():
            if strategy_weights is not None:
                temp_analysis = calculate_export_suitability(filtered_df, strategy_weights)
                top_3 = temp_analysis.nlargest(3, 'Suitability_Score')
                strategy_comparison[strategy_name] = {
                    'avg_score': temp_analysis['Suitability_Score'].mean(),
                    'top_countries': ', '.join(top_3['Country'].tolist()),
                    'avg_growth': top_3['Growth_Rate'].mean(),
                    'avg_risk': top_3['Risk_Index'].mean()
                }
        
        comparison_df = pd.DataFrame(strategy_comparison).T
        comparison_df.columns = ['평균점수', '상위3개국', '평균성장률', '평균위험도']
        comparison_df = comparison_df.round(2)
        
        st.dataframe(comparison_df, use_container_width=True)
    
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
            'Country': lambda x: ', '.join(x.head(5).tolist()),
            'Export_Value': 'mean',
            'Growth_Rate': 'mean', 
            'Risk_Index': 'mean',
            'Suitability_Score': 'mean'
        }).round(2)
        cluster_summary.columns = ['주요국가(상위5)', '평균수출액', '평균성장률', '평균위험도', '평균적합도']
        
        st.dataframe(cluster_summary, use_container_width=True)
        
        # 3D 군집 시각화 - 안전한 버전
        st.subheader("🎲 3D 군집 분석")
        
        try:
            fig_3d = px.scatter_3d(
                clustered_df,
                x='Export_Value',
                y='Growth_Rate',
                z='Risk_Index',
                color='Cluster_Label',
                size='Suitability_Score',
                hover_name='Country',
                title="3차원 국가 포지셔닝",
                labels={
                    'Export_Value': '수출액',
                    'Growth_Rate': '성장률',
                    'Risk_Index': '위험지수'
                }
            )
            st.plotly_chart(fig_3d, use_container_width=True)
        except Exception as e:
            st.error(f"3D 차트 생성 중 오류가 발생했습니다: {str(e)}")
            # 대체 2D 차트 제공
            fig_2d = create_safe_scatter(
                clustered_df,
                x='Export_Value',
                y='Growth_Rate',
                color='Cluster_Label',
                hover_name='Country',
                title="2D 국가 포지셔닝 (3D 차트 대체)"
            )
            st.plotly_chart(fig_2d, use_container_width=True)
    
    with tab4:
        st.header("📈 성장성 분석")
        
        # 성장률 히스토그램
        st.subheader("📊 성장률 분포")
        
        fig_hist = px.histogram(
            analyzed_df,
            x='Growth_Rate',
            nbins=15,
            title="국가별 성장률 분포",
            labels={'Growth_Rate': '성장률 (%)', 'count': '국가 수'},
            color_discrete_sequence=['#FF6B6B']
        )
        
        # 평균선 추가
        avg_growth = analyzed_df['Growth_Rate'].mean()
        fig_hist.add_vline(x=avg_growth, line_dash="dash", line_color="red", 
                          annotation_text=f"평균: {avg_growth:.1f}%")
        
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # 대륙별 성장률 박스플롯
        st.subheader("🌍 대륙별 성장률 비교")
        
        fig_box = px.box(
            analyzed_df,
            x='Continent',
            y='Growth_Rate',
            title="대륙별 성장률 분포",
            color='Continent',
            labels={'Growth_Rate': '성장률 (%)', 'Continent': '대륙'}
        )
        st.plotly_chart(fig_box, use_container_width=True)
        
        # 성장률 vs 수출액 관계 - 안전한 버전
        st.subheader("💹 성장률과 수출액의 관계")
        
        fig_growth_export = create_safe_scatter(
            analyzed_df,
            x='Export_Value',
            y='Growth_Rate',
            size='Suitability_Score',
            color='Continent',
            hover_name='Country',
            title="성장률 vs 수출액",
            labels={
                'Export_Value': '수출액 (억달러)',
                'Growth_Rate': '성장률 (%)'
            }
        )
        
        st.plotly_chart(fig_growth_export, use_container_width=True)
        
        # 성장률 상위/하위 국가
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🚀 고성장 시장 TOP 10")
            high_growth = analyzed_df.nlargest(10, 'Growth_Rate')
            for i, (_, row) in enumerate(high_growth.iterrows(), 1):
                st.write(f"{i}. **{row['Country']}**: {row['Growth_Rate']:.1f}%")
        
        with col2:
            st.subheader("📉 저성장 시장 TOP 10")
            low_growth = analyzed_df.nsmallest(10, 'Growth_Rate')
            for i, (_, row) in enumerate(low_growth.iterrows(), 1):
                st.write(f"{i}. **{row['Country']}**: {row['Growth_Rate']:.1f}%")
    
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
        
        # 위험도와 수출액 관계 - 안전한 버전
        st.subheader("💰 위험도별 수출 현황")
        
        fig_risk = create_safe_scatter(
            analyzed_df,
            x='Risk_Index',
            y='Export_Value',
            size='Growth_Rate',
            color='PDR_Rate',
            hover_name='Country',
            title="위험도 vs 수출액",
            labels={
                'Risk_Index': '위험지수',
                'Export_Value': '수출액 (억달러)',
                'PDR_Rate': '연체율 (%)'
            },
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig_risk, use_container_width=True)
        
        # 결제 위험 분석
        st.subheader("💳 결제 위험도 분석")
        
        fig_payment = px.bar(
            analyzed_df.sort_values('PDR_Rate', ascending=False).head(15),
            x='Country',
            y='PDR_Rate',
            color='Risk_Index',
            title="국가별 결제 연체율 (상위 15개국)",
            labels={
                'Country': '국가',
                'PDR_Rate': '연체율 (%)',
                'Risk_Index': '위험지수'
            },
            color_continuous_scale='RdYlGn_r'
        )
        fig_payment.update_xaxes(tickangle=-45)
        st.plotly_chart(fig_payment, use_container_width=True)
        
        # O/A 비율 분석 - 안전한 버전
        st.subheader("📋 외상거래(O/A) 비율 분석")
        
        high_oa = analyzed_df[analyzed_df['OA_Ratio'] > 80].sort_values('OA_Ratio', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**O/A 비율 80% 이상 국가:**")
            for _, row in high_oa.iterrows():
                risk_level = "🔴" if row['Risk_Index'] > 3 else "🟡" if row['Risk_Index'] > 2 else "🟢"
                st.write(f"• {row['Country']}: {row['OA_Ratio']:.1f}% {risk_level}")
        
        with col2:
            fig_oa = create_safe_scatter(
                analyzed_df,
                x='OA_Ratio',
                y='PDR_Rate',
                size='Export_Value',
                color='Risk_Index',
                hover_name='Country',
                title="O/A 비율 vs 연체율",
                labels={
                    'OA_Ratio': 'O/A 비율 (%)',
                    'PDR_Rate': '연체율 (%)'
                }
            )
            st.plotly_chart(fig_oa, use_container_width=True)
    
    with tab6:
        st.header("🎮 수출 적합도 시뮬레이션")
        
        # 백테스팅 결과 먼저 표시
        st.subheader("📊 전략별 백테스팅 결과")
        
        backtesting_results = {}
        for strategy_name, strategy_weights in strategy_options.items():
            if strategy_weights is not None:
                results = perform_backtesting(df, strategy_weights)
                avg_growth = np.mean([r['Avg_Growth'] for r in results])
                avg_hit_rate = np.mean([r['Hit_Rate'] for r in results])
                backtesting_results[strategy_name] = {
                    'avg_growth': avg_growth,
                    'avg_hit_rate': avg_hit_rate
                }
        
        backtesting_df = pd.DataFrame(backtesting_results).T
        backtesting_df.columns = ['평균 성장률 (%)', '적중률 (%)']
        backtesting_df = backtesting_df.round(2)
        
        # 성과 순으로 정렬
        backtesting_df = backtesting_df.sort_values('적중률 (%)', ascending=False)
        
        st.dataframe(backtesting_df, use_container_width=True)
        
        # 최고 성과 전략 하이라이트
        best_strategy = backtesting_df.index[0]
        st.success(f"🏆 **최우수 전략**: {best_strategy} "
                  f"(적중률: {backtesting_df.loc[best_strategy, '적중률 (%)']:.1f}%, "
                  f"평균성장률: {backtesting_df.loc[best_strategy, '평균 성장률 (%)']:.1f}%)")
        
        st.subheader("🎯 가상 시나리오 분석")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📝 시나리오 입력")
            sim_country = st.text_input("국가명", "가상국가")
            sim_export = st.number_input("수출액 (억달러)", 0.0, 3000.0, 100.0)
            sim_growth = st.number_input("성장률 (%)", -50.0, 200.0, 20.0)
            sim_risk = st.slider("위험지수", 1, 5, 3)
            sim_pdr = st.number_input("연체율 (%)", 0.0, 50.0, 8.0)
            sim_oa = st.number_input("O/A 비율 (%)", 0.0, 100.0, 75.0)
            sim_continent = st.selectbox("대륙", ['Asia', 'Europe', 'North America', 'South America', 'Oceania', 'Africa'])
        
        with col2:
            st.subheader("🎯 예측 결과")
            
            # 시뮬레이션 데이터 생성
            sim_data = pd.DataFrame({
                'Country': [sim_country],
                'Export_Value': [sim_export],
                'Growth_Rate': [sim_growth],
                'Risk_Index': [sim_risk],
                'PDR_Rate': [sim_pdr],
                'OA_Ratio': [sim_oa],
                'Continent': [sim_continent]
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
            
            # 백분위 순위
            rank = (analyzed_df['Suitability_Score'] < score).sum() + 1
            percentile = (rank / len(analyzed_df)) * 100
            st.write(f"**순위**: {len(analyzed_df)}개국 중 {rank}위 (상위 {100-percentile:.1f}%)")
        
        # 유사 국가 추천
        st.subheader("🔍 유사 국가 분석")
        
        # 입력값과 유사한 국가 찾기
        feature_weights = [0.3, 0.3, 0.2, 0.2]  # 각 특성의 중요도
        
        distances = []
        for _, row in analyzed_df.iterrows():
            # 정규화된 거리 계산
            distance = (
                feature_weights[0] * abs(row['Export_Value'] - sim_export) / (analyzed_df['Export_Value'].max() - analyzed_df['Export_Value'].min()) +
                feature_weights[1] * abs(row['Growth_Rate'] - sim_growth) / (analyzed_df['Growth_Rate'].max() - analyzed_df['Growth_Rate'].min()) +
                feature_weights[2] * abs(row['Risk_Index'] - sim_risk) / 4 +  # 위험지수는 1-5 범위
                feature_weights[3] * abs(row['PDR_Rate'] - sim_pdr) / (analyzed_df['PDR_Rate'].max() - analyzed_df['PDR_Rate'].min())
            )
            distances.append(distance)
        
        analyzed_df_copy = analyzed_df.copy()
        analyzed_df_copy['Similarity'] = distances
        similar_countries = analyzed_df_copy.nsmallest(5, 'Similarity')
        
        st.write("**가장 유사한 5개국:**")
        for i, (_, row) in enumerate(similar_countries.iterrows(), 1):
            similarity_pct = (1 - row['Similarity']) * 100
            st.write(f"{i}. **{row['Country']}** (유사도: {similarity_pct:.1f}%) - "
                    f"적합도: {row['Suitability_Score']:.1f}점")
        
        # 시나리오 비교 차트
        st.subheader("📊 시나리오 비교 분석")
        
        comparison_data = pd.concat([
            similar_countries.head(3)[['Country', 'Export_Value', 'Growth_Rate', 'Risk_Index', 'Suitability_Score']],
            pd.DataFrame({
                'Country': [sim_country],
                'Export_Value': [sim_export],
                'Growth_Rate': [sim_growth],
                'Risk_Index': [sim_risk],
                'Suitability_Score': [sim_result['Suitability_Score']]
            })
        ])
        
        fig_comparison = px.bar(
            comparison_data,
            x='Country',
            y='Suitability_Score',
            color='Risk_Index',
            title="유사 국가 대비 수출 적합도 비교",
            color_continuous_scale='RdYlGn_r'
        )
        st.plotly_chart(fig_comparison, use_container_width=True)

    # 푸터
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>🌟 K-Beauty Export Optimizer (KBEO) v2.0 | 
        Developed by 미생s 팀 (장효석, 김성호, 김재형) | 
        Data: KITA, KOTRA, K-SURE</p>
        <p>📧 Contact: misaengs.team@gmail.com | 
        📅 Last Updated: 2025.06.13</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

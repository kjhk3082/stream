import streamlit as st
import pandas as pd
import numpy as np
from data_preprocessing import get_processed_data # Assuming data_preprocessing.py is in the same directory

# Load data
@st.cache_data
def load_data():
    # Use the provided CSV file directly
    df = get_processed_data("comprehensive_cosmetics_export_analysis.csv")
    return df

data = load_data()

# Calculate score based on Min-Max normalization and weighted sum
def calculate_score(item, weights, all_data):
    if not item or pd.isna(item["Export_Value"]) or pd.isna(item["Growth_Rate"]) or pd.isna(item["Risk_Score"]):
        return 0

    export_values = all_data["Export_Value"].dropna()
    growth_rates = all_data["Growth_Rate"].dropna()
    risk_scores = all_data["Risk_Score"].dropna()

    min_export = export_values.min()
    max_export = export_values.max()
    min_growth = growth_rates.min()
    max_growth = growth_rates.max()
    min_risk = risk_scores.min()
    max_risk = risk_scores.max()

    normalized_export = 100 * (item["Export_Value"] - min_export) / (max_export - min_export) if (max_export - min_export) != 0 else 0
    normalized_growth = 100 * (item["Growth_Rate"] - min_growth) / (max_growth - min_growth) if (max_growth - min_growth) != 0 else 0
    normalized_safety = 100 * (max_risk - item["Risk_Score"]) / (max_risk - min_risk) if (max_risk - min_risk) != 0 else 0

    score = (normalized_export * weights["export"] / 100) + \
            (normalized_growth * weights["growth"] / 100) + \
            (normalized_safety * weights["safety"] / 100)

    return score

# Streamlit App
st.set_page_config(layout="wide", page_title="화장품 수출 적합도 분석 대시보드")

st.title("화장품 수출 적합도 분석 대시보드")

# Scenario selection
st.sidebar.header("가중치 시나리오 선택")
scenario_options = {
    "안전 중심": {"export": 20, "growth": 20, "safety": 60},
    "성장률 중심": {"export": 20, "growth": 60, "safety": 20},
    "수출액 중심": {"export": 60, "growth": 20, "safety": 20},
    "밸런스 중심": {"export": 33, "growth": 33, "safety": 34},
    "사용자 정의": None
}

selected_scenario_name = st.sidebar.radio("시나리오", list(scenario_options.keys()))

if selected_scenario_name == "사용자 정의":
    st.sidebar.subheader("사용자 정의 가중치 (%)")
    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        custom_export = st.number_input("수출액", min_value=0, max_value=100, value=33, key="custom_export")
    with col2:
        custom_growth = st.number_input("성장률", min_value=0, max_value=100, value=33, key="custom_growth")
    with col3:
        custom_safety = st.number_input("안전도", min_value=0, max_value=100, value=34, key="custom_safety")
    
    total_custom_weights = custom_export + custom_growth + custom_safety
    if total_custom_weights != 100:
        st.sidebar.warning(f"가중치 총합이 100%가 아닙니다! (현재: {total_custom_weights}%) - 자동으로 조정됩니다.")
        # Simple adjustment for demonstration, more robust logic might be needed
        if total_custom_weights > 0:
            ratio = 100 / total_custom_weights
            custom_export = round(custom_export * ratio)
            custom_growth = round(custom_growth * ratio)
            custom_safety = 100 - custom_export - custom_growth # Ensure sum is 100
        else:
            custom_export = 33
            custom_growth = 33
            custom_safety = 34

    selected_weights = {"export": custom_export, "growth": custom_growth, "safety": custom_safety}
else:
    selected_weights = scenario_options[selected_scenario_name]

st.sidebar.write(f"현재 가중치: 수출액 {selected_weights["export"]}%, 성장률 {selected_weights["growth"]}%, 안전도 {selected_weights["safety"]}% (총합: {selected_weights["export"] + selected_weights["growth"] + selected_weights["safety"]}%) ")

# Calculate scores for all countries based on selected weights
scored_data = data.copy()
scored_data["score"] = scored_data.apply(lambda row: calculate_score(row, selected_weights, data), axis=1)
scored_data = scored_data.sort_values(by="score", ascending=False).reset_index(drop=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Scenario Comparison", "Risk Analysis", "Score Simulation"])

with tab1:
    st.header("Dashboard")

    # Top N countries selection
    top_n = st.slider("상위 국가 수", min_value=5, max_value=20, value=10, step=5)
    top_countries = scored_data.head(top_n)
    bottom_countries = scored_data.tail(5)

    st.subheader(f"상위 {top_n}개국 수출 적합도 점수")
    st.bar_chart(top_countries.set_index("Country")["score"])

    st.subheader("주요 지표")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Highest Export Value", f"{data["Export_Value"].max():,.0f} 천불",
                  f"{data.loc[data["Export_Value"].idxmax(), "Country"]}")
    with col2:
        st.metric("Highest Growth Rate", f"{data["Growth_Rate"].max():.2f}%",
                  f"{data.loc[data["Growth_Rate"].idxmax(), "Country"]}")
    with col3:
        st.metric("Lowest Risk", f"{data["Risk_Score"].min():.1f}",
                  f"{data.loc[data["Risk_Score"].idxmin(), "Country"]}")

    st.subheader("하이리스크 비추천 국가 (하위 5개국)")
    st.bar_chart(bottom_countries.set_index("Country")["score"])

with tab2:
    st.header("시나리오별 상위 5개국 비교")
    
    scenario_comparison_data = []
    for s_name, s_weights in scenario_options.items():
        if s_weights is not None: # Exclude custom for this comparison
            temp_data = data.copy()
            temp_data["score"] = temp_data.apply(lambda row: calculate_score(row, s_weights, data), axis=1)
            top_5 = temp_data.sort_values(by="score", ascending=False).head(5)
            scenario_comparison_data.append({"scenario": s_name, "countries": top_5})

    for sc in scenario_comparison_data:
        st.subheader(f"{sc["scenario"]} 시나리오")
        for idx, row in sc["countries"].iterrows():
            st.write(f"{idx+1}. {row["Country"]} ({row["score"]:.1f}점)")

with tab3:
    st.header("리스크 분석")
    st.write("리스크 점수와 적합도 점수 간의 관계를 시각화합니다.")

    # Using Balanced_Score for Risk Analysis as per previous logic
    risk_analysis_data = data.copy()
    risk_analysis_data["score"] = risk_analysis_data["Balanced_Score"]

    st.scatter_chart(risk_analysis_data, x="Risk_Score", y="score", size="Export_Value", color="Risk_Score")

    st.subheader("리스크 분류")
    st.write("저위험: 리스크 점수 ≤ 2.5")
    st.write("중위험: 2.5 < 리스크 점수 ≤ 3.5")
    st.write("고위험: 리스크 점수 > 3.5")

with tab4:
    st.header("수출 적합도 시뮬레이션")
    st.write("가상의 국가 데이터를 입력하여 적합도 점수를 시뮬레이션합니다.")

    sim_country = st.text_input("국가명", "가상국가")
    sim_export = st.number_input("수출액 (천불)", min_value=0.0, value=10000.0)
    sim_growth = st.number_input("성장률 (%)", value=10.0)
    sim_risk = st.slider("리스크 점수 (1-5)", min_value=1.0, max_value=5.0, value=3.0, step=0.1)

    simulated_item = {
        "Country": sim_country,
        "Export_Value": sim_export,
        "Growth_Rate": sim_growth,
        "Risk_Score": sim_risk
    }

    simulated_score = calculate_score(simulated_item, selected_weights, data)

    st.subheader("시뮬레이션 결과")
    st.write(f"선택된 시나리오: {selected_scenario_name}")
    st.write(f"수출액 가중치: {selected_weights["export"]}%, 성장률 가중치: {selected_weights["growth"]}%, 안전도 가중치: {selected_weights["safety"]}%")
    st.write(f"적합도 점수: **{simulated_score:.2f}점**")

    if simulated_score >= 70:
        st.write("적합도 등급: **고적합도**")
    elif simulated_score >= 40:
        st.write("적합도 등급: **중적합도**")
    else:
        st.write("적합도 등급: **저적합도**")

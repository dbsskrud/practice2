import json
import os
import re
from typing import Dict, List, Tuple

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from streamlit_folium import st_folium

# [기존 상수 및 유틸리티 함수 생략 - 제공된 코드와 동일하게 유지]
# (DISTRICTS, DISTRICT_CENTERS, EMBEDDED_RENT, DISTRICT_REVIEWS, 
#  LOW_BUDGET_WARNINGS, CHECKLIST_ITEMS, CULTURE_CATEGORY_LABELS, 
#  THEME_COLORS, DEFAULT_UNI_TO_DISTRICTS, WORK_TO_DISTRICTS, 
#  LINE_PREFS, EXTENDED_LINES, LINE_CONGESTION, PRIORITY_OPTIONS, 
#  PRICE_BAND_HELP, read_csv_safely, extract_district, format_lines, 
#  chip_html, normalize_type, rank_badge, current_destination_bucket, 
#  realistic_tip, transport_mismatch_reason, rent_band_filter, 
#  priority_weights, estimate_monthly_pressure 등)

# 데이터 로드 함수들 (캐싱 포함)
# [기존 load_base_data, load_rent_summary, build_district_dataframe, 
#  transport_match_score, score_recommendations, build_reason, 
#  load_geojson, make_rank_map 등 함수 내용 유지]

# ============================================================
# Streamlit 실행부 시작
# ============================================================

st.set_page_config(
    page_title="서울, 처음이니? : 어디서 자취할까?",
    page_icon="🏠",
    layout="wide",
)

# 스타일 적용
# [기존 st.markdown 내 CSS 스타일 코드 유지]

# 데이터 준비
base_df, culture_df, library_df, subway_df, prices_df, parks_df, culture_type_top_df, district_lines_map, seoul_avg_rent, using_rent_csv = build_district_dataframe()
seoul_market_avg = int(round(prices_df["가격(원)"].mean(), 0))

# 1. 타이틀 (Hero Section)
st.markdown(
    """
    <div class="hero">
        <h1>서울, 처음이니? : 어디서 자취할까?</h1>
        <p>서울에 처음 올라온 2030 자취생을 위해 만든 지역 추천 서비스예요. 
        중앙 상단에서 조건을 설정하고, 지도를 통해 서울의 분위기를 확인해 보세요.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# 2. 추천 조건 설정 (중앙 상단으로 이동)
with st.container():
    with st.expander("⚙️ 나에게 딱 맞는 자취방 조건 설정하기", expanded=True):
        f1, f2, f3 = st.columns([1.2, 1, 1.5])
        
        with f1:
            st.markdown("**🎯 우선순위 설정**")
            p1 = st.selectbox("1순위", PRIORITY_OPTIONS, index=0, key="p1")
            remain = [x for x in PRIORITY_OPTIONS if x != p1]
            p2 = st.selectbox("2순위", remain, index=0, key="p2")
            order = [p1, p2] + [x for x in remain if x != p2]
            
        with f2:
            st.markdown("**💸 희망 월세**")
            rent_band = st.selectbox(
                "월세 가격대",
                ["상관없음", "50만원대 이하", "60만원대", "70만원대", "80만원대", "90만원대 이상"],
                index=2,
            )
            st.caption(PRICE_BAND_HELP[rent_band] if rent_band != "상관없음" else "전체 지역을 비교합니다.")

        with f3:
            st.markdown("**🚇 주요 이동 동선**")
            sub_c1, sub_c2 = st.columns(2)
            with sub_c1:
                university = st.selectbox("대학교", list(DEFAULT_UNI_TO_DISTRICTS.keys()), index=0)
            with sub_c2:
                work_place = st.selectbox("근무지/업무지구", list(WORK_TO_DISTRICTS.keys()), index=0)
            preferred_lines = st.multiselect("선호 지하철 노선", sorted(LINE_PREFS), placeholder="노선을 선택하세요")

# 추천 계산
work_df = score_recommendations(base_df, university, work_place, preferred_lines, order, district_lines_map)
lo, hi = rent_band_filter(rent_band)
if rent_band != "상관없음":
    work_df = work_df[(work_df["월세"] >= lo) & (work_df["월세"] <= hi)].reset_index(drop=True)

if work_df.empty:
    st.warning("⚠️ 선택하신 조건에 맞는 지역이 없습니다. 월세 범위를 넓히거나 필터를 조정해 보세요.")
    st.stop()

top5 = work_df.head(5).copy()
top5_names = top5["자치구"].tolist()
rank_color_map = {name: THEME_COLORS[i] for i, name in enumerate(top5_names)}

# 3. 서울시 지도 (데이터 시각화 우선)
st.markdown("### 🗺️ 서울 자치구별 추천 현황")
st.caption("지도를 클릭하거나 마우스를 올리면 상세 정보를 볼 수 있습니다. (빨강색일수록 추천도가 높습니다)")

m_left, m_right = st.columns([2, 1])

with m_left:
    fmap = make_rank_map(work_df, top5_names, rank_color_map)
    st_folium(fmap, width=None, height=500, returned_objects=["last_object_clicked_tooltip"])

with m_right:
    # 핵심 지표 요약 (Metric Cards)
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-kicker'>🥇 추천 1순위</div>
        <div class='metric-value'>{top5.iloc[0]['자치구']}</div>
        <div class='metric-desc'>총점 {top5.iloc[0]['총평점']:.1f}점 / 월세 {int(top5.iloc[0]['월세'])}만 원</div>
    </div>""", unsafe_allow_html=True)
    
    st.markdown(f"""<div class='metric-card' style='margin-top:12px;'>
        <div class='metric-kicker'>🛡️ 안심 거주 추천</div>
        <div class='metric-value'>{work_df.sort_values("안심점수", ascending=False).iloc[0]['자치구']}</div>
        <div class='metric-desc'>밤길 안심 및 주거 정주여건 상위 지역</div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# 4. 추천 지역 TOP 5 카드
st.markdown("### ✨ 조건에 맞는 추천 지역 TOP 5")
cols = st.columns(5)
for i, (_, row) in enumerate(top5.iterrows(), start=1):
    reasons = build_reason(row, seoul_avg_rent, university, work_place, preferred_lines, district_lines_map)
    with cols[i-1]:
        st.markdown(
            f"""
            <div class="top-card {rank_badge(i)}">
                <div class="top-rank">추천 {i}위</div>
                <div class="top-name">{row['자치구']}</div>
                <div class="top-meta">
                    <b>평점</b> {row['총평점']:.1f}<br>
                    <b>월세</b> {int(round(row['월세']))}만<br>
                    <b>노선</b> {row['대표노선']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(chip_html(reasons), unsafe_allow_html=True)

# 하단 탭 메뉴 (상세 분석 및 비교)
st.markdown("### 🔍 더 자세히 알아보기")
tab1, tab2, tab3 = st.tabs(["📍 지역별 심층 분석", "🆚 지역 간 비교", "✅ 나의 체크리스트"])

with tab1:
    # [기존 '지역 상세' 페이지 코드 내용 배치]
    detail_district = st.selectbox("분석할 구를 선택하세요", work_df["자치구"].tolist())
    # ... (상세 지표 표시 코드)

with tab2:
    # [기존 '비교 분석' 페이지 코드 내용 배치]
    compare_choices = st.multiselect("비교할 지역 선택 (최대 3개)", work_df["자치구"].tolist(), default=top5_names[:2])
    # ... (비교 테이블 및 카드 코드)

with tab3:
    # [기존 '체크리스트 저장' 페이지 코드 내용 배치]
    # ... (다운로드 버튼 및 체크리스트 코드)

st.markdown("""<div class="footer-note">※ 데이터 출처: 서울시 열린데이터 광장, 2024-2026 통계 기준</div>""", unsafe_allow_html=True)

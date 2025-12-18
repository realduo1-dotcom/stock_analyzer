import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
import json
import io

# Firestore 라이브러리 임포트 에러 방지를 위한 처리
try:
    from google.cloud import firestore
    from google.oauth2 import service_account
    FIRESTORE_AVAILABLE = True
except ImportError:
    FIRESTORE_AVAILABLE = False

# --- 페이지 설정 ---
st.set_page_config(page_title="ETF 통합 대시보드", layout="wide")

# --- 커스텀 CSS (자바스크립트 스타일 반영) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* 카드 스타일 */
    .st-card {
        background-color: white;
        border: 1px solid #e2e8f0;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1);
        margin-bottom: 24px;
    }
    
    /* 지표 섹션 스타일 */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 800 !important;
        color: #1e293b;
    }
    
    /* 탭 스타일 조정 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        padding: 0px 20px;
        background-color: #f1f5f9;
        border-radius: 8px 8px 0px 0px;
        color: #64748b;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6 !important;
        color: white !important;
    }
    
    /* 헤더 가격 위젯 */
    .price-widget {
        background-color: white;
        padding: 10px 20px;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        text-align: right;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Firebase / Firestore 설정 ---
raw_app_id = st.secrets.get("app_id", "default-app-id")
app_id = raw_app_id if raw_app_id and str(raw_app_id).strip() != "" else "default-app-id"
firebase_config_str = st.secrets.get("firebase_config")

@st.cache_resource
def get_db():
    if not FIRESTORE_AVAILABLE: return None
    try:
        if firebase_config_str:
            config_dict = json.loads(firebase_config_str)
            creds = service_account.Credentials.from_service_account_info(config_dict)
            return firestore.Client(credentials=creds, project=config_dict.get("project_id"))
        return None
    except: return None

db = get_db()

# --- 유틸리티 함수 ---
def clean_price(val):
    if pd.isna(val): return 0.0
    if isinstance(val, (int, float)): return float(val)
    s = str(val).replace(',', '').replace('원', '').replace('%', '').strip()
    try: return float(s)
    except: return 0.0

def find_column(df, keywords):
    for col in df.columns:
        if any(key.lower() in str(col).lower() for key in keywords): return col
    return None

def format_date_korean(date_val):
    try:
        clean_date_str = str(date_val).replace('-', '').replace('.', '').replace('/', '').strip()
        if len(clean_date_str) == 8 and clean_date_str.isdigit():
            return datetime.strptime(clean_date_str, "%Y%m%d").strftime("%Y년 %m월 %d일")
        dt = pd.to_datetime(date_val)
        return dt.strftime("%Y년 %m월 %d일") if not pd.isna(dt) else str(date_val)
    except: return str(date_val)

# --- 클라우드 연동 ---
def save_to_cloud(payload):
    if not db: return
    try:
        doc_ref = db.collection("artifacts").document(app_id).collection("public").document("data").collection("dashboard").document("latest")
        doc_ref.set(payload)
        st.success("☁️ 클라우드 저장 완료!")
    except Exception as e: st.error(f"저장 오류: {e}")

def load_from_cloud():
    if not db: return None
    try:
        doc_ref = db.collection("artifacts").document(app_id).collection("public").document("data").collection("dashboard").document("latest")
        doc = doc_ref.get()
        return doc.to_dict() if doc.exists else None
    except: return None

# --- 샘플 데이터 (초기 로드용) ---
def get_mock_data():
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    price_df = pd.DataFrame({
        'Date': dates, 
        'Price': 50000 + np.cumsum(np.random.normal(50, 200, 100)),
        'Benchmark': 2500 + np.cumsum(np.random.normal(2, 10, 100))
    })
    constituents = pd.DataFrame([
        {'Name': '삼성전자', 'Weight': 25.73, 'w1': 2.09, 'm1': 3.67, 'm3': 46.19, 'm6': 79.13, 'y1': 98.7},
        {'Name': 'SK하이닉스', 'Weight': 16.75, 'w1': 4.24, 'm1': -8.72, 'm3': 84.04, 'm6': 135.42, 'y1': 228.87},
        {'Name': '현대차', 'Weight': 2.07, 'w1': 4.23, 'm1': 9.85, 'm3': 32.51, 'm6': 47.01, 'y1': 41.39}
    ])
    basic_info = {
        "종목명": "KODEX 샘플 ETF", "기초지수": "KOSPI 200", "시가총액": 205530000000, "총보수": 0.45,
        "상장일": "2023-01-01", "운용사": "삼성자산운용", "기초지수개요": "기초지수 개요 내용입니다.", "투자포인트": "투자 포인트 내용입니다."
    }
    return price_df, constituents, basic_info

# --- 메인 앱 ---
def main():
    # 상단 헤더 섹션
    col_h1, col_h2 = st.columns([0.7, 0.3])
    with col_h1:
        st.title("📊 ETF 통합 분석 대시보드")
        st.caption("포트폴리오 성과, 분배금 현황, 구성종목 분석 리포트")
    
    # 1. 초기 데이터 로드
    cloud_data = load_from_cloud()
    price_mock, const_mock, basic_mock = get_mock_data()
    
    def parse_df(json_str, fallback_df):
        if not json_str: return fallback_df
        try: return pd.read_json(io.StringIO(json_str))
        except: return fallback_df

    if cloud_data:
        current_basic = cloud_data.get('basic_info', basic_mock)
        current_price = parse_df(cloud_data.get('price_df'), price_mock)
        current_const = parse_df(cloud_data.get('const_df'), const_mock)
        current_div = parse_df(cloud_data.get('div_df'), None)
        current_issues = parse_df(cloud_data.get('issues_df'), None)
        current_financial = cloud_data.get('financial_data', {})
    else:
        current_basic, current_price, current_const = basic_mock, price_mock, const_mock
        current_div, current_issues, current_financial = None, None, {}

    # 우측 상단 현재가 위젯
    with col_h2:
        if not current_price.empty:
            last_p = current_price.iloc[-1]['Price']
            prev_p = current_price.iloc[-2]['Price'] if len(current_price)>1 else last_p
            diff = last_p - prev_p
            pct = (diff/prev_p*100) if prev_p!=0 else 0
            color = "#ef4444" if diff > 0 else "#3b82f6"
            st.markdown(f"""
                <div class="price-widget">
                    <p style="margin:0; font-size: 0.75rem; color: #64748b; font-weight: 600; text-transform: uppercase;">현재가 (Latest)</p>
                    <p style="margin:0; font-size: 1.5rem; font-weight: 800; color: {color};">
                        {last_p:,.0f}원 <span style="font-size: 0.9rem; font-weight: 400;">({pct:+.2f}%)</span>
                    </p>
                </div>
            """, unsafe_allow_html=True)

    # 2. 관리자 인증 사이드바
    st.sidebar.header("🔒 관리자 인증")
    admin_pw = st.sidebar.text_input("비밀번호", type="password")
    is_admin = admin_pw == "admin1234"

    if is_admin:
        st.sidebar.success("인증 완료")
        st.sidebar.markdown("---")
        st.sidebar.header("📁 데이터 업로드")
        u_basic = st.sidebar.file_uploader("1. 기본정보", type=['xlsx', 'csv'])
        u_price = st.sidebar.file_uploader("2. 주가 데이터", type=['xlsx', 'csv'])
        u_div = st.sidebar.file_uploader("3. 분배금 정보", type=['xlsx', 'csv'])
        u_const = st.sidebar.file_uploader("4. 구성종목/성과", type=['xlsx', 'csv'])
        u_issues = st.sidebar.file_uploader("5. 구성종목 이슈", type=['xlsx', 'csv'])
        u_fin = st.sidebar.file_uploader("6. 구성종목 재무데이터", type=['xlsx'])

        if u_basic:
            df = pd.read_excel(u_basic) if u_basic.name.endswith('xlsx') else pd.read_csv(u_basic)
            if not df.empty:
                row = df.iloc[0]
                current_basic = {
                    "종목명": str(row.iloc[2]) if len(row)>2 else "알수없음",
                    "기초지수": str(row.iloc[3]) if len(row)>3 else "-",
                    "시가총액": clean_price(row.iloc[1]) if len(row)>1 else 0,
                    "총보수": clean_price(row.iloc[4]) if len(row)>4 else 0,
                    "상장일": str(row.iloc[5]) if len(row)>5 else "-",
                    "운용사": str(row.iloc[7]) if len(row)>7 else "-",
                    "기초지수개요": str(row.iloc[8]) if len(row)>8 else "-",
                    "투자포인트": str(row.iloc[9]) if len(row)>9 else "-"
                }
        if u_price: current_price = pd.read_excel(u_price) if u_price.name.endswith('xlsx') else pd.read_csv(u_price)
        if u_div: current_div = pd.read_excel(u_div) if u_div.name.endswith('xlsx') else pd.read_csv(u_div)
        if u_const: current_const = pd.read_excel(u_const) if u_const.name.endswith('xlsx') else pd.read_csv(u_const)
        if u_issues: current_issues = pd.read_excel(u_issues) if u_issues.name.endswith('xlsx') else pd.read_csv(u_issues)
        if u_fin:
            xls = pd.ExcelFile(u_fin)
            current_financial = {sheet: pd.read_excel(xls, sheet_name=sheet).to_dict() for sheet in xls.sheet_names}

        if st.sidebar.button("🚀 클라우드에 영구 저장"):
            payload = {
                "basic_info": current_basic,
                "price_df": current_price.to_json() if isinstance(current_price, pd.DataFrame) else None,
                "const_df": current_const.to_json() if isinstance(current_const, pd.DataFrame) else None,
                "div_df": current_div.to_json() if isinstance(current_div, pd.DataFrame) else None,
                "issues_df": current_issues.to_json() if isinstance(current_issues, pd.DataFrame) else None,
                "financial_data": current_financial,
                "updated_at": datetime.now().isoformat()
            }
            save_to_cloud(payload)
    
    # 3. 탭 구성
    tab_info, tab_perf, tab_div_pie, tab_issues, tab_fin = st.tabs(["ℹ️ 기본 정보", "📈 성과 분석", "💰 분배금/비중", "📰 종목 이슈", "🏢 재무 정보"])

    with tab_info:
        st.markdown(f"""<div class="st-card">
            <h2 style="margin-bottom:20px; font-weight:800; color:#0f172a;">🏢 {current_basic['종목명']}</h2>
            <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px;">
                <div style="border-bottom:1px solid #f1f5f9; padding-bottom:10px;">
                    <p style="color:#64748b; font-size:0.85rem; margin-bottom:4px;">기초지수</p>
                    <p style="font-weight:700; color:#1e293b;">{current_basic['기초지수']}</p>
                </div>
                <div style="border-bottom:1px solid #f1f5f9; padding-bottom:10px;">
                    <p style="color:#64748b; font-size:0.85rem; margin-bottom:4px;">시가총액</p>
                    <p style="font-weight:700; color:#1e293b;">{current_basic['시가총액']/100000000:,.0f} 억원</p>
                </div>
                <div style="border-bottom:1px solid #f1f5f9; padding-bottom:10px;">
                    <p style="color:#64748b; font-size:0.85rem; margin-bottom:4px;">총보수율</p>
                    <p style="font-weight:700; color:#1e293b;">{current_basic['총보수']:.2f}%</p>
                </div>
                <div style="border-bottom:1px solid #f1f5f9; padding-bottom:10px;">
                    <p style="color:#64748b; font-size:0.85rem; margin-bottom:4px;">상장일</p>
                    <p style="font-weight:700; color:#1e293b;">{format_date_korean(current_basic['상장일'])}</p>
                </div>
                <div style="border-bottom:1px solid #f1f5f9; padding-bottom:10px;">
                    <p style="color:#64748b; font-size:0.85rem; margin-bottom:4px;">운용사</p>
                    <p style="font-weight:700; color:#1e293b;">{current_basic['운용사']}</p>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            st.info(f"**💡 기초지수 개요**\n\n{current_basic['기초지수개요']}")
        with c2:
            st.success(f"**🎯 투자 포인트**\n\n{current_basic['투자포인트']}")

    with tab_perf:
        if isinstance(current_price, pd.DataFrame) and not current_price.empty:
            d_col = find_column(current_price, ['일자', '날짜', 'Date'])
            p_col = find_column(current_price, ['Price', '종가'])
            b_col = find_column(current_price, ['Benchmark', '벤치마크'])
            
            if d_col and p_col:
                current_price[d_col] = pd.to_datetime(current_price[d_col])
                current_price = current_price.sort_values(d_col)
                
                # 기간 선택 및 성과 지표
                tr = st.radio("기간", ["1주", "1개월", "3개월", "6개월", "1년", "전체"], index=5, horizontal=True)
                
                last_date = current_price[d_col].max()
                delta = {"1주": 7, "1개월": 30, "3개월": 90, "6개월": 180, "1년": 365}.get(tr, 9999)
                filtered_df = current_price[current_price[d_col] >= (last_date - timedelta(days=delta))].copy()
                
                start_p = filtered_df[p_col].iloc[0]
                end_p = filtered_df[p_col].iloc[-1]
                ret = (end_p - start_p) / start_p * 100
                
                # 차트 데이터 정규화
                filtered_df['ETF_Ret'] = (filtered_df[p_col] - start_p) / start_p * 100
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=filtered_df[d_col], y=filtered_df['ETF_Ret'], name='ETF 수익률', line=dict(color='#ef4444', width=3)))
                
                if b_col:
                    start_b = filtered_df[b_col].iloc[0]
                    filtered_df['BM_Ret'] = (filtered_df[b_col] - start_b) / start_b * 100
                    fig.add_trace(go.Scatter(x=filtered_df[d_col], y=filtered_df['BM_Ret'], name='벤치마크', line=dict(color='#94a3b8', width=2, dash='dot')))
                
                fig.update_layout(template="plotly_white", hovermode="x unified", height=500, yaxis_title="수익률 (%)",
                                 legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 기간별 성과 테이블 (TOP 10)
                st.markdown("#### 📊 구성종목 기간 성과 (Top 10)")
                if isinstance(current_const, pd.DataFrame):
                    st.dataframe(current_const.head(10), use_container_width=True)

    with tab_div_pie:
        c_div, c_pie = st.columns(2)
        with c_div:
            st.subheader("💰 분배금 현황")
            if isinstance(current_div, pd.DataFrame) and not current_div.empty:
                fig_div = px.bar(current_div, x=current_div.columns[0], y=current_div.columns[1], text_auto=',.0f', color_discrete_sequence=['#3b82f6'])
                fig_div.update_layout(template="plotly_white", height=400, plot_bgcolor='rgba(242, 242, 242, 0.6)')
                st.plotly_chart(fig_div, use_container_width=True)
            else: st.info("분배금 데이터가 없습니다.")
            
        with c_pie:
            st.subheader("🍕 상위 10개 구성종목 비중")
            if isinstance(current_const, pd.DataFrame) and not current_const.empty:
                fig_pie = px.pie(current_const.head(10), names=current_const.columns[0], values=current_const.columns[1], hole=0.4,
                                color_discrete_sequence=px.colors.qualitative.T10)
                fig_pie.update_layout(template="plotly_white", height=400)
                st.plotly_chart(fig_pie, use_container_width=True)
            else: st.info("구성종목 데이터가 없습니다.")

    with tab_issues:
        if isinstance(current_issues, pd.DataFrame) and not current_issues.empty:
            stocks = current_issues[current_issues.columns[1]].unique()
            sel_issue_stock = st.selectbox("이슈 확인할 종목", stocks)
            filtered_is = current_issues[current_issues[current_issues.columns[1]] == sel_issue_stock]
            for _, row in filtered_is.iterrows():
                with st.expander(f"[{row[0]}] {row[1]}"):
                    st.write(row[2])
        else: st.info("등록된 이슈가 없습니다.")

    with tab_fin:
        if current_financial:
            stock = st.selectbox("재무정보 종목 선택", list(current_financial.keys()))
            df_fin = pd.DataFrame(current_financial[stock])
            
            # 자바스크립트 코드의 연간/분기별 분리 로직 반영
            st.markdown(f"### 🏢 {stock} 재무제표")
            view_mode = st.radio("보기 모드", ["연간", "분기"], horizontal=True)
            
            # 첫 4개 데이터 열을 연간, 나머지를 분기로 간주하는 로직
            cols = df_fin.columns.tolist()
            label_col = cols[0]
            data_cols = cols[1:]
            
            if view_mode == "연간":
                display_cols = [label_col] + data_cols[:4]
            else:
                display_cols = [label_col] + data_cols[4:]
                
            st.table(df_fin[display_cols])
        else: st.info("재무 데이터가 없습니다.")

if __name__ == "__main__":
    main()

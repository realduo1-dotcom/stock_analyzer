import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
import json
import io

# Firestore 라이브러리 임포트
try:
    from google.cloud import firestore
    from google.oauth2 import service_account
    FIRESTORE_AVAILABLE = True
except ImportError:
    FIRESTORE_AVAILABLE = False

# --- 페이지 설정 ---
st.set_page_config(page_title="ETF 통합 대시보드", layout="wide")

# --- 커스텀 CSS (React/Tailwind 스타일 완벽 재현) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, sans-serif;
        background-color: #f8fafc;
    }
    
    /* 카드 레이아웃 */
    .st-card {
        background-color: white;
        border: 1px solid #e2e8f0;
        padding: 1.5rem;
        border-radius: 0.75rem;
        box-shadow: 0 1px 2px 0 rgb(0 0 0 / 0.05);
        margin-bottom: 1.5rem;
    }
    
    /* 메트릭 가독성 */
    [data-testid="stMetricValue"] {
        font-size: 1.75rem !important;
        font-weight: 800 !important;
        color: #0f172a;
    }
    
    /* 탭 커스텀 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        padding: 0 24px;
        background-color: #f1f5f9;
        border-radius: 8px 8px 0 0;
        color: #64748b;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563eb !important;
        color: white !important;
    }
    
    /* 헤더 현재가 위젯 */
    .price-widget {
        background: white;
        padding: 12px 20px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        text-align: right;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# --- Firebase / Firestore 설정 (Secrets 대응) ---
def get_clean_app_id():
    # Secrets에서 값을 가져오되, 없거나 비어있으면 기본값 반환
    try:
        val = st.secrets.get("app_id")
        if val:
            # 문자열로 변환 후 양끝 공백 제거 및 내부 슬래시 제거
            s_val = str(val).strip().replace("/", "")
            if s_val:
                return s_val
    except:
        pass
    return "stock_analyzer"

app_id = get_clean_app_id()
firebase_config_raw = st.secrets.get("firebase_config")

@st.cache_resource
def get_db():
    if not FIRESTORE_AVAILABLE or not firebase_config_raw:
        return None
    try:
        if isinstance(firebase_config_raw, str):
            config_dict = json.loads(firebase_config_raw)
        else:
            config_dict = dict(firebase_config_raw)
        
        # private_key 내의 \n 문자 처리
        if 'private_key' in config_dict:
            config_dict['private_key'] = config_dict['private_key'].replace('\\n', '\n')
            
        creds = service_account.Credentials.from_service_account_info(config_dict)
        return firestore.Client(credentials=creds, project=config_dict.get("project_id"))
    except Exception as e:
        st.sidebar.error(f"DB 초기화 실패: {e}")
        return None

db = get_db()

# --- 유틸리티 ---
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
        d_str = str(date_val).replace('-', '').replace('.', '').replace('/', '').strip()
        if len(d_str) >= 8 and d_str[:8].isdigit():
            return datetime.strptime(d_str[:8], "%Y%m%d").strftime("%Y년 %m월 %d일")
        dt = pd.to_datetime(date_val)
        return dt.strftime("%Y년 %m월 %d일") if not pd.isna(dt) else str(date_val)
    except: return str(date_val)

# --- 클라우드 저장 및 로드 (경로 에러 해결을 위한 구조 변경) ---
def save_to_cloud(payload):
    if not db: 
        st.error("데이터베이스 연결 설정이 필요합니다.")
        return
    
    # App ID 재검증
    safe_app_id = str(app_id).strip()
    if not safe_app_id: safe_app_id = "stock_analyzer"
    
    try:
        # 경로를 명시적 문자열로 구성하되, 슬래시 중복 방지 및 컴포넌트 방식 혼합
        # db.collection().document() 방식을 사용하여 'One or more components is empty' 에러 원천 차단
        doc_ref = db.collection("artifacts").document(safe_app_id)\
                    .collection("public").document("data")\
                    .collection("dashboard").document("latest")
        
        doc_ref.set(payload)
        st.success(f"☁️ 클라우드 저장 완료! (ID: {safe_app_id})")
    except Exception as e:
        st.error(f"저장 실패. (ID: {safe_app_id})\n에러 상세: {e}")

def load_from_cloud():
    if not db: return None
    try:
        safe_app_id = str(app_id).strip()
        if not safe_app_id: safe_app_id = "stock_analyzer"
        
        # 저장할 때와 동일한 안전한 컴포넌트 방식 사용
        doc_ref = db.collection("artifacts").document(safe_app_id)\
                    .collection("public").document("data")\
                    .collection("dashboard").document("latest")
        
        doc = doc_ref.get()
        return doc.to_dict() if doc.exists else None
    except: return None

# --- 기본 샘플 데이터 ---
def get_mock_data():
    dates = pd.date_range(end=datetime.now(), periods=50, freq='D')
    price_df = pd.DataFrame({
        'Date': dates, 'Price': 50000 + np.cumsum(np.random.normal(10, 100, 50)),
        'Benchmark': 2500 + np.cumsum(np.random.normal(1, 5, 50))
    })
    constituents = pd.DataFrame([
        {'Name': '삼성전자', 'Weight': 25.73, 'w1': 2.0, 'm1': 3.5, 'y1': 15.0},
        {'Name': 'SK하이닉스', 'Weight': 16.75, 'w1': 4.1, 'm1': -2.5, 'y1': 45.0}
    ])
    basic_info = {
        "종목명": "KODEX 건설", "기초지수": "KRX 건설", "시가총액": 34572000000, "총보수": 0.45,
        "상장일": "20250428", "운용사": "삼성자산운용(ETF)", "기초지수개요": "기초지수 개요입니다.", "투자포인트": "투자 포인트입니다."
    }
    return price_df, constituents, basic_info

# --- 메인 앱 뷰 ---
def main():
    # 1. 상단 헤더
    h_col1, h_col2 = st.columns([0.7, 0.3])
    
    # 데이터 로드
    cloud_data = load_from_cloud()
    p_mock, c_mock, b_mock = get_mock_data()
    
    def parse_df(json_str, fallback):
        if not json_str: return fallback
        try: return pd.read_json(io.StringIO(json_str))
        except: return fallback

    if cloud_data:
        cur_basic = cloud_data.get('basic_info', b_mock)
        cur_price = parse_df(cloud_data.get('price_df'), p_mock)
        cur_const = parse_df(cloud_data.get('const_df'), c_mock)
        cur_div = parse_df(cloud_data.get('div_df'), None)
        cur_issues = parse_df(cloud_data.get('issues_df'), None)
        cur_financial = cloud_data.get('financial_data', {})
    else:
        cur_basic, cur_price, cur_const = b_mock, p_mock, c_mock
        cur_div, cur_issues, cur_financial = None, None, {}

    with h_col1:
        st.title("📊 ETF 통합 분석 대시보드")
        st.caption("포트폴리오 성과 및 구성종목 심층 분석 리포트")
        
    with h_col2:
        if isinstance(cur_price, pd.DataFrame) and not cur_price.empty:
            p_col = find_column(cur_price, ['Price', '종가'])
            if p_col:
                last_val = cur_price.iloc[-1][p_col]
                prev_val = cur_price.iloc[-2][p_col] if len(cur_price)>1 else last_val
                change = last_val - prev_val
                pct = (change/prev_val*100) if prev_val != 0 else 0
                color = "#ef4444" if change >= 0 else "#3b82f6"
                st.markdown(f"""
                    <div class="price-widget">
                        <div style="font-size: 0.75rem; color: #64748b; font-weight: 700;">현재가 (Latest)</div>
                        <div style="font-size: 1.6rem; font-weight: 800; color: {color};">
                            {last_val:,.0f}원 <span style="font-size: 0.9rem; font-weight: 400;">({pct:+.2f}%)</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

    # 2. 관리자 인증 사이드바
    st.sidebar.header("🔒 관리자 인증")
    pw = st.sidebar.text_input("비밀번호", type="password")
    is_admin = pw == "admin1234"

    if is_admin:
        st.sidebar.success("인증됨")
        st.sidebar.markdown("---")
        u_basic = st.sidebar.file_uploader("1. 기본정보", type=['xlsx', 'csv'])
        u_price = st.sidebar.file_uploader("2. 주가 데이터", type=['xlsx', 'csv'])
        u_div = st.sidebar.file_uploader("3. 분배금 정보", type=['xlsx', 'csv'])
        u_const = st.sidebar.file_uploader("4. 구성종목/성과", type=['xlsx', 'csv'])
        u_issues = st.sidebar.file_uploader("5. 이슈 데이터", type=['xlsx', 'csv'])
        u_fin = st.sidebar.file_uploader("6. 재무데이터", type=['xlsx'])

        if u_basic:
            df = pd.read_excel(u_basic) if u_basic.name.endswith('xlsx') else pd.read_csv(u_basic)
            if not df.empty:
                row = df.iloc[0]
                cur_basic = {
                    "종목명": str(row.iloc[2]) if len(row)>2 else "알수없음",
                    "기초지수": str(row.iloc[3]) if len(row)>3 else "-",
                    "시가총액": clean_price(row.iloc[1]) if len(row)>1 else 0,
                    "총보수": clean_price(row.iloc[4]) if len(row)>4 else 0,
                    "상장일": str(row.iloc[5]) if len(row)>5 else "-",
                    "운용사": str(row.iloc[7]) if len(row)>7 else "-",
                    "기초지수개요": str(row.iloc[8]) if len(row)>8 else "-",
                    "투자포인트": str(row.iloc[9]) if len(row)>9 else "-"
                }
        if u_price: cur_price = pd.read_excel(u_price) if u_price.name.endswith('xlsx') else pd.read_csv(u_price)
        if u_div: cur_div = pd.read_excel(u_div) if u_div.name.endswith('xlsx') else pd.read_csv(u_div)
        if u_const: cur_const = pd.read_excel(u_const) if u_const.name.endswith('xlsx') else pd.read_csv(u_const)
        if u_issues: cur_issues = pd.read_excel(u_issues) if u_issues.name.endswith('xlsx') else pd.read_csv(u_issues)
        if u_fin:
            xls = pd.ExcelFile(u_fin)
            cur_financial = {sh: pd.read_excel(xls, sheet_name=sh).to_dict() for sh in xls.sheet_names}

        if st.sidebar.button("🚀 클라우드에 영구 저장"):
            save_to_cloud({
                "basic_info": cur_basic,
                "price_df": cur_price.to_json() if isinstance(cur_price, pd.DataFrame) else None,
                "const_df": cur_const.to_json() if isinstance(cur_const, pd.DataFrame) else None,
                "div_df": cur_div.to_json() if isinstance(cur_div, pd.DataFrame) else None,
                "issues_df": cur_issues.to_json() if isinstance(cur_issues, pd.DataFrame) else None,
                "financial_data": cur_financial,
                "updated_at": datetime.now().isoformat()
            })

    # 3. 탭 구성
    tabs = st.tabs(["ℹ️ 기본 정보", "📈 성과 분석", "💰 분배금/비중", "📰 종목 이슈", "🏢 재무 정보"])

    with tabs[0]:
        st.markdown(f"""<div class="st-card">
            <h2 style="margin-bottom:24px; color:#0f172a; font-weight:800;">🏢 {cur_basic['종목명']}</h2>
            <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 24px;">
                <div><div style="color:#64748b; font-size:0.8rem; font-weight:600; margin-bottom:4px;">기초지수</div><div style="font-weight:700;">{cur_basic['기초지수']}</div></div>
                <div><div style="color:#64748b; font-size:0.8rem; font-weight:600; margin-bottom:4px;">시가총액</div><div style="font-weight:700;">{cur_basic['시가총액']/100000000:,.0f} 억원</div></div>
                <div><div style="color:#64748b; font-size:0.8rem; font-weight:600; margin-bottom:4px;">총보수율</div><div style="font-weight:700;">{cur_basic['총보수']:.2f}%</div></div>
                <div><div style="color:#64748b; font-size:0.8rem; font-weight:600; margin-bottom:4px;">상장일</div><div style="font-weight:700;">{format_date_korean(cur_basic['상장일'])}</div></div>
                <div><div style="color:#64748b; font-size:0.8rem; font-weight:600; margin-bottom:4px;">운용사</div><div style="font-weight:700;">{cur_basic['운용사']}</div></div>
            </div>
        </div>""", unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1: st.info(f"**💡 기초지수 개요**\n\n{cur_basic['기초지수개요']}")
        with c2: st.success(f"**🎯 투자 포인트**\n\n{cur_basic['투자포인트']}")

    with tabs[1]:
        if isinstance(cur_price, pd.DataFrame) and not cur_price.empty:
            d_col = find_column(cur_price, ['Date', '일자', '날짜'])
            p_col = find_column(cur_price, ['Price', '종가'])
            b_col = find_column(cur_price, ['Benchmark', '벤치마크'])
            
            if d_col and p_col:
                cur_price[d_col] = pd.to_datetime(cur_price[d_col])
                cur_price = cur_price.sort_values(d_col)
                tr = st.radio("기간 선택", ["1주", "1개월", "3개월", "6개월", "1년", "전체"], index=5, horizontal=True)
                
                delta = {"1주": 7, "1개월": 30, "3개월": 90, "6개월": 180, "1년": 365}.get(tr, 9999)
                f_df = cur_price[cur_price[d_col] >= (cur_price[d_col].max() - timedelta(days=delta))].copy()
                
                if not f_df.empty:
                    start_p = f_df[p_col].iloc[0]
                    f_df['ETF_Ret'] = (f_df[p_col] - start_p) / start_p * 100
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=f_df[d_col], y=f_df['ETF_Ret'], name='ETF 수익률', line=dict(color='#ef4444', width=3)))
                    if b_col:
                        start_b = f_df[b_col].iloc[0]
                        f_df['BM_Ret'] = (f_df[b_col] - start_b) / start_b * 100
                        fig.add_trace(go.Scatter(x=f_df[d_col], y=f_df['BM_Ret'], name='벤치마크', line=dict(color='#94a3b8', width=2, dash='dot')))
                    
                    fig.update_layout(template="plotly_white", hovermode="x unified", height=500, margin=dict(t=10))
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("#### 📊 구성종목 기간 성과 (Top 10)")
                if isinstance(cur_const, pd.DataFrame): st.dataframe(cur_const.head(10), use_container_width=True)

    with tabs[2]:
        c_bar, c_pie = st.columns(2)
        with c_bar:
            st.subheader("💰 분배금 현황")
            if isinstance(cur_div, pd.DataFrame) and not cur_div.empty:
                st.plotly_chart(px.bar(cur_div, x=cur_div.columns[0], y=cur_div.columns[1], text_auto=',.0f', color_discrete_sequence=['#3b82f6']), use_container_width=True)
            else: st.info("데이터 없음")
        with c_pie:
            st.subheader("🍕 상위 10개 구성종목 비중")
            if isinstance(cur_const, pd.DataFrame) and not cur_const.empty:
                st.plotly_chart(px.pie(cur_const.head(10), names=cur_const.columns[0], values=cur_const.columns[1], hole=0.4, color_discrete_sequence=px.colors.qualitative.T10), use_container_width=True)

    with tabs[3]:
        if isinstance(cur_issues, pd.DataFrame) and not cur_issues.empty:
            stocks = cur_issues[cur_issues.columns[1]].unique()
            sel = st.selectbox("종목 선택", stocks, key="issue_stock_sel")
            f_is = cur_issues[cur_issues[cur_issues.columns[1]] == sel]
            for _, row in f_is.iterrows():
                with st.expander(f"[{row[0]}] {row[1]}"): st.write(row[2])
        else: st.info("데이터 없음")

    with tabs[4]:
        if cur_financial:
            stock = st.selectbox("종목 선택", list(cur_financial.keys()), key="fin_stock_sel")
            df_fin = pd.DataFrame(cur_financial[stock])
            vm = st.radio("보기 모드", ["연간", "분기"], horizontal=True, key="fin_view_mode")
            cols = df_fin.columns.tolist()
            if vm == "연간": st.table(df_fin[[cols[0]] + cols[1:5]])
            else: st.table(df_fin[[cols[0]] + cols[5:]])
        else: st.info("데이터 없음")

if __name__ == "__main__":
    main()

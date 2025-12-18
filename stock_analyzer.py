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

# --- Firebase / Firestore 설정 ---
raw_app_id = st.secrets.get("app_id", "default-app-id")
app_id = raw_app_id if raw_app_id and str(raw_app_id).strip() != "" else "default-app-id"
firebase_config_str = st.secrets.get("firebase_config")

@st.cache_resource
def get_db():
    """Firestore 클라이언트 초기화 및 캐싱"""
    if not FIRESTORE_AVAILABLE:
        return None
    try:
        if firebase_config_str:
            config_dict = json.loads(firebase_config_str)
            creds = service_account.Credentials.from_service_account_info(config_dict)
            client = firestore.Client(credentials=creds, project=config_dict.get("project_id"))
            return client
        return None
    except Exception as e:
        st.sidebar.error(f"DB 초기화 실패: {e}")
        return None

db = get_db()

# --- 안내 메시지 ---
if not FIRESTORE_AVAILABLE:
    st.error("⚠️ 'google-cloud-firestore' 라이브러리 설치가 필요합니다.")

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

# --- Firestore 데이터 연동 함수 (RULE 1 경로 준수) ---
def save_to_cloud(payload):
    if not db:
        st.error("DB가 연결되지 않았습니다. 설정을 확인해주세요.")
        return
    try:
        doc_ref = db.collection("artifacts").document(app_id).collection("public").document("data").collection("dashboard").document("latest")
        doc_ref.set(payload)
        st.success("☁️ 클라우드 저장 완료!")
    except Exception as e:
        err_msg = str(e)
        if "404" in err_msg and "database" in err_msg.lower():
            st.error("❌ Firestore 데이터베이스 인스턴스가 생성되지 않았습니다.")
            st.info("Firebase Console > Build > Firestore Database 메뉴에서 '데이터베이스 만들기'를 완료해주세요.")
        elif "403" in err_msg:
            st.error("❌ 권한 오류 또는 API 미활성화 상태입니다.")
        else:
            st.error(f"저장 오류: {e}")

def load_from_cloud():
    if not db: return None
    try:
        doc_ref = db.collection("artifacts").document(app_id).collection("public").document("data").collection("dashboard").document("latest")
        doc = doc_ref.get()
        return doc.to_dict() if doc.exists else None
    except:
        return None

# --- 샘플 데이터 ---
def get_mock_data():
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    price_df = pd.DataFrame({
        'Date': dates, 
        'Price': 50000 + np.cumsum(np.random.normal(50, 200, 100)),
        'Benchmark': 2500 + np.cumsum(np.random.normal(2, 10, 100))
    })
    constituents = pd.DataFrame([{'Name': '삼성전자', 'Weight': 30.0, '1Y': 15.5}, {'Name': 'SK하이닉스', 'Weight': 20.0, '1Y': 25.0}])
    basic_info = {
        "종목명": "데이터를 업로드해주세요", "기초지수": "-", "시가총액": 0, "총보수": 0.0,
        "상장일": "2025-01-01", "운용사": "-", "기초지수개요": "현재 저장된 데이터가 없습니다.", "투자포인트": "관리자 로그인이 필요합니다."
    }
    return price_df, constituents, basic_info

# --- 메인 앱 ---
def main():
    st.title("📊 ETF 통합 분석 대시보드")
    
    # 1. 초기 데이터 로드
    cloud_data = load_from_cloud()
    price_mock, const_mock, basic_mock = get_mock_data()
    
    # 데이터 파싱 함수 (ValueError 방지)
    def parse_df(json_str, fallback_df):
        if not json_str or not isinstance(json_str, str):
            return fallback_df
        try:
            return pd.read_json(io.StringIO(json_str))
        except:
            return fallback_df

    if cloud_data:
        st.sidebar.info("📡 클라우드 동기화 완료")
        current_basic = cloud_data.get('basic_info', basic_mock)
        current_price = parse_df(cloud_data.get('price_df'), price_mock)
        current_const = parse_df(cloud_data.get('const_df'), const_mock)
        current_div = parse_df(cloud_data.get('div_df'), None)
        current_issues = parse_df(cloud_data.get('issues_df'), None)
        current_financial = cloud_data.get('financial_data', {})
    else:
        current_basic, current_price, current_const = basic_mock, price_mock, const_mock
        current_div, current_issues, current_financial = None, None, {}

    # 2. 관리자 인증
    st.sidebar.header("🔒 관리자 인증")
    admin_pw = st.sidebar.text_input("비밀번호", type="password")
    
    if admin_pw == "admin1234":
        st.sidebar.success("관리자 모드 활성화")
        st.sidebar.markdown("---")
        u_basic = st.sidebar.file_uploader("1. 기본정보", type=['xlsx', 'csv'])
        u_price = st.sidebar.file_uploader("2. 주가 데이터", type=['xlsx', 'csv'])
        u_div = st.sidebar.file_uploader("3. 분배금 정보", type=['xlsx', 'csv'])
        u_const = st.sidebar.file_uploader("4. 구성종목", type=['xlsx', 'csv'])
        u_issues = st.sidebar.file_uploader("5. 이슈 데이터", type=['xlsx', 'csv'])
        u_fin = st.sidebar.file_uploader("6. 재무데이터", type=['xlsx'])

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
    
    # 3. 화면 표시
    tabs = st.tabs(["ℹ️ 기본 정보", "📈 성과 분석", "💰 분배금/비중", "📰 종목 이슈", "🏢 재무 정보"])

    with tabs[0]:
        st.header(f"🏢 {current_basic['종목명']}")
        st.markdown("---")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("기초지수", current_basic["기초지수"])
        c2.metric("시가총액", f"{current_basic['시가총액']/100000000:,.0f} 억원")
        c3.metric("총보수(연)", f"{current_basic['총보수']:.3f}%")
        c4.metric("상장일", format_date_korean(current_basic["상장일"]))
        c5.metric("운용사", current_basic["운용사"])
        st.markdown("---")
        cd1, cd2 = st.columns(2)
        with cd1: st.info(f"💡 **기초지수 개요**\n\n{current_basic['기초지수개요']}")
        with cd2: st.success(f"🎯 **투자 포인트**\n\n{current_basic['투자포인트']}")

    with tabs[1]:
        if isinstance(current_price, pd.DataFrame) and not current_price.empty:
            d_col = find_column(current_price, ['일자', '날짜', 'Date'])
            p_col = find_column(current_price, ['Price', '종가'])
            if d_col and p_col:
                current_price[d_col] = pd.to_datetime(current_price[d_col])
                current_price = current_price.sort_values(d_col)
                time_range = st.radio("기간", ["1주", "1개월", "3개월", "6개월", "1년", "전체"], index=5, horizontal=True)
                
                # 기간 필터링
                last_date = current_price[d_col].max()
                if time_range == "1주": start_date = last_date - timedelta(weeks=1)
                elif time_range == "1개월": start_date = last_date - timedelta(days=30)
                elif time_range == "3개월": start_date = last_date - timedelta(days=90)
                elif time_range == "6개월": start_date = last_date - timedelta(days=180)
                elif time_range == "1년": start_date = last_date - timedelta(days=365)
                else: start_date = current_price[d_col].min()
                
                filtered_df = current_price[current_price[d_col] >= start_date].copy()
                
                if not filtered_df.empty:
                    fig = px.line(filtered_df, x=d_col, y=p_col, title=f"주가 추이 ({time_range})")
                    fig.update_xaxes(tickformat="%Y년 %m월")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("선택한 기간에 데이터가 없습니다.")
            else:
                st.warning("데이터 컬럼 형식이 맞지 않습니다.")

    with tabs[2]:
        c_bar, c_pie = st.columns(2)
        with c_bar:
            st.subheader("분배금")
            if current_div is not None and not current_div.empty: 
                st.bar_chart(current_div)
            else:
                st.info("표시할 분배금 데이터가 없습니다.")
        with c_pie:
            st.subheader("구성종목")
            if isinstance(current_const, pd.DataFrame) and not current_const.empty:
                st.plotly_chart(px.pie(current_const.head(10), names=current_const.columns[0], values=current_const.columns[1], hole=0.4), use_container_width=True)
            else:
                st.info("표시할 구성종목 데이터가 없습니다.")

    with tabs[3]:
        if current_issues is not None and not current_issues.empty: 
            st.table(current_issues)
        else: 
            st.info("등록된 이슈 데이터가 없습니다.")

    with tabs[4]:
        if current_financial:
            stock = st.selectbox("종목 선택", list(current_financial.keys()))
            st.table(pd.DataFrame(current_financial[stock]))
        else: 
            st.info("재무 데이터가 없습니다.")

if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
import io

# --- 페이지 설정 ---
st.set_page_config(page_title="ETF 통합 대시보드", layout="wide")

# --- 커스텀 CSS (지표 카드 스타일) ---
st.markdown("""
    <style>
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 유틸리티 함수 ---
def clean_price(val):
    if pd.isna(val): return 0.0
    if isinstance(val, (int, float)): return float(val)
    s = str(val).replace(',', '').replace('원', '').replace('%', '').strip()
    try:
        return float(s)
    except:
        return 0.0

def find_column(df, keywords):
    """주어진 키워드 중 하나를 포함하는 컬럼명을 찾음"""
    for col in df.columns:
        if any(key.lower() in str(col).lower() for key in keywords):
            return col
    return None

def format_date_korean(date_val):
    """날짜를 'YYYY년 MM월 DD일' 형식으로 변환"""
    try:
        clean_date_str = str(date_val).replace('-', '').replace('.', '').replace('/', '').strip()
        if len(clean_date_str) == 8 and clean_date_str.isdigit():
            dt = datetime.strptime(clean_date_str, "%Y%m%d")
            return dt.strftime("%Y년 %m월 %d일")
        
        dt = pd.to_datetime(date_val)
        if not pd.isna(dt):
            return dt.strftime("%Y년 %m월 %d일")
        return str(date_val)
    except:
        return str(date_val)

# --- 샘플 데이터 (초기 로드용) ---
def get_mock_data():
    dates = pd.date_range(end=datetime.now(), periods=200, freq='D')
    prices = 50000 + np.cumsum(np.random.normal(50, 200, 200))
    benchmarks = 2500 + np.cumsum(np.random.normal(2, 10, 200))
    
    price_df = pd.DataFrame({
        'Date': dates,
        'Price': prices,
        'Benchmark': benchmarks
    })
    
    constituents = pd.DataFrame([
        {'Name': '삼성전자', 'Weight': 25.73, '1Y': 98.7},
        {'Name': 'SK하이닉스', 'Weight': 16.75, '1Y': 228.87},
        {'Name': '현대차', 'Weight': 2.07, '1Y': 41.39},
        {'Name': 'KB금융', 'Weight': 2.05, '1Y': 46.14},
        {'Name': 'NAVER', 'Weight': 1.69, '1Y': 11.7},
    ])

    basic_info = {
        "종목명": "KODEX 샘플 ETF",
        "기초지수": "KOSPI 200 지수",
        "시가총액": 205530000000,
        "총보수": 0.45,
        "상장일": "2023-01-01",
        "운용사": "삼성자산운용",
        "기초지수개요": "대한민국 상장 주식 중 시장 대표성 및 유동성을 고려하여 선정된 200개 종목으로 구성된 지수입니다.",
        "투자포인트": "1. 대한민국 대표 기업에 분산 투자\n2. 낮은 보수로 시장 수익률 추구\n3. 높은 거래량과 유동성 확보"
    }

    return price_df, constituents, basic_info

# --- 메인 앱 로직 ---
def main():
    st.title("📊 ETF 통합 분석 대시보드")
    st.caption("포트폴리오 성과, 분배금 현황, 구성종목 분석 리포트")

    # --- 사이드바: 파일 업로드 섹션 ---
    st.sidebar.header("📁 데이터 업로드")
    
    upload_basic = st.sidebar.file_uploader("1. 기본정보 (Excel/CSV)", type=['xlsx', 'csv'])
    upload_price = st.sidebar.file_uploader("2. 주가 데이터 (Excel/CSV)", type=['xlsx', 'csv'])
    upload_div = st.sidebar.file_uploader("3. 분배금 정보 (Excel/CSV)", type=['xlsx', 'csv'])
    upload_const = st.sidebar.file_uploader("4. 구성종목/성과 (Excel/CSV)", type=['xlsx', 'csv'])
    upload_issues = st.sidebar.file_uploader("5. 구성종목 이슈 (Excel/CSV)", type=['xlsx', 'csv'])
    upload_financial = st.sidebar.file_uploader("6. 구성종목 재무데이터 (Excel)", type=['xlsx'])

    # --- 데이터 로드 ---
    price_mock, const_mock, basic_mock = get_mock_data()

    # 1. 기본 정보 처리
    if upload_basic:
        try:
            df_basic_raw = pd.read_excel(upload_basic) if upload_basic.name.endswith('xlsx') else pd.read_csv(upload_basic)
            if not df_basic_raw.empty:
                row = df_basic_raw.iloc[0]
                
                def get_val_refined(df, row, keywords, col_idx, default):
                    col = find_column(df, keywords)
                    if col is not None:
                        val = row[col]
                        if not (pd.isna(val) or str(val).strip() in ['', '0', '0.0']):
                            return val
                    if len(row) > col_idx:
                        val = row.iloc[col_idx]
                        if not (pd.isna(val) or str(val).strip() in ['', '0', '0.0']):
                            return val
                    return default

                basic_info = {
                    "종목명": get_val_refined(df_basic_raw, row, ['종목명', '이름', 'Name'], 2, basic_mock["종목명"]),
                    "기초지수": get_val_refined(df_basic_raw, row, ['기초지수', 'Index'], 3, basic_mock["기초지수"]),
                    "시가총액": clean_price(get_val_refined(df_basic_raw, row, ['시가총액', 'Market Cap'], 1, 0)),
                    "총보수": clean_price(get_val_refined(df_basic_raw, row, ['보수', 'Fee'], 4, 0)),
                    "상장일": str(get_val_refined(df_basic_raw, row, ['상장일', 'Listing'], 5, basic_mock["상장일"])),
                    "운용사": get_val_refined(df_basic_raw, row, ['운용사', 'Manager'], 7, basic_mock["운용사"]),
                    "기초지수개요": get_val_refined(df_basic_raw, row, ['개요', 'Desc'], 8, basic_mock["기초지수개요"]),
                    "투자포인트": get_val_refined(df_basic_raw, row, ['포인트', 'Point'], 9, basic_mock["투자포인트"])
                }
            else:
                basic_info = basic_mock
        except Exception:
            basic_info = basic_mock
    else:
        basic_info = basic_mock

    # 2. 주가 데이터 처리
    if upload_price:
        df_price = pd.read_excel(upload_price) if upload_price.name.endswith('xlsx') else pd.read_csv(upload_price)
        date_col = find_column(df_price, ['일자', '날짜', 'Date', 'date'])
        price_col = find_column(df_price, ['Price', '종가', 'Close'])
        bench_col = find_column(df_price, ['Benchmark', '벤치마크', 'Index'])
        
        cols = df_price.columns
        if not date_col and len(cols) >= 1: date_col = cols[0]
        if not price_col and len(cols) >= 2: price_col = cols[1]
        if not bench_col and len(cols) >= 3: bench_col = cols[2]
        
        if date_col: df_price = df_price.rename(columns={date_col: 'Date'})
        if price_col: df_price = df_price.rename(columns={price_col: 'Price'})
        if bench_col: df_price = df_price.rename(columns={bench_col: 'Benchmark'})
    else:
        df_price = price_mock

    if 'Date' in df_price.columns:
        df_price['Date'] = pd.to_datetime(df_price['Date'])
        df_price = df_price.sort_values('Date')

    # 3. 구성종목 데이터 처리
    if upload_const:
        df_const = pd.read_excel(upload_const) if upload_const.name.endswith('xlsx') else pd.read_csv(upload_const)
        name_col = find_column(df_const, ['종목', 'Name'])
        weight_col = find_column(df_const, ['비중', 'Weight'])
        if name_col: df_const = df_const.rename(columns={name_col: 'Name'})
        if weight_col: df_const = df_const.rename(columns={weight_col: 'Weight'})
    else:
        df_const = const_mock

    # 데이터 정제
    if not df_price.empty and 'Price' in df_price.columns:
        df_price['Price'] = df_price['Price'].apply(clean_price)

    # --- 탭 구성 ---
    tab0, tab1, tab2, tab3, tab4 = st.tabs(["ℹ️ 기본 정보", "📈 성과 분석", "💰 분배금/비중", "📰 종목 이슈", "🏢 재무 정보"])

    with tab0:
        st.header(f"🏢 {basic_info['종목명']}")
        st.markdown("---")
        
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("기초지수", basic_info["기초지수"])
        c2.metric("시가총액", f"{basic_info['시가총액']/100000000:,.0f} 억원")
        c3.metric("총보수(연)", f"{basic_info['총보수']:.3f}%")
        
        formatted_listing_date = format_date_korean(basic_info["상장일"])
        c4.metric("상장일", formatted_listing_date)
        c5.metric("운용사", basic_info["운용사"])

        st.markdown("---")

        col_desc, col_points = st.columns(2)
        with col_desc:
            st.info("💡 **기초지수 개요**")
            st.write(basic_info["기초지수개요"])

        with col_points:
            st.success("🎯 **투자 포인트**")
            points = basic_info["투자포인트"]
            if isinstance(points, str):
                for p in points.split('\n'):
                    if p.strip(): st.write(f"{p.strip()}")
            else:
                st.write(points)

    with tab1:
        if not df_price.empty and 'Date' in df_price.columns:
            # 1. 기간 선택 라디오 버튼
            time_range = st.radio(
                "📅 조회 기간 선택",
                ["1주", "1개월", "3개월", "6개월", "1년", "전체"],
                index=5,
                horizontal=True,
                key="perf_range"
            )
            
            # 데이터 필터링
            last_date = df_price['Date'].max()
            if time_range == "1주": start_date = last_date - timedelta(weeks=1)
            elif time_range == "1개월": start_date = last_date - timedelta(days=30)
            elif time_range == "3개월": start_date = last_date - timedelta(days=90)
            elif time_range == "6개월": start_date = last_date - timedelta(days=180)
            elif time_range == "1년": start_date = last_date - timedelta(days=365)
            else: start_date = df_price['Date'].min()
            
            filtered_df = df_price[df_price['Date'] >= start_date].copy()
            
            if not filtered_df.empty:
                # 2. 지표 계산
                latest_p = df_price.iloc[-1]['Price']
                prev_p = df_price.iloc[-2]['Price'] if len(df_price) > 1 else latest_p
                diff = latest_p - prev_p
                pct = (diff / prev_p * 100) if prev_p != 0 else 0

                period_max = filtered_df['Price'].max()
                period_min = filtered_df['Price'].min()

                start_price = clean_price(filtered_df.iloc[0]['Price'])
                end_price = clean_price(filtered_df.iloc[-1]['Price'])
                period_return = ((end_price - start_price) / start_price) * 100
                
                filtered_df['Daily_Return'] = filtered_df['Price'].pct_change()
                volatility = filtered_df['Daily_Return'].std() * np.sqrt(252) * 100
                
                bm_return = None
                if 'Benchmark' in filtered_df.columns:
                    filtered_df['Benchmark'] = filtered_df['Benchmark'].apply(clean_price)
                    start_bm = filtered_df.iloc[0]['Benchmark']
                    end_bm = filtered_df.iloc[-1]['Benchmark']
                    if start_bm != 0:
                        bm_return = ((end_bm - start_bm) / start_bm) * 100

                # --- 지표 레이아웃 개선 ---
                st.markdown("### 📊 주요 성과 지표")
                
                # 첫 번째 줄: 가격 관련 지표
                price_container = st.container()
                with price_container:
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("현재가", f"{latest_p:,.0f}원", f"{pct:+.2f}%")
                    with c2:
                        st.metric(f"기간 내 최고가", f"{period_max:,.0f}원")
                    with c3:
                        st.metric(f"기간 내 최저가", f"{period_min:,.0f}원")
                
                # 두 번째 줄: 수익률 및 리스크 지표
                perf_container = st.container()
                with perf_container:
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric(f"{time_range} 수익률", f"{period_return:.2f}%")
                    with c2:
                        st.metric(f"연환산 변동성", f"{volatility:.2f}%")
                    with c3:
                        if bm_return is not None:
                            st.metric("벤치마크 수익률", f"{bm_return:.2f}%", f"{period_return - bm_return:+.2f}%p")
                        else:
                            st.metric("벤치마크", "데이터 없음")
                
                st.markdown("---")

                # 3. 차트 생성
                filtered_df['ETF_Ret_Chart'] = (filtered_df['Price'] - start_price) / start_price * 100
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=filtered_df['Date'], y=filtered_df['ETF_Ret_Chart'], name='ETF 수익률', 
                    line=dict(color='#ef4444', width=3),
                    hovertemplate='ETF: %{y:.2f}%<extra></extra>'
                ))
                
                if bm_return is not None:
                    filtered_df['BM_Ret_Chart'] = (filtered_df['Benchmark'] - start_bm) / start_bm * 100
                    fig.add_trace(go.Scatter(
                        x=filtered_df['Date'], y=filtered_df['BM_Ret_Chart'], name='벤치마크 (BM)', 
                        line=dict(color='#4b5563', width=2, dash='dot'),
                        hovertemplate='BM: %{y:.2f}%<extra></extra>'
                    ))
                
                fig.update_layout(
                    template="plotly_white", hovermode="x unified",
                    yaxis_title="누적 수익률 (%)",
                    height=600,
                    plot_bgcolor='rgba(242, 242, 242, 0.6)',
                    paper_bgcolor='white',
                    font=dict(color="black", size=12),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    xaxis=dict(showgrid=True, gridcolor='white'),
                    yaxis=dict(showgrid=True, gridcolor='white')
                )

                fig.update_xaxes(tickformat="%Y년 %m월", hoverformat="%Y년 %m월 %d일")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("선택한 기간에 해당하는 데이터가 없습니다.")

    with tab2:
        col_bar, col_pie = st.columns([1, 1])
        with col_bar:
            st.subheader("분배금 지급 현황")
            if upload_div:
                df_div = pd.read_excel(upload_div) if upload_div.name.endswith('xlsx') else pd.read_csv(upload_div)
            else:
                df_div = pd.DataFrame({'날짜': ['24-01', '24-04', '24-07', '24-10'], '분배금': [100, 450, 150, 120]})
            
            fig_div = px.bar(df_div, x=df_div.columns[0], y=df_div.columns[1], text_auto=',.0f', color_discrete_sequence=['#3b82f6'])
            fig_div.update_layout(
                template="plotly_white",
                height=450,
                plot_bgcolor='rgba(242, 242, 242, 0.6)',
                yaxis_title="분배금 (원)",
                font=dict(color="black")
            )
            st.plotly_chart(fig_div, use_container_width=True)

        with col_pie:
            st.subheader("상위 10개 구성종목 비중")
            if 'Name' in df_const.columns and 'Weight' in df_const.columns:
                df_const['Weight'] = df_const['Weight'].apply(clean_price)
                top_10 = df_const.sort_values(by='Weight', ascending=False).head(10)
                fig_pie = px.pie(top_10, names='Name', values='Weight', hole=0.4,
                                color_discrete_sequence=px.colors.qualitative.T10)
                fig_pie.update_layout(
                    template="plotly_white",
                    height=450,
                    plot_bgcolor='rgba(242, 242, 242, 0.6)',
                    font=dict(color="black")
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)

    with tab3:
        st.subheader("구성종목 주요 이슈")
        if upload_issues:
            df_is = pd.read_excel(upload_issues) if upload_issues.name.endswith('xlsx') else pd.read_csv(upload_issues)
            stocks = df_is[df_is.columns[1]].unique()
            selected_is_stock = st.selectbox("종목 선택", stocks)
            filtered_is = df_is[df_is[df_is.columns[1]] == selected_is_stock]
            for _, row in filtered_is.iterrows():
                with st.expander(f"[{row[df_is.columns[0]]}] {row[df_is.columns[1]]}"):
                    st.write(row[df_is.columns[2]])
        else:
            st.info("이슈 데이터 파일을 업로드해주세요.")

    with tab4:
        st.subheader("종목별 상세 재무제표")
        if upload_financial:
            xls = pd.ExcelFile(upload_financial)
            selected_fin_stock = st.selectbox("분석할 종목 선택", xls.sheet_names)
            df_fin = pd.read_excel(xls, sheet_name=selected_fin_stock)
            st.dataframe(df_fin, use_container_width=True, height=600)
        else:
            st.info("재무데이터(다중 시트 엑셀)를 업로드해주세요.")

    # --- 데이터 다운로드 섹션 ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("📥 데이터 내보내기")
    if st.sidebar.button("분석 리포트 CSV 생성"):
        csv = df_price.to_csv(index=False).encode('utf-8-sig')
        st.sidebar.download_button("CSV 다운로드", data=csv, file_name="etf_report.csv", mime="text/csv")

if __name__ == "__main__":
    main()

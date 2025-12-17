import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 페이지 설정
st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")


def main():
    st.title("📈 주가 데이터 시각화 도구")
    st.markdown("""
    엑셀(.xlsx) 또는 CSV 파일을 업로드하여 주가 차트를 확인하세요.
    파일에는 **Date(날짜), Open(시가), High(고가), Low(저가), Close(종가)** 컬럼이 포함되어 있어야 합니다.
    """)

    # 사이드바: 파일 업로드
    st.sidebar.header("설정")
    uploaded_file = st.sidebar.file_uploader("파일 업로드", type=["xlsx", "csv"])

    if uploaded_file is not None:
        try:
            # 파일 읽기
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            # 날짜 형식 변환
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.sort_values('Date')
            else:
                st.error("'Date' 컬럼을 찾을 수 없습니다.")
                return

            # 데이터 프리뷰
            with st.expander("데이터 미리보기"):
                st.dataframe(df.head())

            # 필수 컬럼 체크
            required_cols = ['Open', 'High', 'Low', 'Close']
            if all(col in df.columns for col in required_cols):

                # 차트 생성 (캔들스틱 + 거래량)
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                    vertical_spacing=0.03, subplot_titles=('Candlestick', 'Volume'),
                                    row_width=[0.2, 0.7])

                # 캔들스틱 차트 추가
                fig.add_trace(go.Candlestick(
                    x=df['Date'],
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name="Price"
                ), row=1, col=1)

                # 거래량이 있는 경우 추가
                if 'Volume' in df.columns:
                    fig.add_trace(go.Bar(
                        x=df['Date'],
                        y=df['Volume'],
                        name="Volume",
                        marker_color='rgba(100, 149, 237, 0.5)'
                    ), row=2, col=1)

                # 레이아웃 업데이트
                fig.update_layout(
                    title_text=f"{uploaded_file.name} 분석 결과",
                    xaxis_rangeslider_visible=False,
                    height=800,
                    template="plotly_white"
                )

                st.plotly_chart(fig, use_container_width=True)

                # 통계 요약
                st.subheader("📊 주요 통계")
                col1, col2, col3, col4 = st.columns(4)

                last_price = df['Close'].iloc[-1]
                prev_price = df['Close'].iloc[-2] if len(df) > 1 else last_price
                diff = last_price - prev_price
                pct_change = (diff / prev_price) * 100

                col1.metric("현재가 (종가)", f"{last_price:,.0f}", f"{diff:,.0f} ({pct_change:.2f}%)")
                col2.metric("최고가 (기간 내)", f"{df['High'].max():,.0f}")
                col3.metric("최저가 (기간 내)", f"{df['Low'].max():,.0f}")
                col4.metric("평균 거래량", f"{df['Volume'].mean():,.0f}" if 'Volume' in df.columns else "N/A")

            else:
                st.warning(f"필수 컬럼이 부족합니다: {', '.join(required_cols)}")

        except Exception as e:
            st.error(f"파일을 처리하는 중 오류가 발생했습니다: {e}")
    else:
        st.info("왼쪽 사이드바에서 파일을 업로드해 주세요.")

        # 샘플 데이터 안내
        st.subheader("💡 엑셀 양식 예시")
        example_data = {
            'Date': ['2023-01-01', '2023-01-02'],
            'Open': [100, 110],
            'High': [115, 120],
            'Low': [95, 105],
            'Close': [110, 118],
            'Volume': [1000, 1500]
        }
        st.table(pd.DataFrame(example_data))


if __name__ == "__main__":
    main()

import asyncio
import json
import os
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from multi_agent_stock_research import TickerData, analyze_ticker


st.set_page_config(page_title="AI Stock Research Terminal", page_icon="📈", layout="wide")


def parse_json_text(value: str, field_name: str) -> dict[str, Any]:
    """Parse a JSON text field, returning an empty dict if blank or invalid."""
    if value is None or value.strip() == "":
        return {}
    try:
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
        raise ValueError(f"{field_name} must be a JSON object.")
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {field_name}: {exc.msg}") from exc


def parse_lines(value: str) -> list[str]:
    return [line.strip() for line in value.splitlines() if line.strip()]


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def score_to_color(score: Any) -> str:
    try:
        score_value = float(score)
    except (TypeError, ValueError):
        return "#6c757d"
    if score_value >= 0.5:
        return "#2ecc71"
    if score_value >= 0.2:
        return "#f4c542"
    return "#ff5c5c"


def build_division_summary(result: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for item in result.get("division_outputs", []):
        rows.append(
            {
                "division": item.get("division", "Unknown"),
                "score": safe_float(item.get("composite_score"), 0.0),
                "confidence": safe_float(item.get("confidence"), 0.0),
                "summary": item.get("summary", "No summary available."),
            }
        )
    return pd.DataFrame(rows)


def build_leaf_summary(result: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for division in result.get("division_outputs", []):
        division_name = division.get("division", "Unknown")
        for analyst in division.get("sub_analyses", []):
            rows.append(
                {
                    "division": division_name,
                    "analyst": analyst.get("analyst", "Unknown"),
                    "score": safe_float(analyst.get("score"), 0.0),
                    "confidence": safe_float(analyst.get("confidence"), 0.0),
                    "rationale": analyst.get("rationale", "No rationale."),
                    "key_data_points": "; ".join(analyst.get("key_data_points", [])) or "-",
                }
            )
    return pd.DataFrame(rows)


def build_watchlist_df(final: dict[str, Any], current_price: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Ticker": final.get("ticker", "N/A"),
                "Sector": final.get("sector", "N/A"),
                "Price": round(current_price, 2),
                "Target": round(safe_float(final.get("target_price"), 0.0), 2),
                "Buy Low": round(safe_float(final.get("buy_price_low"), 0.0), 2),
                "Buy High": round(safe_float(final.get("buy_price_high"), 0.0), 2),
                "Upside %": round(safe_float(final.get("upside_pct"), 0.0), 2),
                "Conviction": final.get("conviction", "N/A"),
            }
        ]
    )


def build_sentiment_df(result: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for division in result.get("division_outputs", []):
        rows.append(
            {
                "Division": division.get("division", "Unknown"),
                "Score": safe_float(division.get("composite_score"), 0.0),
                "Confidence": safe_float(division.get("confidence"), 0.0),
                "Signal": "Bullish" if safe_float(division.get("composite_score"), 0.0) > 0 else "Neutral" if safe_float(division.get("composite_score"), 0.0) == 0 else "Bearish",
            }
        )
    return pd.DataFrame(rows)


def build_price_chart(final: dict[str, Any], current_price: float) -> go.Figure:
    buy_low = safe_float(final.get("buy_price_low"), 0.0)
    buy_high = safe_float(final.get("buy_price_high"), 0.0)
    target = safe_float(final.get("target_price"), current_price)

    dates = pd.date_range("2025-08-01", periods=12, freq="D")
    base = current_price
    opens = [base * (0.995 + i * 0.0015) for i in range(12)]
    closes = [base * (1.002 + i * 0.0015) for i in range(12)]
    highs = [max(open_, close_) * 1.01 for open_, close_ in zip(opens, closes)]
    lows = [min(open_, close_) * 0.99 for open_, close_ in zip(opens, closes)]

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=dates,
                open=opens,
                high=highs,
                low=lows,
                close=closes,
                increasing_line_color="#2ecc71",
                decreasing_line_color="#ff5c5c",
                name="Price",
            )
        ]
    )

    fig.add_hline(y=current_price, line_dash="solid", line_color="#bdc3c7", annotation_text="Current")
    fig.add_hline(y=buy_low, line_dash="dot", line_color="#f39c12", annotation_text="Buy Low")
    fig.add_hline(y=target, line_dash="dash", line_color="#3498db", annotation_text="Target")
    fig.add_hline(y=buy_high, line_dash="dot", line_color="#8e44ad", annotation_text="Buy High")

    fig.update_layout(
        template="plotly_dark",
        title=f"{final.get('ticker', 'Ticker')} Price Action",
        xaxis_title="Date",
        yaxis_title="Price",
        margin=dict(l=20, r=20, t=40, b=20),
        height=360,
    )
    return fig


def build_risk_badges(final: dict[str, Any]) -> list[str]:
    return final.get("risk_flags", []) or ["No major risks flagged."]


st.markdown(
    """
    <style>
        .main .block-container { padding-top: 1rem; }
        .stApp { background: linear-gradient(180deg, #0b1220 0%, #101827 100%); color: white; }
        .subheader { color: #f8fafc; }
        [data-testid="stMetricValue"] { font-size: 1.2rem; }
        div[data-testid="stVerticalBlock"] > div { background: rgba(15, 23, 42, 0.60); border: 1px solid rgba(148, 163, 184, 0.18); border-radius: 14px; }
        .stTabs [role="tablist"] button { background: rgba(15, 23, 42, 0.80); color: #dfe7f2; border-radius: 10px; }
        .stTabs [role="tablist"] .st-bj { background: rgba(30, 41, 59, 0.75); }
    </style>
    """,
    unsafe_allow_html=True,
)


st.title("📈 AI Stock Research Terminal")
st.caption("Multi-layer investment research built from structed market intelligence and model synthesis.")

with st.sidebar:
    st.header("Research setup")
    api_key = st.text_input("Anthropic API key", type="password", value=os.getenv("ANTHROPIC_API_KEY", ""))
    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key

    st.markdown("---")
    st.caption("Replace the sample data with your own research inputs and run the pipeline.")


def build_demo_form() -> dict[str, str]:
    return {
        "ticker": "EXAMPLE.NS",
        "sector": "Industrials",
        "current_price": "1310.0",
        "news_items": (
            "Company announced a new capex plan for capacity expansion.\n"
            "Analyst commentary mixed on near-term margin pressure."
        ),
        "financial_statements": json.dumps(
            {
                "revenue_growth_yoy": 0.14,
                "ebitda_margin": 0.18,
                "net_debt_to_ebitda": 1.2,
            },
            indent=2,
        ),
        "industry_notes": "Sector benefiting from import substitution policy push.",
        "geopolitical_notes": "Minor exposure to tariff changes on input steel.",
        "volume_data": json.dumps({"avg_volume_30d": 1200000, "recent_volume_spike": True}, indent=2),
        "insider_filings": "Promoter increased stake by 0.5% last quarter.",
        "fund_holdings": json.dumps({"institutional_ownership_pct": 22.5, "trend": "increasing"}, indent=2),
        "short_interest": json.dumps({"days_to_cover": 1.1}, indent=2),
    }


defaults = build_demo_form()

with st.form("research_form"):
    col1, col2 = st.columns(2)
    with col1:
        ticker = st.text_input("Ticker", value=defaults["ticker"])
        sector = st.text_input("Sector", value=defaults["sector"])
        current_price = st.number_input("Current price", min_value=0.0, value=float(defaults["current_price"]), step=1.0)
    with col2:
        news_items = st.text_area("News items (one per line)", value=defaults["news_items"], height=120)

    st.subheader("Research data")
    tab1, tab2, tab3 = st.tabs(["Fundamentals", "Signals", "Macro / Ownership"])

    with tab1:
        financial_statements = st.text_area("Financial statements JSON", value=defaults["financial_statements"], height=170)
        industry_notes = st.text_area("Industry notes", value=defaults["industry_notes"], height=110)
        geopolitical_notes = st.text_area("Geopolitical notes", value=defaults["geopolitical_notes"], height=110)

    with tab2:
        volume_data = st.text_area("Volume data JSON", value=defaults["volume_data"], height=150)
        insider_filings = st.text_area("Insider filings (one per line)", value=defaults["insider_filings"], height=110)

    with tab3:
        fund_holdings = st.text_area("Fund holdings JSON", value=defaults["fund_holdings"], height=150)
        short_interest = st.text_area("Short interest JSON", value=defaults["short_interest"], height=150)

    submitted = st.form_submit_button("Run analysis", use_container_width=True, type="primary")

if submitted:
    try:
        if not os.getenv("ANTHROPIC_API_KEY"):
            st.error("Please provide an Anthropic API key in the sidebar before running the analysis.")
            st.stop()

        data = TickerData(
            ticker=ticker,
            sector=sector,
            current_price=float(current_price),
            news_items=parse_lines(news_items),
            financial_statements=parse_json_text(financial_statements, "Financial statements"),
            industry_notes=industry_notes,
            geopolitical_notes=geopolitical_notes,
            volume_data=parse_json_text(volume_data, "Volume data"),
            insider_filings=parse_lines(insider_filings),
            fund_holdings=parse_json_text(fund_holdings, "Fund holdings"),
            short_interest=parse_json_text(short_interest, "Short interest"),
        )

        with st.spinner("Running the research engine..."):
            result = asyncio.run(analyze_ticker(data))

        final = result.get("final", {})
        division_df = build_division_summary(result)
        analyst_df = build_leaf_summary(result)
        watchlist_df = build_watchlist_df(final, float(current_price))
        sentiment_df = build_sentiment_df(result)

        st.success("Research completed successfully.")

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Ticker", final.get("ticker", ticker))
        k2.metric("Composite Score", f"{safe_float(final.get('composite_score'), 0.0):.2f}")
        k3.metric("Target Price", f"${safe_float(final.get('target_price'), 0.0):,.2f}")
        k4.metric("Conviction", final.get("conviction", "N/A"))

        with st.container():
            left, right = st.columns([2, 1])

            with left:
                st.plotly_chart(build_price_chart(final, float(current_price)), use_container_width=True)

            with right:
                st.markdown("### Thesis")
                st.write(final.get("rationale", "No rationale available."))
                st.markdown("### Risk flags")
                for risk in build_risk_badges(final):
                    st.markdown(f"- {risk}")

        st.download_button(
            label="Download research JSON",
            data=json.dumps(result, indent=2),
            file_name=f"{ticker}_research.json",
            mime="application/json",
            use_container_width=True,
        )

        tabs = st.tabs(["Dashboard", "Division Scorecard", "Analyst Detail", "Watchlist", "Audit Trail"])

        with tabs[0]:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Buy Range", f"${safe_float(final.get('buy_price_low'), 0.0):,.2f} - ${safe_float(final.get('buy_price_high'), 0.0):,.2f}")
            c2.metric("Upside", f"{safe_float(final.get('upside_pct'), 0.0):.1f}%")
            c3.metric("Current Price", f"${float(current_price):,.2f}")
            c4.metric("Compliance Review", "Required" if final.get("requires_compliance_review") else "Not required")

            st.markdown("### Division sentiment")
            if not sentiment_df.empty:
                st.dataframe(
                    sentiment_df.style.background_gradient(cmap="RdYlGn", subset=["Score", "Confidence"]),
                    use_container_width=True,
                )
            else:
                st.info("No division sentiment data available.")

        with tabs[1]:
            st.markdown("### Division scorecard")
            if not division_df.empty:
                st.dataframe(
                    division_df.style.applymap(lambda v: f"color: {score_to_color(v)}" if isinstance(v, (int, float)) else ""),
                    use_container_width=True,
                )
                st.bar_chart(division_df.set_index("division")["score"], use_container_width=True)
            else:
                st.info("No division scorecard available.")

        with tabs[2]:
            st.markdown("### Analyst detail")
            if not analyst_df.empty:
                st.dataframe(analyst_df, use_container_width=True)
            else:
                st.info("No analyst detail available.")

        with tabs[3]:
            st.markdown("### Watchlist")
            st.dataframe(watchlist_df, use_container_width=True)

        with tabs[4]:
            st.markdown("### Full audit trail")
            st.json(result)

    except ValueError as exc:
        st.error(f"Input error: {exc}")
    except Exception as exc:
        st.error(f"Research failed: {exc}")

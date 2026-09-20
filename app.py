import asyncio
import json
import os
from typing import Any

import pandas as pd
import streamlit as st

from multi_agent_stock_research import TickerData, analyze_ticker


st.set_page_config(page_title="AI Stock Research Dashboard", page_icon="📊", layout="wide")


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
        return "#1f9d55"
    if score_value >= 0.2:
        return "#f0ad4e"
    return "#d9534f"


def build_division_summary(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in result.get("division_outputs", []):
        rows.append(
            {
                "division": item.get("division", "Unknown"),
                "score": safe_float(item.get("composite_score"), 0.0),
                "confidence": safe_float(item.get("confidence"), 0.0),
                "summary": item.get("summary", "No summary available."),
            }
        )
    return rows


def build_leaf_summary(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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
    return rows


st.markdown(
    """
    <style>
        .main .block-container {
            padding-top: 2rem;
        }
        .stApp {
            background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
            color: #e5e7eb;
        }
        .stMetric > div {
            background: rgba(15, 23, 42, 0.75);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 14px;
            padding: 0.9rem 1rem;
        }
        h1, h2, h3, h4, h5 {
            color: #f8fafc !important;
        }
        .stTabs [role="tablist"] {
            gap: 0.6rem;
        }
        .stTabs [role="tab"] {
            border-radius: 10px 10px 0 0;
            padding: 0.5rem 1rem;
            background: rgba(30, 41, 59, 0.8);
        }
        .block-container {
            max-width: 1500px;
        }
        .card {
            background: rgba(15, 23, 42, 0.68);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 14px;
            padding: 1rem;
            margin-bottom: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


st.title("📈 AI Stock Research Dashboard")
st.caption("Multi-layer equity research across news, fundamentals, industry, and ownership signals.")

with st.sidebar:
    st.header("Research setup")
    api_key = st.text_input("Anthropic API key", type="password", value=os.getenv("ANTHROPIC_API_KEY", ""))
    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key

    st.markdown("---")
    st.caption("Use the sample inputs below or replace values with your own research data.")

    if st.button("Load sample data"):
        st.session_state["sample_loaded"] = True


def build_demo_form():
    return {
        "ticker": "EXAMPLE.NS",
        "sector": "Industrials",
        "current_price": 1310.0,
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


if "sample_loaded" in st.session_state:
    defaults = build_demo_form()
else:
    defaults = build_demo_form()

with st.form("research_form"):
    col1, col2 = st.columns(2)
    with col1:
        ticker = st.text_input("Ticker", value=defaults["ticker"])
        sector = st.text_input("Sector", value=defaults["sector"])
        current_price = st.number_input("Current price", min_value=0.0, value=defaults["current_price"], step=1.0)
    with col2:
        news_items = st.text_area("News items (one per line)", value=defaults["news_items"], height=120)

    st.subheader("Core research inputs")
    tab1, tab2, tab3 = st.tabs(["Fundamentals", "Market Signals", "Macro / Ownership"])

    with tab1:
        financial_statements = st.text_area("Financial statements JSON", value=defaults["financial_statements"], height=180)
        industry_notes = st.text_area("Industry notes", value=defaults["industry_notes"], height=120)
        geopolitical_notes = st.text_area("Geopolitical notes", value=defaults["geopolitical_notes"], height=120)

    with tab2:
        volume_data = st.text_area("Volume data JSON", value=defaults["volume_data"], height=150)
        insider_filings = st.text_area("Insider filings (one per line)", value=defaults["insider_filings"], height=100)

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

        with st.spinner("Running the research pipeline..."):
            result = asyncio.run(analyze_ticker(data))

        final = result.get("final", {})

        st.success("Research completed successfully.")

        final_score = safe_float(final.get("composite_score"), 0.0)
        target_price = safe_float(final.get("target_price"), 0.0)
        buy_low = safe_float(final.get("buy_price_low"), 0.0)
        buy_high = safe_float(final.get("buy_price_high"), 0.0)
        upside_pct = safe_float(final.get("upside_pct"), 0.0)
        current = float(current_price)

        st.subheader("Executive summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ticker", final.get("ticker", ticker))
        c2.metric("Sector", final.get("sector", sector))
        c3.metric("Current price", f"${current:,.2f}")
        c4.metric("Conviction", final.get("conviction", "N/A"))

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Composite score", f"{final_score:.2f}")
        c6.metric("Target price", f"${target_price:,.2f}")
        c7.metric("Buy range", f"${buy_low:,.2f} - ${buy_high:,.2f}")
        c8.metric("Upside", f"{upside_pct:.1f}%")

        st.markdown("---")

        overview_col, risk_col = st.columns([2, 1])

        with overview_col:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.write(final.get("rationale", "No rationale available."))
            st.markdown('</div>', unsafe_allow_html=True)

        with risk_col:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Risk flags")
            if final.get("risk_flags"):
                for flag in final.get("risk_flags", []):
                    st.markdown(f"- {flag}")
            else:
                st.write("No material risk flags reported.")
            st.markdown('</div>', unsafe_allow_html=True)

        division_rows = build_division_summary(result)
        division_df = pd.DataFrame(division_rows)

        leaf_rows = build_leaf_summary(result)
        leaf_df = pd.DataFrame(leaf_rows)

        pricing_df = pd.DataFrame(
            {
                "Metric": ["Current Price", "Buy Low", "Target Price", "Buy High"],
                "Value": [current, buy_low, target_price, buy_high],
            }
        )

        tabs = st.tabs(["Overview", "Charts", "Division scores", "Analyst detail", "Audit trail"])

        with tabs[0]:
            st.subheader("Market positioning")
            st.dataframe(
                pricing_df.style.apply(
                    lambda x: [f"color: {score_to_color(x['Value'] if x.name == 1 else 0)}" for _ in x],
                    axis=1,
                ),
                use_container_width=True,
            )

        with tabs[1]:
            st.subheader("Division score comparison")
            if not division_df.empty:
                st.bar_chart(division_df.set_index("division")["score"], use_container_width=True)

            st.subheader("Target price range")
            st.bar_chart(pricing_df.set_index("Metric")["Value"], use_container_width=True)

        with tabs[2]:
            st.subheader("Division summary table")
            if not division_df.empty:
                st.dataframe(division_df, use_container_width=True)

        with tabs[3]:
            st.subheader("Analyst-level detail")
            if not leaf_df.empty:
                st.dataframe(leaf_df, use_container_width=True)
            else:
                st.info("No leaf analysis available yet.")

        with tabs[4]:
            st.subheader("Full research result")
            st.json(result)

    except ValueError as exc:
        st.error(f"Input error: {exc}")
    except Exception as exc:
        st.error(f"Research failed: {exc}")

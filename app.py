import asyncio
import json
import os
from typing import Any

import streamlit as st

from multi_agent_stock_research import TickerData, analyze_ticker


st.set_page_config(page_title="AI Stock Research", page_icon="📈", layout="wide")


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
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    return lines


st.title("Multi-Layer AI Stock Research")
st.caption("Run equity research across news, fundamentals, industry, and ownership signals.")

with st.sidebar:
    st.header("API credentials")
    api_key = st.text_input("Anthropic API key", type="password", value=os.getenv("ANTHROPIC_API_KEY", ""))
    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key

    st.markdown("---")
    st.caption("Tip: you can also export ANTHROPIC_API_KEY before running the app.")


with st.form("research_form"):
    col1, col2 = st.columns(2)

    with col1:
        ticker = st.text_input("Ticker", value="EXAMPLE.NS")
        sector = st.text_input("Sector", value="Industrials")
        current_price = st.number_input("Current price", min_value=0.0, value=1310.0, step=1.0)

    with col2:
        news_items = st.text_area(
            "News items (one per line)",
            value=(
                "Company announced a new capex plan for capacity expansion.\n"
                "Analyst commentary mixed on near-term margin pressure."
            ),
            height=150,
        )

    st.subheader("Research inputs")
    financial_statements = st.text_area(
        "Financial statements JSON",
        value=json.dumps(
            {
                "revenue_growth_yoy": 0.14,
                "ebitda_margin": 0.18,
                "net_debt_to_ebitda": 1.2,
            },
            indent=2,
        ),
        height=150,
    )

    industry_notes = st.text_area("Industry notes", value="Sector benefiting from import substitution policy push.")
    geopolitical_notes = st.text_area("Geopolitical notes", value="Minor exposure to tariff changes on input steel.")

    volume_data = st.text_area(
        "Volume data JSON",
        value=json.dumps({"avg_volume_30d": 1200000, "recent_volume_spike": True}, indent=2),
        height=120,
    )

    insider_filings = st.text_area(
        "Insider filings (one per line)",
        value="Promoter increased stake by 0.5% last quarter.",
        height=100,
    )

    fund_holdings = st.text_area(
        "Fund holdings JSON",
        value=json.dumps({"institutional_ownership_pct": 22.5, "trend": "increasing"}, indent=2),
        height=120,
    )

    short_interest = st.text_area(
        "Short interest JSON",
        value=json.dumps({"days_to_cover": 1.1}, indent=2),
        height=120,
    )

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

        st.subheader("Final investment call")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Ticker", final.get("ticker", ticker))
        with col_b:
            st.metric("Composite score", f"{final.get('composite_score', 0):.2f}")
        with col_c:
            st.metric("Conviction", final.get("conviction", "N/A"))

        st.write(final.get("rationale", "No rationale provided."))

        st.json({
            "target_price": final.get("target_price"),
            "buy_price_low": final.get("buy_price_low"),
            "buy_price_high": final.get("buy_price_high"),
            "upside_pct": final.get("upside_pct"),
            "horizon": final.get("horizon"),
            "risk_flags": final.get("risk_flags", []),
            "requires_compliance_review": final.get("requires_compliance_review", False),
        })

        st.subheader("Detailed output")
        st.json(result)

    except ValueError as exc:
        st.error(f"Input error: {exc}")
    except Exception as exc:
        st.error(f"Research failed: {exc}")

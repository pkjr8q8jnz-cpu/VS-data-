"""
Multi-Layer AI Stock Research System
=====================================

Implements the architecture from multi_agent_stock_research_spec.md:

    Layer 0: CMIO (orchestrator / synthesizer)
    Layer 1: 5 Division Heads
    Layer 2: Leaf analysts (parallel fan-out per division)

Requires:
    pip install anthropic

Usage:
    export ANTHROPIC_API_KEY=...
    python multi_agent_stock_research.py

You plug in your own data (news text, financial statements, ownership
data, etc.) via the `TickerData` object below — this script does not
fetch live market data itself.
"""

import asyncio
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from anthropic import AsyncAnthropic

MODEL = "claude-sonnet-4-6"
client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


# ---------------------------------------------------------------------------
# Input data container — fill this in with real data per ticker
# ---------------------------------------------------------------------------

@dataclass
class TickerData:
    ticker: str
    sector: str
    current_price: float
    # Raw text/data blobs; keep as strings/dicts, agents will reason over them
    news_items: list[str] = field(default_factory=list)
    financial_statements: dict[str, Any] = field(default_factory=dict)
    industry_notes: str = ""
    geopolitical_notes: str = ""
    volume_data: dict[str, Any] = field(default_factory=dict)
    insider_filings: list[str] = field(default_factory=list)
    fund_holdings: dict[str, Any] = field(default_factory=dict)
    short_interest: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core: one call to Claude that MUST return JSON matching a given shape
# ---------------------------------------------------------------------------

async def call_agent(system_prompt: str, user_content: str) -> dict:
    """Calls the model with a strict 'JSON only' instruction and parses it."""
    resp = await client.messages.create(
        model=MODEL,
        max_tokens=1000,
        system=system_prompt + "\n\nRespond with ONLY a valid JSON object. "
        "No prose, no markdown fences, no preamble.",
        messages=[{"role": "user", "content": user_content}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    text = text.strip("`")
    if text.startswith("json"):
        text = text[4:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"error": "unparseable_response", "raw": text}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Layer 2 — Leaf analyst definitions
# Each entry: (analyst_name, system_prompt_builder, data_selector)
# ---------------------------------------------------------------------------

def leaf_prompt(role: str, focus: str) -> str:
    return (
        f"You are the {role} inside an equity research desk. "
        f"Your sole focus: {focus}. "
        "Given the data provided, return a JSON object with exactly these "
        "fields: analyst (string), ticker (string), score (float -1.0 to "
        "1.0), confidence (float 0.0 to 1.0), rationale (string, 2-3 "
        "sentences), key_data_points (array of short strings). "
        "If data is insufficient, use confidence below 0.3 rather than "
        "inventing figures."
    )


NEWS_ANALYSTS = [
    ("research_analyst", "identifying and tagging significant news events and themes"),
    ("sentiment_analyst", "scoring the sentiment of news/text flow"),
    ("stock_movement_analyst", "determining whether price moves are news-driven or technical"),
    ("news_credibility_analyst", "weighting news reliability by source quality"),
    ("market_impact_analyst", "estimating magnitude/duration of news impact on price"),
]

EQUITY_ANALYSTS = [
    ("fundamental_research_analyst", "deriving fair value via DCF/comparable multiples and assessing growth quality"),
    ("financial_forensics_analyst", "detecting accounting red flags, related-party risk, aggressive assumptions"),
    ("pre_earnings_intelligence_analyst", "estimating probability and direction of an earnings surprise"),
]

INDUSTRY_ANALYSTS = [
    ("industry_research_analyst", "assessing sector-level tailwinds/headwinds, capacity, and pricing power"),
    ("geopolitical_risk_analyst", "assessing trade policy, regulatory, and FX risk overlays"),
]

OWNERSHIP_ANALYSTS = [
    ("market_volume_intelligence_analyst", "reading volume/open-interest patterns for accumulation or distribution"),
    ("insider_activity_analyst", "scoring conviction implied by insider buy/sell filings"),
    ("fund_holdings_research_analyst", "assessing institutional/smart-money positioning"),
    ("short_seller_research_analyst", "assessing crowded-short risk from short interest data"),
]


async def run_leaf(name: str, focus: str, ticker: str, data_str: str) -> dict:
    prompt = leaf_prompt(name, focus)
    result = await call_agent(prompt, f"Ticker: {ticker}\n\nData:\n{data_str}")
    result.setdefault("analyst", name)
    result.setdefault("ticker", ticker)
    result["as_of"] = now_iso()
    return result


# ---------------------------------------------------------------------------
# Layer 1 — Division heads (aggregate their leaf analysts)
# ---------------------------------------------------------------------------

async def division_head(
    division_name: str,
    analysts: list[tuple[str, str]],
    ticker: str,
    data_str: str,
    fair_value_from: Optional[str] = None,
) -> dict:
    # Fan out to all leaf analysts in this division concurrently
    leaf_results = await asyncio.gather(
        *[run_leaf(name, focus, ticker, data_str) for name, focus in analysts]
    )

    aggregation_prompt = (
        f"You are the {division_name.replace('_', ' ').title()}, head of a "
        "research division. Your analysts have each returned a scored "
        "verdict (see data). Synthesize them into ONE JSON object with "
        "fields: division (string), ticker (string), composite_score "
        "(float -1.0 to 1.0, confidence-weighted average of your analysts), "
        "confidence (float 0.0 to 1.0), summary (2-3 sentences), "
        "sub_analyses (pass through the array you were given unchanged)."
        + (
            " Also include fair_value_estimate (float, from the fundamental "
            "analyst's reasoning) if derivable, else null."
            if fair_value_from
            else ""
        )
    )
    agg = await call_agent(
        aggregation_prompt,
        f"Ticker: {ticker}\nAnalyst outputs:\n{json.dumps(leaf_results, indent=2)}",
    )
    agg.setdefault("division", division_name)
    agg.setdefault("ticker", ticker)
    agg["sub_analyses"] = leaf_results
    return agg


# ---------------------------------------------------------------------------
# Layer 1 special — Independent Research Review Lead (adversarial)
# ---------------------------------------------------------------------------

async def review_lead(ticker: str, division_outputs: list[dict], current_price: float) -> dict:
    context = json.dumps(division_outputs, indent=2)

    bull_task = call_agent(
        "You are the Bull Case Analyst. Build the strongest, most rigorous "
        "long thesis for this ticker using the division research provided — "
        "but do not simply agree with it; independently justify your own "
        "numbers. Return JSON: {score (float 0-1), target_price (float), "
        "thesis (string, 3-4 sentences)}.",
        f"Ticker: {ticker}\nCurrent price: {current_price}\nDivision research:\n{context}",
    )
    bear_task = call_agent(
        "You are the Bear Case Analyst. Build the strongest, most rigorous "
        "short/avoid thesis for this ticker using the division research "
        "provided — independently justify your own numbers, don't just "
        "invert the bull case. Return JSON: {score (float -1-0), "
        "downside_price (float), thesis (string, 3-4 sentences)}.",
        f"Ticker: {ticker}\nCurrent price: {current_price}\nDivision research:\n{context}",
    )
    validation_task = call_agent(
        "You are the Research Validation Analyst. Audit the division "
        "research for internal consistency, stale or missing data, and "
        "conflicting signals between divisions (e.g. strong sentiment vs "
        "weak fundamentals). Return JSON: {validation_flags (array of short "
        "strings, empty if none), net_adjustment (float -0.3 to 0.3, how "
        "much to adjust the eventual composite score for the issues you "
        "found)}.",
        f"Ticker: {ticker}\nDivision research:\n{context}",
    )

    bull, bear, validation = await asyncio.gather(bull_task, bear_task, validation_task)

    return {
        "division": "independent_review",
        "ticker": ticker,
        "bull_case": bull,
        "bear_case": bear,
        "validation_flags": validation.get("validation_flags", []),
        "net_adjustment": validation.get("net_adjustment", 0.0),
    }


# ---------------------------------------------------------------------------
# Layer 0 — CMIO: final synthesis into buy price / target price / conviction
# ---------------------------------------------------------------------------

async def cmio_synthesize(ticker: str, sector: str, current_price: float,
                           division_outputs: list[dict], review: dict) -> dict:
    prompt = (
        "You are the Chief Market Intelligence Officer. You receive "
        "composite research from four divisions (news, equity/fundamental, "
        "industry/geopolitical, ownership/activity) plus an independent "
        "Bull Case, Bear Case, and Validation audit. Synthesize a FINAL "
        "call. Rules:\n"
        "1. Do not invent data not implied by the inputs.\n"
        "2. target_price should reflect a weighted blend of the "
        "fundamental fair_value_estimate and the bull case target_price, "
        "pulled toward the bear case if validation flags are serious.\n"
        "3. buy_price_low/high should bracket current_price modestly, "
        "narrower if ownership data shows active accumulation, wider or "
        "flagged if distribution/short pressure is evident.\n"
        "4. conviction is 'High' only if bull/bear divergence is narrow "
        "AND confidence across divisions is high; otherwise 'Medium' or "
        "'Low'.\n"
        "5. Always set requires_compliance_review to true.\n\n"
        "Return JSON with exactly these fields: ticker, sector, "
        "composite_score (float -1 to 1), conviction (High/Medium/Low), "
        "buy_price_low (float), buy_price_high (float), target_price "
        "(float), upside_pct (float), horizon (string), rationale "
        "(3-5 sentence paragraph), risk_flags (array of strings), "
        "requires_compliance_review (boolean, always true)."
    )
    payload = {
        "ticker": ticker,
        "sector": sector,
        "current_price": current_price,
        "division_outputs": division_outputs,
        "independent_review": review,
    }
    result = await call_agent(prompt, json.dumps(payload, indent=2))
    result.setdefault("ticker", ticker)
    result.setdefault("sector", sector)
    result["requires_compliance_review"] = True
    return result


# ---------------------------------------------------------------------------
# Top-level orchestration for one ticker
# ---------------------------------------------------------------------------

async def analyze_ticker(data: TickerData) -> dict:
    news_str = "\n".join(data.news_items) or "No news provided."
    equity_str = json.dumps(data.financial_statements) or "{}"
    industry_str = f"{data.industry_notes}\n\nGeopolitical:\n{data.geopolitical_notes}"
    ownership_str = json.dumps({
        "volume_data": data.volume_data,
        "insider_filings": data.insider_filings,
        "fund_holdings": data.fund_holdings,
        "short_interest": data.short_interest,
    })

    # Layer 1 (four data-gathering divisions) run concurrently
    news_div, equity_div, industry_div, ownership_div = await asyncio.gather(
        division_head("market_news_research", NEWS_ANALYSTS, data.ticker, news_str),
        division_head("equity_research", EQUITY_ANALYSTS, data.ticker, equity_str,
                      fair_value_from="fundamental_research_analyst"),
        division_head("industry_geopolitical_research", INDUSTRY_ANALYSTS, data.ticker, industry_str),
        division_head("ownership_activity_research", OWNERSHIP_ANALYSTS, data.ticker, ownership_str),
    )
    division_outputs = [news_div, equity_div, industry_div, ownership_div]

    # Layer 1 (adversarial) runs AFTER the above, since it audits them
    review = await review_lead(data.ticker, division_outputs, data.current_price)

    # Layer 0 — final synthesis
    final = await cmio_synthesize(data.ticker, data.sector, data.current_price,
                                   division_outputs, review)

    return {
        "final": final,
        "division_outputs": division_outputs,   # keep for audit trail
        "independent_review": review,           # keep for audit trail
    }


# ---------------------------------------------------------------------------
# Example run
# ---------------------------------------------------------------------------

async def main():
    sample = TickerData(
        ticker="EXAMPLE.NS",
        sector="Industrials",
        current_price=1310.0,
        news_items=[
            "Company announced a new capex plan for capacity expansion.",
            "Analyst commentary mixed on near-term margin pressure.",
        ],
        financial_statements={
            "revenue_growth_yoy": 0.14,
            "ebitda_margin": 0.18,
            "net_debt_to_ebitda": 1.2,
        },
        industry_notes="Sector benefiting from import substitution policy push.",
        geopolitical_notes="Minor exposure to tariff changes on input steel.",
        volume_data={"avg_volume_30d": 1200000, "recent_volume_spike": True},
        insider_filings=["Promoter increased stake by 0.5% last quarter."],
        fund_holdings={"institutional_ownership_pct": 22.5, "trend": "increasing"},
        short_interest={"days_to_cover": 1.1},
    )

    result = await analyze_ticker(sample)
    print(json.dumps(result["final"], indent=2))

    with open("research_output_full_audit.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())

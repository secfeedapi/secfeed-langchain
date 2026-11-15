"""LangChain tools for SEC EDGAR data via the SecFeed API. Set SECFEED_API_KEY in the environment."""
from typing import List, Literal, Optional

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from . import core


class FilingQuestion(BaseModel):
    ticker: str = Field(..., description="Stock ticker, e.g. NVDA")
    question: str = Field(..., description="What you want to find out from the filing")
    section: Optional[str] = Field(
        None, description="Optional: business, risk, mdna or market_risk. Picked from the question when left empty.")


class FinancialsInput(BaseModel):
    ticker: str = Field(..., description="Stock ticker, e.g. NVDA")
    form: Literal["10-K", "10-Q"] = Field("10-K", description="10-K for annual figures, 10-Q for the latest quarter")


class InsiderInput(BaseModel):
    ticker: str = Field(..., description="Stock ticker, e.g. NVDA")
    code: Optional[str] = Field(None, description="P for open market buys only, S for sales only, empty for all")
    limit: int = Field(20, description="How many transactions, up to 50")


class EightKInput(BaseModel):
    ticker: str = Field(..., description="Stock ticker, e.g. NVDA")
    limit: int = Field(10, description="How many reports, up to 50")


def _10k(ticker: str, question: str, section: Optional[str] = None) -> str:
    return core.search_filing(ticker, question, "10-K", section or None)


def _10q(ticker: str, question: str, section: Optional[str] = None) -> str:
    return core.search_filing(ticker, question, "10-Q", section or None)


def _fin(ticker: str, form: str = "10-K") -> str:
    return core.financials(ticker, form)


def _insider(ticker: str, code: Optional[str] = None, limit: int = 20) -> str:
    code = (code or "").strip().upper()[:1] or None
    return core.insider_trades(ticker, code if code in ("P", "S") else None, limit)


def _8k(ticker: str, limit: int = 10) -> str:
    return core.recent_8k(ticker, limit)


search_10k = StructuredTool.from_function(
    _10k, name="search_latest_10k", args_schema=FilingQuestion,
    description=("Reads the company's latest annual report (10-K) and returns the passages most relevant to the question, "
                 "with a link to the filing. Good for business model, risk factors and full-year results."))
search_10q = StructuredTool.from_function(
    _10q, name="search_latest_10q", args_schema=FilingQuestion,
    description=("Reads the company's latest quarterly report (10-Q) and returns the passages most relevant to the "
                 "question, with a link to the filing. Good for the most recent quarter's results and outlook."))
financial_statements = StructuredTool.from_function(
    _fin, name="financial_statements", args_schema=FinancialsInput,
    description=("Income statement, balance sheet and cash flow figures from the latest 10-K or 10-Q, taken from the "
                 "filing's XBRL data, with the last periods side by side."))
insider_trades = StructuredTool.from_function(
    _insider, name="insider_trades", args_schema=InsiderInput,
    description=("Latest Form 4 insider transactions for a company: who traded, their role, buy or sell, shares, price, "
                 "value and whether it was under a 10b5-1 plan."))
recent_8k = StructuredTool.from_function(
    _8k, name="recent_8k_reports", args_schema=EightKInput,
    description=("The company's latest 8-K current reports and what each one covers (earnings releases, material "
                 "agreements, leadership changes, other events), with links."))


def all_tools() -> List[BaseTool]:
    return [search_10k, search_10q, financial_statements, insider_trades, recent_8k]

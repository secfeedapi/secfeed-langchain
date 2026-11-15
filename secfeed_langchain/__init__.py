"""SEC EDGAR tools for LangChain and LangGraph agents, powered by the SecFeed API (https://secfeedapi.com)."""
__version__ = "0.1.0"

try:
    from .tools import (all_tools, financial_statements, insider_trades, recent_8k, search_10k,  # noqa: F401
                        search_10q)
except ImportError:  # langchain-core not installed: the plain functions in secfeed_langchain.core still work
    pass

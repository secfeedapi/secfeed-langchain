# SEC filing tools for LangChain

Give your LangChain and LangGraph agents SEC EDGAR data: 10-K and 10-Q sections, XBRL financial statements, Form 4 insider trades and 8-K reports. Powered by the [SecFeed API](https://secfeedapi.com).

No vector store or embeddings needed: SecFeed returns the exact filing section (Risk Factors, MD&A, Business), and the tool hands your agent the paragraphs that match its question.

```bash
pip install secfeed-langchain
export SECFEED_API_KEY=sf_...   # free key: https://secfeedapi.com/signup/
```

```python
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from secfeed_langchain import all_tools

agent = create_agent(ChatOpenAI(model="gpt-4o-mini"), all_tools())
reply = agent.invoke({"messages": [("user", "What are NVDA's biggest risks?")]})
print(reply["messages"][-1].content)
```

Each tool is a normal LangChain `StructuredTool`, so it also works with `llm.bind_tools(...)`, LangChain agents and any framework that accepts LangChain tools.

## Tools

| Tool | What the agent gets | API calls |
|---|---|---|
| `search_10k` | The paragraphs of the latest 10-K that answer a question, from the right section (Business, Risk Factors, MD&A, Market Risk), with the filing link | 2 |
| `search_10q` | Same for the latest 10-Q | 2 |
| `financial_statements` | Income statement, balance sheet and cash flow from XBRL, last periods side by side | 1 |
| `insider_trades` | Latest Form 4 trades: who, role, buy or sell, shares, price, value, 10b5-1 flag | 1 |
| `recent_8k` | Latest 8-K reports and the items each one covers, with links | 2 |

Repeat questions about the same filing are cached for the life of the process, so they cost no extra calls.

You can call a tool directly to see what the model sees:

```python
from secfeed_langchain import insider_trades
print(insider_trades.invoke({"ticker": "MSFT", "code": "S", "limit": 3}))
```

## Example agent

```bash
pip install secfeed-langchain langchain langchain-openai
python examples/stock_agent.py "Is AMD's growth funded by its own cash flow, and are insiders selling?"
```

Any OpenAI compatible endpoint works: set `OPENAI_BASE_URL` and `MODEL`.

## Also available

- [secfeed-crewai](https://github.com/secfeedapi/secfeed-crewai): the same tools for CrewAI, with a three agent stock research crew
- [secfeed](https://pypi.org/project/secfeed/) Python and Node SDKs for every SecFeed endpoint (13F holdings, Form D, full text search, XBRL frames and more)
- [SecFeed MCP server](https://secfeedapi.com/mcp/) for Claude, Cursor and other MCP clients

## License

MIT

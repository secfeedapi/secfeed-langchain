"""A LangChain agent that answers questions about a company from its SEC filings.

    pip install secfeed-langchain langchain langchain-openai
    export SECFEED_API_KEY=sf_...        # free key: https://secfeedapi.com/signup/
    export OPENAI_API_KEY=...
    python examples/stock_agent.py "Is NVDA's growth funded by cash flow, and are insiders selling?"

Any OpenAI compatible endpoint works: set OPENAI_BASE_URL (for example https://openrouter.ai/api/v1) and MODEL.
"""
import os
import sys

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from secfeed_langchain import all_tools

question = sys.argv[1] if len(sys.argv) > 1 else "What are NVDA's biggest risks, and how did its latest quarter go?"
llm = ChatOpenAI(model=os.environ.get("MODEL", "gpt-4o-mini"), temperature=0)
agent = create_agent(llm, all_tools(), system_prompt=(
    "You are an equity research assistant. Answer only from what the SEC filing tools return, cite the filing links, "
    "and say so when the filings do not answer the question."))

if __name__ == "__main__":
    result = agent.invoke({"messages": [("user", question)]})
    print(result["messages"][-1].content)

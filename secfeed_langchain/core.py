"""Plain functions behind the LangChain tools (same code as secfeed-crewai). No LangChain import here.

Every function returns compact text sized for an LLM context window, and costs one or two SecFeed API calls.
"""
import os
import re

from secfeed import Client, SecFeedError

# Which part of a filing answers which kind of question. 10-K items: 1 Business, 1A Risk Factors, 7 MD&A, 7A market risk.
# 10-Q items: part1item2 MD&A, part2item1a risk factor updates.
SECTIONS = {
    "10-K": {"business": "1", "risk": "1A", "mdna": "7", "market_risk": "7A"},
    "10-Q": {"business": "part1item2", "risk": "part2item1a", "mdna": "part1item2", "market_risk": "part1item3"},
}
ROUTES = [  # (words in the question, section) checked in order; MD&A is the default
    (("risk", "threat", "uncertain", "litigation", "regulat", "lawsuit"), "risk"),
    (("interest rate", "currency", "foreign exchange", "hedg", "market risk"), "market_risk"),
    (("product", "segment", "customer", "competit", "employee", "strategy", "business model", "what does"), "business"),
]
SECTION_NAMES = {"business": "Business", "risk": "Risk Factors", "mdna": "Management's Discussion and Analysis",
                 "market_risk": "Market Risk"}
STOP = set("the a an of to in and or for on is are was were be by with what how did does do its it this that from as at "
           "which who their they has have had last year quarter company".split())
MAX_CHARS = 6000


def client(api_key=None):
    key = api_key or os.environ.get("SECFEED_API_KEY")
    if not key:
        raise SecFeedError("Set SECFEED_API_KEY (free key at https://secfeedapi.com/signup/)")
    return Client(key)


def _section_for(question):
    q = question.lower()
    for words, section in ROUTES:
        if any(w in q for w in words):
            return section
    return "mdna"


def _terms(text):
    return {w for w in re.findall(r"[a-z][a-z0-9&\-]+", text.lower()) if w not in STOP and len(w) > 2}


def best_passages(text, question, max_chars=MAX_CHARS):
    """Paragraphs of `text` ranked by how many of the question's words they contain, kept in document order.
    Simple keyword ranking: no embeddings, no vector store, no extra API key."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if len(p.strip()) > 80]
    terms = _terms(question)
    if not paras:
        return text[:max_chars]
    scored = sorted(range(len(paras)), key=lambda i: -len(terms & _terms(paras[i])))
    keep, size = set(), 0
    for i in scored:
        if size + len(paras[i]) > max_chars and keep:
            continue
        keep.add(i)
        size += len(paras[i])
        if size >= max_chars:
            break
    return "\n\n".join(paras[i] for i in sorted(keep))


_FILINGS, _SECTIONS = {}, {}  # per-process caches: agents often ask several questions about the same filing


def latest_filing(sf, ticker, form):
    k = (ticker.upper(), form)
    if k not in _FILINGS:
        res = sf.filings(f'ticker:{ticker.upper()} AND formType:"{form}"', size=1)
        _FILINGS[k] = (res.get("filings") or [None])[0]
    return _FILINGS[k]


def _section_text(sf, url, item):
    if (url, item) not in _SECTIONS:
        _SECTIONS[(url, item)] = sf.extract(url, item)
    return _SECTIONS[(url, item)]


def search_filing(ticker, question, form="10-K", section=None, api_key=None):
    """Answer material for `question` from the company's latest 10-K or 10-Q: picks the right section (or `section`:
    business, risk, mdna, market_risk), then returns the most relevant paragraphs."""
    sf = client(api_key)
    form = form.upper()
    if form not in SECTIONS:
        return "form must be 10-K or 10-Q"
    f = latest_filing(sf, ticker, form)
    if not f:
        return f"No {form} found for {ticker}. Check the ticker."
    section = section if section in SECTIONS[form] else _section_for(question)
    try:
        text = _section_text(sf, f["linkToFilingDetails"], SECTIONS[form][section])
    except SecFeedError as e:
        return f"Could not read {SECTION_NAMES[section]} of {f['companyName']} {form} filed {f['filedAt']}: {e}"
    head = (f"{f['companyName']} ({ticker.upper()}) {form} filed {f['filedAt']}, section: {SECTION_NAMES[section]}\n"
            f"Source: {f['linkToFilingDetails']}\n\n")
    return head + best_passages(text, question)


def _money(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    for div, unit in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(v) >= div:
            return f"{v / div:,.2f}{unit}"
    return f"{v:,.2f}"


def financials(ticker, form="10-K", api_key=None):
    """Income statement, balance sheet and cash flow from the latest 10-K or 10-Q (XBRL), newest periods first."""
    sf = client(api_key)
    rows = sf.financial_statements(ticker.upper(), form_type=form.upper(), n=1)
    if not rows:
        return f"No XBRL financial statements found for {ticker}."
    data = rows[0]
    out = [f"{ticker.upper()} {form.upper()} financial statements (accession {data.get('accessionNo')}), USD unless noted"]
    for name, items in data.items():
        if not isinstance(items, dict):
            continue
        out.append(f"\n{name}")
        for concept, facts in list(items.items())[:25]:
            if not isinstance(facts, list):
                continue
            vals = []
            for x in [x for x in facts if not x.get("segment")][:2]:
                p = x.get("period") or {}
                when = p.get("endDate") or p.get("instant") or ""
                vals.append(f"{when}: {_money(x.get('value'))}")
            if vals:
                out.append(f"  {concept}: " + "; ".join(vals))
    return "\n".join(out)[:12000]


CODES = {"P": "open market buy", "S": "open market sale", "A": "grant", "M": "option exercise", "F": "tax withholding",
         "G": "gift"}


def insider_trades(ticker, code=None, limit=20, api_key=None):
    """Latest Form 4 insider transactions. code: P (buys), S (sales) or None for all."""
    sf = client(api_key)
    res = sf.insider_trades(ticker=ticker.upper(), code=code, size=min(int(limit), 50))
    tx = res.get("transactions") or []
    if not tx:
        return f"No insider transactions found for {ticker}."
    out = [f"{ticker.upper()} insider transactions ({res.get('total')} on file, latest {len(tx)}):"]
    for t in tx:
        o = (t.get("owners") or [{}])[0]
        role = o.get("officerTitle") or ("Director" if o.get("isDirector") else "10% owner" if o.get("isTenPercentOwner") else "")
        val = f" = ${_money(t['shares'] * t['price'])}" if t.get("shares") and t.get("price") else ""
        plan = ", 10b5-1 plan" if t.get("aff10b5one") else ""
        out.append(f"- {t.get('tx_date')} {o.get('name')} ({role}): {CODES.get(t.get('code'), t.get('code'))} "
                   f"{_money(t.get('shares'))} sh @ {t.get('price')}{val}{plan}")
    return "\n".join(out)


def recent_8k(ticker, limit=10, api_key=None):
    """Latest 8-K current reports with the items each one covers (earnings, deals, leadership, etc.)."""
    sf = client(api_key)
    f = latest_filing(sf, ticker, "8-K")
    if not f:
        return f"No 8-K found for {ticker}."
    res = sf.events_8k(cik=f["cik"], size=min(int(limit), 50))
    out = [f"{f['companyName']} ({ticker.upper()}) latest 8-K reports:"]
    for e in res.get("filings") or []:
        items = "; ".join(f"{i['item']} {i['title']}" for i in e.get("items") or [])
        out.append(f"- {e.get('filed')}: {items}  {e.get('url')}")
    return "\n".join(out)

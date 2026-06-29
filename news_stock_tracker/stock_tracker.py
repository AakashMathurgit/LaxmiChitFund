import time
import re
import os
import sys
import json
import requests
import xml.etree.ElementTree as ET
from datetime import datetime

import yaml  # type: ignore
from openai import OpenAI  # type: ignore

# Add LCF root and src to path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_LCF_ROOT = os.path.dirname(_SCRIPT_DIR)
_SRC_DIR = os.path.join(_LCF_ROOT, "src")
if _LCF_ROOT not in sys.path:
    sys.path.insert(0, _LCF_ROOT)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# Enable debug output
os.environ["LCF_DEBUG"] = "1"

# -------- CONFIG --------
UPDATE_INTERVAL = 600  # 10 minutes
OUTPUT_FILE = os.path.join(_SCRIPT_DIR, "stocks.txt")
NSE_SYMBOL_FILE = os.path.join(_SCRIPT_DIR, "nse_symbols.txt")
RESULTS_FILE = os.path.join(_SCRIPT_DIR, "analysis_results.json")
HISTORY_FILE = os.path.join(_SCRIPT_DIR, "analysis_history.jsonl")
NEWS_CACHE_FILE = os.path.join(_LCF_ROOT, "data", "news_cache_ind.jsonl")

RSS_FEEDS = [
    # Moneycontrol
    "https://www.moneycontrol.com/rss/marketreports.xml",
    "https://www.moneycontrol.com/rss/results.xml",

    # The Hindu
    "https://www.thehindu.com/business/markets/feeder/default.rss",

    # Economic Times
    "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",

    # Google News aggregators
    "https://news.google.com/rss/search?q=stock+market+india&hl=en-IN&gl=IN&ceid=IN:en",
    "https://news.google.com/rss/search?q=company+results+India+stocks&hl=en-IN&gl=IN&ceid=IN:en",
]

HEADERS = {"User-Agent": "strict-stock-tracker/1.0"}
TOKEN_REGEX = re.compile(r"\b[A-Z]{2,15}\b")
SIGNIFICANCE_CACHE = {}
MAX_CACHE_SIZE = 1000  # Prevent unbounded memory growth

HOT_NEWS_KEYWORDS = {
    "merger",
    "acquisition",
    "stake sale",
    "buyback",
    "dividend",
    "results",
    "profit",
    "loss",
    "guidance",
    "regulatory",
    "sebi",
    "downgrade",
    "upgrade",
    "fraud",
    "default",
    "bankruptcy",
    "order win",
    "contract",
    "investigation",
    "insolvency",
    "debt",
    "fii",
    "dii",
}
# ------------------------

def load_nse_symbols():
    with open(NSE_SYMBOL_FILE) as f:
        return set(line.strip() for line in f if line.strip())

def fetch_rss(url):
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    return r.text


CREDENTIALS_FILE = os.path.join(os.path.dirname(__file__), "..", "credentials.yaml")


def _load_credentials():
    """Load LLM credentials from credentials.yml, falling back to env vars."""
    endpoint = os.getenv("OPENAI_BASE_URL", "").strip()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("OPENAI_MODEL", "gpt-4.1")

    creds_path = os.path.normpath(CREDENTIALS_FILE)
    if os.path.exists(creds_path):
        with open(creds_path, "r", encoding="utf-8") as f:
            creds = yaml.safe_load(f) or {}
        llm_cfg = creds.get("llm", {})
        endpoint = llm_cfg.get("endpoint", endpoint)
        api_key = llm_cfg.get("api_key", api_key)
        model = llm_cfg.get("model", model)

    return endpoint, api_key, model


def create_llm_client():
    """Create an OpenAI client from credentials.yml (or env vars).

    Returns (client | None, model_name).
    """
    endpoint, api_key, model = _load_credentials()
    if not api_key:
        return None, model

    base_url = endpoint or None
    return OpenAI(base_url=base_url, api_key=api_key), model


def is_significant_news(title, description, llm_client, model_name):
    """Use LLM to classify if a news item is market-significant.

    Falls back to a keyword heuristic when LLM is not configured.
    """
    text = f"{title} {description}".strip()
    if not text:
        return False

    cache_key = text.lower()
    cached = SIGNIFICANCE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if llm_client:
        system_prompt = (
            "You are a strict Indian equity market news significance classifier. "
            "Return JSON only with keys: significant (boolean), confidence (0..1), reason (string). "
            "Mark significant=true only if the headline can plausibly move the stock price."
        )
        user_prompt = (
            "Classify this news item for price impact significance.\n"
            f"Title: {title}\n"
            f"Description: {description}"
        )

        try:
            response = llm_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                max_tokens=120,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or "{}"
            data = json.loads(raw)
            significant = bool(data.get("significant", False))
            SIGNIFICANCE_CACHE[cache_key] = significant
            return significant
        except Exception:
            # Fallback to keyword heuristics if LLM call fails.
            pass

    lowered = text.lower()
    significant = any(keyword in lowered for keyword in HOT_NEWS_KEYWORDS)
    # Limit cache size to prevent memory growth
    if len(SIGNIFICANCE_CACHE) >= MAX_CACHE_SIZE:
        # Remove oldest entries (first 100)
        keys_to_remove = list(SIGNIFICANCE_CACHE.keys())[:100]
        for k in keys_to_remove:
            del SIGNIFICANCE_CACHE[k]
    SIGNIFICANCE_CACHE[cache_key] = significant
    return significant


def extract_tickers(xml_text, valid_symbols, llm_client, model_name, feed_url=""):
    """Extract tickers and return (tickers_set, news_items_list)."""
    root = ET.fromstring(xml_text)
    found = set()
    news_items = []  # For caching

    for item in root.findall(".//item"):
        title = item.findtext("title", "")
        desc = item.findtext("description", "")
        link = item.findtext("link", "")
        pub_date = item.findtext("pubDate", "")
        text = f"{title} {desc}"

        significant = is_significant_news(title, desc, llm_client, model_name)
        if not significant:
            continue

        matched = set()
        for token in TOKEN_REGEX.findall(text):
            if token in valid_symbols:
                matched.add(token)
                found.add(token)

        if matched:
            for sym in matched:
                news_items.append({
                    "symbol": sym,
                    "headline": title,
                    "description": desc,
                    "source": "",
                    "url": link,
                    "pub_date": pub_date,
                    "fetched_at": datetime.now().isoformat(),
                    "significant": True,
                    "matched_symbols": sorted(matched),
                    "market": "IND",
                    "feed_url": feed_url,
                })

    return found, news_items

def update_stocks():
    valid_symbols = load_nse_symbols()
    all_stocks = set()
    all_news_items = []
    llm_client, model_name = create_llm_client()

    for feed in RSS_FEEDS:
        try:
            xml = fetch_rss(feed)
            tickers, news_items = extract_tickers(
                xml, valid_symbols, llm_client, model_name, feed_url=feed
            )
            all_stocks.update(tickers)
            all_news_items.extend(news_items)
            # Tag source from feed URL
            source_name = feed.split("//")[1].split("/")[0] if "//" in feed else feed[:30]
            for item in news_items:
                if not item.get("source"):
                    item["source"] = source_name
            print(f"  [{len(tickers):>2} tickers] {feed[:60]}")
        except Exception as e:
            print(f"  [error] {feed[:60]} — {e}")

    with open(OUTPUT_FILE, "w") as f:
        for stock in sorted(all_stocks):
            f.write(stock + "\n")

    # HTML scraper discovery — scrape news sites for additional tickers
    try:
        from src.main.controllers.html_news_scraper import NewsPageScraper, NewsScraperConfig
        config = NewsScraperConfig(market="IND", max_items_per_site=20)
        scraper = NewsPageScraper(config)
        scraped_items = scraper.scrape_all()
        scraper_tickers = set()
        scraper_news = []
        for item in scraped_items:
            text = f"{item.headline} {item.description}"
            for token in TOKEN_REGEX.findall(text):
                if token in valid_symbols:
                    scraper_tickers.add(token)
                    scraper_news.append({
                        "symbol": token,
                        "headline": item.headline,
                        "description": item.description,
                        "source": item.source,
                        "url": item.url,
                        "pub_date": item.pub_date,
                        "fetched_at": datetime.now().isoformat(),
                        "significant": True,
                        "matched_symbols": [token],
                        "market": "IND",
                        "feed_url": item.url,
                    })
        new_from_scraper = scraper_tickers - all_stocks
        all_stocks.update(scraper_tickers)
        all_news_items.extend(scraper_news)
        print(f"  [scraper] {len(scraper_tickers)} tickers ({len(new_from_scraper)} new) from {len(scraped_items)} headlines")
    except Exception as e:
        print(f"  [scraper] failed: {e}")

    # Rewrite stocks.txt with scraper additions
    with open(OUTPUT_FILE, "w") as f:
        for stock in sorted(all_stocks):
            f.write(stock + "\n")

    # NLP enrichment — FinBERT sentiment + NER ticker extraction
    nlp_tickers = set()
    try:
        from src.main.controllers.nlp_processor import NLPProcessor
        nlp = NLPProcessor()

        # Extract tickers via NER from all news items
        for item in all_news_items:
            text = f"{item.get('headline', '')} {item.get('description', '')}"
            found = nlp.extract_tickers(text, valid_symbols)
            for t in found:
                if t not in all_stocks:
                    nlp_tickers.add(t)
                    all_stocks.add(t)

        # Batch sentiment on all headlines
        headlines = [item.get("headline", "") for item in all_news_items if item.get("headline")]
        if headlines and nlp.finbert_available:
            sentiments = nlp.analyze_sentiment_batch(headlines)
            for i, item in enumerate(all_news_items):
                if i < len(sentiments):
                    item["sentiment_label"] = sentiments[i]["label"]
                    item["sentiment_score"] = sentiments[i]["score"]
            print(f"  [nlp] FinBERT scored {len(sentiments)} headlines | {len(nlp_tickers)} new NER tickers")
        else:
            print(f"  [nlp] NER found {len(nlp_tickers)} new tickers (FinBERT unavailable)")
    except Exception as e:
        print(f"  [nlp] enrichment failed: {e}")

    # Final rewrite with NLP-discovered tickers
    if nlp_tickers:
        with open(OUTPUT_FILE, "w") as f:
            for stock in sorted(all_stocks):
                f.write(stock + "\n")

    # Write to shared news cache (now with sentiment scores)
    try:
        from src.main.controllers.news_cache import NewsCache
        cache = NewsCache(NEWS_CACHE_FILE, ttl_hours=72)
        written = cache.write_raw(all_news_items)
        print(f"Updated {len(all_stocks)} stocks | Cached {written} news items")
    except Exception as e:
        print(f"Updated {len(all_stocks)} stocks | Cache write failed: {e}")

    return sorted(all_stocks)


def run_pipeline_analysis(symbols: list) -> list:
    """Run LCF pipeline analysis on the given symbols.
    
    Returns list of analysis results with decisions.
    """
    if not symbols:
        print("  No symbols to analyze")
        return []
    
    print(f"\n{'='*60}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Running LCF Pipeline for {len(symbols)} stocks...")
    print(f"{'='*60}")
    
    try:
        import yfinance as yf
        import uuid as uuid_mod
        
        from main.agents.interfaces.agent import AgentContext
        from main.agents.interfaces.signals import (
            AgentFeatureBundle, RegimeSignal, MarketRegime, VolatilityState
        )
        from main.agents.technical_agent import TechnicalAgent
        from main.agents.fundamental_agent import FundamentalAgent
        from main.agents.sentiment_agent import SentimentAgent
        from main.agents.event_agent import EventAgent
        from main.agents.judge_agent import JudgeAgent
        from main.agents.regime_detector_agent import RegimeDetectorAgent
        from main.agents.risk_manager_agent import RiskManagerAgent
        from main.agents.trade_planner_agent import TradePlannerAgent
        from main.controllers.rss_news_provider import RSSNewsProvider
        from main.controllers.data_provider import DataType
        
        # Initialize agents once
        tech_agent = TechnicalAgent()
        fund_agent = FundamentalAgent()
        sent_agent = SentimentAgent()
        evt_agent = EventAgent()
        judge_agent = JudgeAgent()
        regime_agent = RegimeDetectorAgent()
        risk_agent = RiskManagerAgent()
        trade_planner = TradePlannerAgent()
        
        # RSS provider for extra news
        rss_provider = RSSNewsProvider()
        
        # ============================================================
        # STEP 1: Detect Market Regime from Nifty50
        # ============================================================
        print("\n[Regime Detection] Fetching Nifty50 data...")
        try:
            nifty = yf.Ticker("^NSEI")
            nifty_hist = nifty.history(period="1y", interval="1d")
            nifty_ohlc = [
                {"open": float(r["Open"]), "high": float(r["High"]),
                 "low": float(r["Low"]), "close": float(r["Close"])}
                for _, r in nifty_hist.iterrows()
            ] if not nifty_hist.empty else []
            
            # Try to get India VIX
            try:
                vix = yf.Ticker("^INDIAVIX")
                vix_hist = vix.history(period="5d")
                vix_value = float(vix_hist["Close"].iloc[-1]) if not vix_hist.empty else None
            except Exception:
                vix_value = None
            
            # Run regime detection
            regime_ctx = AgentContext(
                run_id=str(uuid_mod.uuid4()),
                rules_version="1.0.0",
                input_data={},
                config={},
                llm=None
            )
            regime_result = regime_agent.run(
                regime_ctx, 
                index_ohlc=nifty_ohlc, 
                vix_value=vix_value
            )
            
            detected_regime = regime_result.payload.get("raw_signal")
            if not detected_regime:
                detected_regime = RegimeSignal(
                    market_regime=MarketRegime.SIDEWAYS,
                    volatility_state=VolatilityState.MODERATE,
                    regime_confidence=0.5
                )
            
            print(f"[Regime Detection] Market: {detected_regime.market_regime.value}, "
                  f"Volatility: {detected_regime.volatility_state.value}, "
                  f"Confidence: {detected_regime.regime_confidence:.1%}")
        except Exception as e:
            print(f"[Regime Detection] Error: {e}, using SIDEWAYS default")
            detected_regime = RegimeSignal(
                market_regime=MarketRegime.SIDEWAYS,
                volatility_state=VolatilityState.MODERATE,
                regime_confidence=0.5
            )
        
        results = []
        
        for symbol in symbols:
            try:
                print(f"\n--- Analyzing {symbol} ---")
                
                # Fetch data from Yahoo Finance
                ticker = yf.Ticker(symbol + ".NS")
                info = ticker.info or {}
                hist = ticker.history(period="2y", interval="1d")
                
                if hist.empty:
                    print(f"  [SKIP] No price data for {symbol}")
                    results.append({"symbol": symbol, "error": "No price data"})
                    continue
                
                # Build OHLCV
                ohlc_daily = [
                    {"open": float(r["Open"]), "high": float(r["High"]), 
                     "low": float(r["Low"]), "close": float(r["Close"])}
                    for _, r in hist.iterrows()
                ]
                volume_daily = [int(r["Volume"]) for _, r in hist.iterrows()]
                closes = [b["close"] for b in ohlc_daily]
                last_close = closes[-1]
                prev_close = closes[-2] if len(closes) > 1 else last_close
                week52_high = max(b["high"] for b in ohlc_daily[-252:]) if ohlc_daily else 0
                
                # Get Yahoo news (nested under 'content' key in newer yfinance)
                yf_news = []
                for n in (ticker.news or []):
                    content = n.get("content", n)  # fallback to top-level if no 'content'
                    provider = content.get("provider", {}) if isinstance(content, dict) else {}
                    yf_news.append({
                        "headline": content.get("title", "") if isinstance(content, dict) else "",
                        "summary": content.get("summary", "") if isinstance(content, dict) else "",
                        "date": content.get("pubDate", "") if isinstance(content, dict) else "",
                        "source": provider.get("displayName", "") if isinstance(provider, dict) else "",
                    })
                
                # Get RSS news for this symbol
                try:
                    rss_result = rss_provider.fetch([symbol])
                    rss_news = [
                        {"headline": n.headline or "", "summary": n.news_text or "",
                         "date": n.date, "source": n.source or "RSS"}
                        for n in rss_result.get_data(DataType.NEWS)
                    ]
                except Exception:
                    rss_news = []
                
                all_news = yf_news + rss_news
                
                # Calculate gap
                gap_pct = 0
                if len(ohlc_daily) >= 2:
                    gap_pct = (ohlc_daily[-1]["open"] - ohlc_daily[-2]["close"]) / ohlc_daily[-2]["close"]
                
                # Build input data
                input_data = {
                    "symbol": symbol,
                    "date": hist.index[-1].strftime("%Y-%m-%d"),
                    "ohlc_daily": ohlc_daily,
                    "volume_daily": volume_daily,
                    "latest_price": last_close,
                    "52_week_high": week52_high,
                    "pe_ratio": info.get("trailingPE"),
                    "forward_pe": info.get("forwardPE"),
                    "revenue_growth": info.get("revenueGrowth"),
                    "profit_margin": info.get("profitMargins"),
                    "debt_to_equity": info.get("debtToEquity"),
                    "roe": info.get("returnOnEquity"),
                    "news_articles": all_news,
                    "recent_price_change": (last_close - prev_close) / prev_close if prev_close and prev_close != 0 else 0,
                    "earnings_date": None,
                    "dividend_info": None,
                    "stock_split_info": None,
                    "recent_gap_data": {"gap_pct": gap_pct},
                    "major_news_flag": len(all_news) > 5,
                }
                
                ctx = AgentContext(
                    run_id=str(uuid_mod.uuid4()),
                    rules_version="1.0.0",
                    input_data=input_data,
                    config={},
                    llm=None
                )
                
                # Run agents
                tech_result = tech_agent.run(ctx)
                fund_result = fund_agent.run(ctx)
                sent_result = sent_agent.run(ctx)
                evt_result = evt_agent.run(ctx)
                
                # Build bundle for judge (using detected regime)
                bundle = AgentFeatureBundle(
                    symbol=symbol,
                    date=input_data["date"],
                    technical=tech_result.payload.get("raw_signal"),
                    fundamental=fund_result.payload.get("raw_signal"),
                    sentiment=sent_result.payload.get("raw_signal"),
                    event=evt_result.payload.get("raw_signal"),
                    regime=detected_regime,
                )
                
                judge_result = judge_agent.run(ctx, bundle=bundle)
                
                # Rate limiting: small delay between Yahoo Finance calls
                time.sleep(0.5)
                
                result = {
                    "symbol": symbol,
                    "decision": judge_result.payload.get("decision"),
                    "prob_up_5d": judge_result.payload.get("prob_up_5d", 0),
                    "expected_return_5d": judge_result.payload.get("expected_return_5d", 0),
                    "confidence": judge_result.payload.get("confidence", 0),
                    "price": last_close,
                    "news_count": len(all_news),
                }
                results.append(result)
                
            except Exception as e:
                print(f"  [ERROR] {symbol}: {e}")
                results.append({"symbol": symbol, "error": str(e)})
        
        # Print summary
        print(f"\n{'='*60}")
        print("ANALYSIS SUMMARY")
        print(f"{'='*60}")
        
        buy_signals = []
        sell_signals = []
        hold_signals = []
        
        for r in results:
            symbol = r.get("symbol", "N/A")
            decision = r.get("decision", "ERROR")
            prob_up = r.get("prob_up_5d", 0) * 100
            expected_ret = r.get("expected_return_5d", 0) * 100
            confidence = r.get("confidence", 0) * 100
            
            if "error" in r:
                print(f"  {symbol}: ERROR - {r['error'][:50]}")
                continue
            
            indicator = "[BUY]" if decision == "BUY" else "[SELL]" if decision == "SELL" else "[HOLD]"
            print(f"  {indicator:6} {symbol:10} | Prob Up: {prob_up:5.1f}% | Return: {expected_ret:+6.2f}% | Conf: {confidence:5.1f}%")
            
            if decision == "BUY":
                buy_signals.append(r)
            elif decision == "SELL":
                sell_signals.append(r)
            else:
                hold_signals.append(r)
        
        print(f"\n  Summary: {len(buy_signals)} BUY | {len(hold_signals)} HOLD | {len(sell_signals)} SELL")
        
        # ============================================================
        # STEP 3: Apply Risk Management to BUY signals
        # ============================================================
        if buy_signals:
            print(f"\n{'='*60}")
            print("RISK MANAGEMENT ADJUSTMENTS")
            print(f"{'='*60}")
            
            try:
                from main.agents.interfaces.signals import JudgeDecision
                
                # Build JudgeDecision objects for risk assessment
                judge_decisions = []
                for sig in buy_signals:
                    # Create a JudgeDecision-like object
                    jd = JudgeDecision(
                        symbol=sig["symbol"],
                        date=datetime.now().strftime("%Y-%m-%d"),
                        decision=sig["decision"],
                        prob_up_5d=sig.get("prob_up_5d", 0.5),
                        expected_return_5d=sig.get("expected_return_5d", 0),
                        downside_risk_prob=1 - sig.get("prob_up_5d", 0.5),
                        confidence=sig.get("confidence", 0.5),
                        position_size_pct=0.02,  # Default 2%
                        stop_loss_pct=-0.03,
                        take_profit_pct=0.06,
                    )
                    judge_decisions.append(jd)
                
                # Run risk assessment
                risk_result = risk_agent.assess_risk(
                    decisions=judge_decisions,
                    regime=detected_regime,
                    current_drawdown=0.0,  # Would need portfolio tracking
                )
                
                print(f"  Overall Risk Level: {risk_result.overall_risk_level.value}")
                print(f"  Total Exposure: {risk_result.total_exposure_pct:.1%}")
                print(f"  Correlation Risk: {risk_result.correlation_risk:.1%}")
                
                if risk_result.warnings:
                    for w in risk_result.warnings:
                        print(f"  [WARNING] {w}")
                
                print("\n  Adjusted Positions:")
                for pos in risk_result.positions:
                    if pos.blocked:
                        print(f"    {pos.symbol}: BLOCKED - {pos.block_reason}")
                    else:
                        print(f"    {pos.symbol}: {pos.adjusted_position_size:.2%} "
                              f"(Stop: {pos.stop_loss_pct:.1%}, TP: {pos.take_profit_pct:.1%}) "
                              f"Risk: {pos.risk_level.value}")
                        if pos.warnings:
                            for w in pos.warnings:
                                print(f"      - {w}")
            except Exception as e:
                print(f"  [Risk Manager Error] {e}")
        
        # ============================================================
        # STEP 4: Generate Trade Plans for BUY signals
        # ============================================================
        trade_plans = []
        if buy_signals:
            print(f"\n{'='*60}")
            print("TRADE PLANS")
            print(f"{'='*60}")
            
            try:
                from main.agents.interfaces.signals import JudgeDecision
                
                for sig in buy_signals:
                    symbol = sig["symbol"]
                    
                    # Get cached OHLC data for this symbol
                    try:
                        ticker = yf.Ticker(symbol + ".NS")
                        hist = ticker.history(period="2y", interval="1d")
                        ohlc_data = [
                            {"open": float(r["Open"]), "high": float(r["High"]), 
                             "low": float(r["Low"]), "close": float(r["Close"])}
                            for _, r in hist.iterrows()
                        ] if not hist.empty else []
                        current_price = float(hist["Close"].iloc[-1]) if not hist.empty else sig.get("price", 0)
                    except Exception:
                        ohlc_data = []
                        current_price = sig.get("price", 0)
                    
                    # Build JudgeDecision
                    jd = JudgeDecision(
                        symbol=symbol,
                        date=datetime.now().strftime("%Y-%m-%d"),
                        decision=sig["decision"],
                        prob_up_5d=sig.get("prob_up_5d", 0.5),
                        expected_return_5d=sig.get("expected_return_5d", 0),
                        downside_risk_prob=1 - sig.get("prob_up_5d", 0.5),
                        confidence=sig.get("confidence", 0.5),
                    )
                    
                    # Generate trade plan
                    plan = trade_planner.create_trade_plan(
                        decision=jd,
                        current_price=current_price,
                        ohlc=ohlc_data,
                        regime=detected_regime,
                    )
                    trade_plans.append(plan)
                    
                    print(f"\n  {symbol}:")
                    print(f"    Entry: {plan.entry_type.value.upper()} @ ₹{plan.entry_price:,.2f}")
                    print(f"    Stop Loss: ₹{plan.stop_loss_price:,.2f} ({(plan.entry_price - plan.stop_loss_price)/plan.entry_price*100:.1f}%)")
                    print(f"    Target: ₹{plan.target_price:,.2f} ({(plan.target_price - plan.entry_price)/plan.entry_price*100:.1f}%)")
                    print(f"    Risk:Reward = 1:{plan.risk_reward_ratio:.1f}")
                    print(f"    Position: {plan.position_size_pct:.1%} ({plan.suggested_shares} shares)")
                    print(f"    Max Loss: ₹{plan.max_loss_amount:,.2f}")
                    if plan.support_level:
                        print(f"    Support: ₹{plan.support_level:,.2f}")
                    if plan.resistance_level:
                        print(f"    Resistance: ₹{plan.resistance_level:,.2f}")
                    print(f"    Reasoning: {plan.reasoning}")
                    
                    time.sleep(0.3)  # Rate limiting
                    
            except Exception as e:
                print(f"  [Trade Planner Error] {e}")
                import traceback
                traceback.print_exc()
        
        # Save results to JSON (overwrite latest)
        run_output = {
            "timestamp": datetime.now().isoformat(),
            "market_regime": detected_regime.market_regime.value,
            "volatility_state": detected_regime.volatility_state.value,
            "regime_confidence": detected_regime.regime_confidence,
            "symbols_analyzed": len(symbols),
            "buy_count": len(buy_signals),
            "hold_count": len(hold_signals),
            "sell_count": len(sell_signals),
            "results": results,
            "trade_plans": [p.to_dict() for p in trade_plans] if trade_plans else [],
        }
        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(run_output, f, indent=2, default=str)

        # Append to history (one JSON line per cycle, never overwritten)
        with open(HISTORY_FILE, "a", encoding="utf-8") as hf:
            hf.write(json.dumps(run_output, default=str) + "\n")
        
        print(f"\n  Results saved to: {RESULTS_FILE}")
        print(f"  History appended to: {HISTORY_FILE}")
        
        return results
        
    except Exception as e:
        print(f"  Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return []


def main():
    print(f"\n{'#'*60}")
    print(f"# LCF News Stock Tracker + Pipeline Analyzer")
    print(f"# Update interval: {UPDATE_INTERVAL//60} minutes")
    print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}\n")
    
    while True:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Fetching RSS feeds...")
        
        # Step 1: Update stocks from RSS
        detected_symbols = update_stocks()
        
        # Step 2: Run pipeline analysis on detected stocks
        if detected_symbols:
            run_pipeline_analysis(detected_symbols)
        
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Sleeping for {UPDATE_INTERVAL//60} minutes...")
        print("-" * 60)
        time.sleep(UPDATE_INTERVAL)

if __name__ == "__main__":
    main()
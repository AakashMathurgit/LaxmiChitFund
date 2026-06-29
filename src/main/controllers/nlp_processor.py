"""NLP Processor for LCF — FinBERT sentiment + entity extraction.

Provides two core capabilities:
  1. FinBERT sentiment analysis (financial-domain BERT model)
  2. Entity extraction (spaCy NER when available, regex+DB fallback)

Both models are loaded lazily on first use and cached for reuse.
Gracefully degrades if dependencies are missing.

Usage:
    processor = NLPProcessor()

    # Sentiment
    result = processor.analyze_sentiment("Tesla beats Q3 earnings estimates")
    # → {"label": "positive", "score": 0.94}

    # Batch sentiment
    results = processor.analyze_sentiment_batch([
        "Tesla beats earnings",
        "Plug Power reports wider loss",
    ])

    # Entity extraction
    entities = processor.extract_entities("Plug Power signs deal with Amazon")
    # → [("Plug Power", "ORG"), ("Amazon", "ORG")]

    # Full pipeline: article → ticker + sentiment
    enriched = processor.process_article("Plug Power signs $1.2B hydrogen deal")
    # → {"entities": [...], "tickers": ["PLUG"], "sentiment": "positive", "confidence": 0.92}
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from ...utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Company name → ticker mapping (comprehensive)
# ---------------------------------------------------------------------------

COMPANY_TO_TICKER: Dict[str, str] = {
    # US — Mag 7 + major tech
    "apple": "AAPL", "microsoft": "MSFT", "alphabet": "GOOGL", "google": "GOOGL",
    "amazon": "AMZN", "meta platforms": "META", "meta": "META", "facebook": "META",
    "nvidia": "NVDA", "tesla": "TSLA", "netflix": "NFLX",
    "amd": "AMD", "advanced micro devices": "AMD", "intel": "INTC",
    "salesforce": "CRM", "oracle": "ORCL", "adobe": "ADBE",
    "paypal": "PYPL", "block": "SQ", "square": "SQ", "shopify": "SHOP",
    "snowflake": "SNOW", "palantir": "PLTR", "coinbase": "COIN",
    "uber": "UBER", "airbnb": "ABNB", "broadcom": "AVGO", "qualcomm": "QCOM",
    "arm holdings": "ARM", "arm": "ARM",
    "crowdstrike": "CRWD", "datadog": "DDOG", "cloudflare": "NET",
    "super micro": "SMCI", "supermicro": "SMCI",
    "plug power": "PLUG", "rivian": "RIVN", "lucid": "LCID", "nio": "NIO",
    "gamestop": "GME", "amc entertainment": "AMC", "sofi": "SOFI",
    "robinhood": "HOOD", "moderna": "MRNA", "biontech": "BNTX",
    # US — Finance
    "jpmorgan": "JPM", "jp morgan": "JPM", "goldman sachs": "GS",
    "morgan stanley": "MS", "bank of america": "BAC", "wells fargo": "WFC",
    "citigroup": "C", "visa": "V", "mastercard": "MA", "blackrock": "BLK",
    # US — Healthcare
    "johnson & johnson": "JNJ", "johnson and johnson": "JNJ",
    "unitedhealth": "UNH", "pfizer": "PFE", "abbvie": "ABBV",
    "merck": "MRK", "eli lilly": "LLY", "lilly": "LLY",
    # US — Energy / Industrial / Consumer
    "exxon": "XOM", "exxon mobil": "XOM", "chevron": "CVX",
    "walmart": "WMT", "costco": "COST", "home depot": "HD",
    "nike": "NKE", "starbucks": "SBUX", "mcdonalds": "MCD", "mcdonald's": "MCD",
    "coca-cola": "KO", "coca cola": "KO", "pepsi": "PEP", "pepsico": "PEP",
    "procter & gamble": "PG", "procter and gamble": "PG",
    "disney": "DIS", "boeing": "BA", "caterpillar": "CAT",
    "general electric": "GE",

    # India — major companies
    "reliance": "RELIANCE", "reliance industries": "RELIANCE",
    "tata consultancy": "TCS", "tcs": "TCS",
    "infosys": "INFY", "wipro": "WIPRO",
    "hdfc bank": "HDFCBANK", "icici bank": "ICICIBANK",
    "state bank": "SBIN", "sbi": "SBIN",
    "bharti airtel": "BHARTIARTL", "airtel": "BHARTIARTL",
    "hindustan unilever": "HINDUNILVR", "hul": "HINDUNILVR",
    "itc": "ITC", "kotak mahindra": "KOTAKBANK",
    "larsen & toubro": "LT", "l&t": "LT",
    "bajaj finance": "BAJFINANCE", "maruti suzuki": "MARUTI", "maruti": "MARUTI",
    "sun pharma": "SUNPHARMA", "titan": "TITAN",
    "hcl tech": "HCLTECH", "ntpc": "NTPC", "ongc": "ONGC",
    "adani enterprises": "ADANIENT", "adani ports": "ADANIPORTS",
    "tata motors": "TATAMOTORS", "tata steel": "TATASTEEL",
    "cipla": "CIPLA", "bharat electronics": "BEL", "bel": "BEL",
    "bharat heavy electricals": "BHEL", "bhel": "BHEL",
    "suzlon": "SUZLON", "suzlon energy": "SUZLON",
    "exide": "EXIDEIND", "exide industries": "EXIDEIND",
    "nucleus software": "NUCLEUS", "pine labs": "PINELABS",
    "thyrocare": "THYROCARE",
}

# Pre-compile patterns sorted by length (longest first to match "bank of america" before "bank")
_COMPANY_PATTERNS: List[Tuple[re.Pattern, str]] = []
for name, ticker in sorted(COMPANY_TO_TICKER.items(), key=lambda x: -len(x[0])):
    if len(name) > 2:  # Skip very short names
        _COMPANY_PATTERNS.append((
            re.compile(r"\b" + re.escape(name) + r"\b", re.IGNORECASE),
            ticker,
        ))


# ---------------------------------------------------------------------------
# FinBERT Singleton
# ---------------------------------------------------------------------------

class _FinBERTSingleton:
    """Lazy-loaded FinBERT model. Loaded once on first use."""

    _instance = None
    _pipeline = None
    _available = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._pipeline = None
        self._available = None

    @property
    def available(self) -> bool:
        if self._available is None:
            self._check_availability()
        return self._available

    def _check_availability(self):
        try:
            import transformers
            import torch
            self._available = True
        except ImportError:
            self._available = False
            logger.info("FinBERT unavailable: transformers or torch not installed")

    def _load(self):
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline as hf_pipeline
            logger.info("Loading FinBERT model (ProsusAI/finbert)...")
            self._pipeline = hf_pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
                top_k=None,  # Return all labels with scores
                truncation=True,
                max_length=512,
            )
            logger.info("FinBERT model loaded successfully")
        except Exception as e:
            logger.warning(f"FinBERT load failed: {e}")
            self._available = False

    def predict(self, text: str) -> Dict[str, Any]:
        """Predict sentiment for a single text.

        Returns: {"label": "positive", "score": 0.94, "all_scores": {...}}
        """
        if not self.available:
            return {"label": "neutral", "score": 0.5, "all_scores": {}}

        self._load()
        if self._pipeline is None:
            return {"label": "neutral", "score": 0.5, "all_scores": {}}

        try:
            results = self._pipeline(text[:512])
            # results is list of list of dicts: [[{label, score}, ...]]
            if results and isinstance(results[0], list):
                scores = {r["label"]: round(r["score"], 4) for r in results[0]}
            elif results:
                scores = {r["label"]: round(r["score"], 4) for r in results}
            else:
                return {"label": "neutral", "score": 0.5, "all_scores": {}}

            # Find top label
            top_label = max(scores, key=scores.get)
            return {
                "label": top_label,
                "score": scores[top_label],
                "all_scores": scores,
            }
        except Exception as e:
            logger.debug(f"FinBERT predict error: {e}")
            return {"label": "neutral", "score": 0.5, "all_scores": {}}

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Predict sentiment for multiple texts."""
        if not self.available or not texts:
            return [{"label": "neutral", "score": 0.5, "all_scores": {}} for _ in texts]

        self._load()
        if self._pipeline is None:
            return [{"label": "neutral", "score": 0.5, "all_scores": {}} for _ in texts]

        try:
            truncated = [t[:512] for t in texts]
            batch_results = self._pipeline(truncated, batch_size=8)

            output = []
            for results in batch_results:
                if isinstance(results, list):
                    scores = {r["label"]: round(r["score"], 4) for r in results}
                else:
                    scores = {results["label"]: round(results["score"], 4)}
                top_label = max(scores, key=scores.get)
                output.append({
                    "label": top_label,
                    "score": scores[top_label],
                    "all_scores": scores,
                })
            return output
        except Exception as e:
            logger.warning(f"FinBERT batch predict error: {e}")
            return [{"label": "neutral", "score": 0.5, "all_scores": {}} for _ in texts]


# ---------------------------------------------------------------------------
# spaCy Singleton (optional, degrades gracefully)
# ---------------------------------------------------------------------------

class _SpacySingleton:
    """Lazy-loaded spaCy model. Falls back to regex if unavailable."""

    _instance = None
    _nlp = None
    _available = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._nlp = None
        self._available = None

    @property
    def available(self) -> bool:
        if self._available is None:
            self._check_availability()
        return self._available

    def _check_availability(self):
        try:
            import spacy
            spacy.load("en_core_web_sm")
            self._available = True
        except Exception:
            self._available = False
            logger.info("spaCy unavailable, using regex-based entity extraction")

    def _load(self):
        if self._nlp is not None:
            return
        try:
            import spacy
            self._nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded (en_core_web_sm)")
        except Exception as e:
            logger.info(f"spaCy load failed ({e}), using regex fallback")
            self._available = False

    def extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """Extract named entities. Returns [(text, label), ...]."""
        if not self.available:
            return []
        self._load()
        if self._nlp is None:
            return []
        try:
            doc = self._nlp(text[:5000])
            return [(ent.text, ent.label_) for ent in doc.ents]
        except Exception:
            return []


# ---------------------------------------------------------------------------
# Main NLP Processor
# ---------------------------------------------------------------------------

class NLPProcessor:
    """Central NLP processor for LCF.

    Provides:
    - FinBERT-based financial sentiment analysis
    - Entity extraction (spaCy NER + regex company name matching)
    - Combined article processing pipeline
    """

    def __init__(self):
        self._finbert = _FinBERTSingleton.get()
        self._spacy = _SpacySingleton.get()

    # ------------------------------------------------------------------
    # Sentiment Analysis (FinBERT)
    # ------------------------------------------------------------------

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze financial sentiment of a single text.

        Returns:
            {"label": "positive"|"negative"|"neutral", "score": 0.0-1.0, "all_scores": {...}}
        """
        return self._finbert.predict(text)

    def analyze_sentiment_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Analyze financial sentiment of multiple texts."""
        return self._finbert.predict_batch(texts)

    @property
    def finbert_available(self) -> bool:
        return self._finbert.available

    # ------------------------------------------------------------------
    # Entity Extraction (spaCy + regex fallback)
    # ------------------------------------------------------------------

    def extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """Extract named entities from text.

        Uses spaCy NER if available, otherwise uses regex company name matching.
        Returns list of (entity_text, entity_label) tuples.
        """
        # Try spaCy first
        if self._spacy.available:
            return self._spacy.extract_entities(text)

        # Fallback: regex-based company name extraction
        return self._extract_entities_regex(text)

    def _extract_entities_regex(self, text: str) -> List[Tuple[str, str]]:
        """Regex-based entity extraction using company name database."""
        entities = []
        seen = set()
        for pattern, ticker in _COMPANY_PATTERNS:
            match = pattern.search(text)
            if match and match.group() not in seen:
                seen.add(match.group())
                entities.append((match.group(), "ORG"))
        return entities

    @property
    def spacy_available(self) -> bool:
        return self._spacy.available

    # ------------------------------------------------------------------
    # Ticker Extraction
    # ------------------------------------------------------------------

    def extract_tickers(
        self,
        text: str,
        valid_symbols: Optional[Set[str]] = None,
    ) -> List[str]:
        """Extract stock tickers from text using NER + company name matching.

        Combines:
        1. spaCy NER ORG entities → company name DB → ticker
        2. Regex company name patterns → ticker
        3. Direct uppercase ticker mentions

        Args:
            text: Article text
            valid_symbols: Optional set of valid tickers to filter against

        Returns:
            List of unique ticker symbols found.
        """
        tickers = set()

        # 1. Entity extraction → company name → ticker
        entities = self.extract_entities(text)
        for ent_text, ent_label in entities:
            if ent_label == "ORG":
                key = ent_text.lower().strip()
                if key in COMPANY_TO_TICKER:
                    tickers.add(COMPANY_TO_TICKER[key])

        # 2. Regex company patterns (catches names NER might miss)
        for pattern, ticker in _COMPANY_PATTERNS:
            if pattern.search(text):
                tickers.add(ticker)

        # 3. Direct uppercase ticker mentions
        for match in re.findall(r"\b[A-Z]{1,5}\b", text):
            if valid_symbols and match in valid_symbols:
                tickers.add(match)

        return sorted(tickers)

    # ------------------------------------------------------------------
    # Full Article Processing Pipeline
    # ------------------------------------------------------------------

    def process_article(
        self,
        text: str,
        valid_symbols: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """Full NLP pipeline for a single article.

        Extracts entities, maps to tickers, and analyzes sentiment.

        Returns:
            {
                "entities": [("Plug Power", "ORG"), ...],
                "tickers": ["PLUG"],
                "sentiment_label": "positive",
                "sentiment_score": 0.92,
                "sentiment_all": {"positive": 0.92, "negative": 0.03, "neutral": 0.05},
            }
        """
        entities = self.extract_entities(text)
        tickers = self.extract_tickers(text, valid_symbols)
        sentiment = self.analyze_sentiment(text)

        return {
            "entities": entities,
            "tickers": tickers,
            "sentiment_label": sentiment["label"],
            "sentiment_score": sentiment["score"],
            "sentiment_all": sentiment.get("all_scores", {}),
        }

    def process_articles_batch(
        self,
        texts: List[str],
        valid_symbols: Optional[Set[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Process multiple articles efficiently (batched FinBERT).

        Returns list of dicts, one per article.
        """
        # Extract entities + tickers for each (fast, CPU)
        all_entities = [self.extract_entities(t) for t in texts]
        all_tickers = [self.extract_tickers(t, valid_symbols) for t in texts]

        # Batch sentiment (GPU/CPU, most expensive)
        all_sentiment = self.analyze_sentiment_batch(texts)

        results = []
        for i, text in enumerate(texts):
            results.append({
                "entities": all_entities[i],
                "tickers": all_tickers[i],
                "sentiment_label": all_sentiment[i]["label"],
                "sentiment_score": all_sentiment[i]["score"],
                "sentiment_all": all_sentiment[i].get("all_scores", {}),
            })

        return results

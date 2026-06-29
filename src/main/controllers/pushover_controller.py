"""Pushover Notification Controller for LCF.

Sends real-time stock alerts to your phone via Pushover push notifications.
Much faster and more reliable than WhatsApp (no browser needed).

Setup:
    1. Install Pushover app on phone (iOS/Android)
    2. Create account at https://pushover.net
    3. Create application at https://pushover.net/apps/build
    4. Add credentials to credentials.yaml:
        pushover:
            api_token: "your_api_token"
            user_key: "your_user_key"

Usage:
    notifier = PushoverController()
    notifier.send_trade_alert(result)
    notifier.send_daily_summary(results, mode="adaptive")
    notifier.send_news_alert(symbol="NVDA", headline="...", sentiment="positive")
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests
import yaml

try:
    from ...utils.logger import get_logger
except ImportError:
    import logging
    def get_logger(name):
        return logging.getLogger(name)

logger = get_logger(__name__)

PUSHOVER_API_URL = "https://api.pushover.net/1/messages.json"

_CREDENTIALS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "credentials.yaml"
)


class PushoverController:
    """Sends stock alerts via Pushover push notifications."""

    def __init__(
        self,
        api_token: str = "",
        user_key: str = "",
        enabled: bool = True,
        min_confidence: float = 0.0,   # Only send alerts above this confidence
        send_hold: bool = False,        # Send HOLD signals too
        send_news: bool = True,         # Send significant news alerts
        market: str = "IND",            # IND or US (affects currency symbol)
    ):
        self._api_token = api_token
        self._user_key = user_key
        self._enabled = enabled
        self._min_confidence = min_confidence
        self._send_hold = send_hold
        self._send_news = send_news
        self._market = market
        self._currency = "$" if market == "US" else "Rs."

        # Load from credentials.yaml if not provided
        if not self._api_token or not self._user_key:
            self._load_credentials()

        if self._enabled and not (self._api_token and self._user_key):
            logger.warning("Pushover disabled: missing api_token or user_key in credentials.yaml")
            self._enabled = False

        if self._enabled:
            logger.info("PushoverController ready")

    def _load_credentials(self):
        """Load Pushover credentials from env vars (preferred) or credentials.yaml."""
        # Environment variables take precedence (used in cloud / containers).
        self._api_token = self._api_token or os.environ.get("PUSHOVER_TOKEN", "")
        self._user_key = self._user_key or os.environ.get("PUSHOVER_USER", "")
        if self._api_token and self._user_key:
            return

        creds_path = os.path.normpath(_CREDENTIALS_PATH)
        if not os.path.exists(creds_path):
            return
        try:
            with open(creds_path, "r", encoding="utf-8") as f:
                creds = yaml.safe_load(f) or {}
            pushover = creds.get("pushover", {})
            self._api_token = self._api_token or pushover.get("api_token", "")
            self._user_key = self._user_key or pushover.get("user_key", "")
        except Exception as e:
            logger.warning(f"Failed to load Pushover credentials: {e}")

    # ------------------------------------------------------------------
    # Core send
    # ------------------------------------------------------------------

    def _send(
        self,
        message: str,
        title: str = "LCF Stock Alert",
        priority: int = 0,
        url: str = "",
        url_title: str = "",
        sound: str = "",
        html: bool = True,
    ) -> bool:
        """Send a push notification via Pushover API.

        Args:
            message: Message body (supports HTML if html=True)
            title: Notification title
            priority: -2 (silent) to 2 (emergency)
            url: Optional URL to include
            url_title: Display text for the URL
            sound: Notification sound name
            html: Enable HTML formatting

        Returns:
            True if sent successfully.
        """
        if not self._enabled:
            logger.debug(f"Pushover disabled, would send: {title}")
            return False

        data = {
            "token": self._api_token,
            "user": self._user_key,
            "message": message,
            "title": title,
            "html": "1" if html else "0",
        }
        if priority:
            data["priority"] = str(priority)
            if priority == 2:
                data["retry"] = "60"
                data["expire"] = "3600"
        if url:
            data["url"] = url
        if url_title:
            data["url_title"] = url_title
        if sound:
            data["sound"] = sound

        try:
            resp = requests.post(PUSHOVER_API_URL, data=data, timeout=10)
            if resp.status_code == 200:
                logger.info(f"Pushover sent: {title}")
                return True
            else:
                logger.warning(f"Pushover error {resp.status_code}: {resp.text}")
                return False
        except Exception as e:
            logger.warning(f"Pushover send failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Trade Alerts (full detail — same as WhatsApp)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_raw_field(raw_signal, field: str) -> Optional[str]:
        """Extract a field from a raw_signal string repr or object."""
        if raw_signal is None:
            return None
        if hasattr(raw_signal, field):
            val = getattr(raw_signal, field)
            return str(val) if val is not None else None
        if isinstance(raw_signal, str):
            import re
            match = re.search(rf"{field}=([^,)]+)", raw_signal)
            return match.group(1).strip("'\"") if match else None
        return None

    def send_trade_alert(self, result: Dict[str, Any]) -> bool:
        """Send full trade alert for a pipeline analysis result."""
        symbol = result.get("symbol", "?")
        date = result.get("date", "")

        jd = result.get("judge_decision", {})
        payload = jd.get("payload", jd)
        decision = payload.get("decision", "?")
        confidence = payload.get("confidence", 0)
        prob_up = payload.get("prob_up_5d", 0)
        expected_return = payload.get("expected_return_5d", 0)
        downside_risk = payload.get("downside_risk_prob", 0)

        if decision == "HOLD" and not self._send_hold:
            return False
        if confidence < self._min_confidence:
            return False
        if "error" in result:
            return False

        c = self._currency
        tp = result.get("trade_plan") or {}
        current_price = tp.get("current_price") or tp.get("entry_price", 0)

        if decision == "BUY":
            emoji = "🟢"
            priority = 1
            sound = "cashregister"
        elif decision == "SELL":
            emoji = "🔴"
            priority = 1
            sound = "falling"
        else:
            emoji = "🟡"
            priority = 0
            sound = ""

        lines = []

        # ---- Header ----
        lines.append(f"{emoji} <b>{decision} SIGNAL</b>")
        lines.append(f"<b>Stock: {symbol}</b>  |  {date}")
        lines.append("")

        # ---- Price & Key Metrics ----
        lines.append(f"<b>Price:</b> {c}{current_price:,.2f}")
        lines.append(f"<b>Confidence:</b> {confidence*100:.0f}%")
        lines.append(f"<b>Prob Up 5d:</b> {prob_up*100:.0f}%")
        lines.append(f"<b>Expected Return:</b> {expected_return*100:+.2f}%")
        lines.append(f"<b>Downside Risk:</b> {downside_risk*100:.0f}%")

        # ---- Trade Plan (BUY) ----
        if tp and decision == "BUY":
            lines.append("")
            lines.append("<b>━━ TRADE PLAN ━━</b>")
            entry_type = (tp.get("entry_type") or "market").upper()
            lines.append(f"Entry: {entry_type} @ {c}{tp.get('entry_price', 0):,.2f}")
            sl = tp.get("stop_loss_price", 0)
            target = tp.get("target_price", 0)
            if sl > 0:
                sl_pct = ((sl - current_price) / current_price * 100) if current_price else 0
                lines.append(f"Stop Loss: {c}{sl:,.2f} ({sl_pct:+.1f}%)")
            if target > 0:
                tgt_pct = ((target - current_price) / current_price * 100) if current_price else 0
                lines.append(f"Target: {c}{target:,.2f} ({tgt_pct:+.1f}%)")
            rr = tp.get("risk_reward_ratio", 0)
            if rr > 0:
                lines.append(f"Risk:Reward = 1:{rr:.1f}")
            hold_days = tp.get("expected_holding_days", 0)
            if hold_days > 0:
                lines.append(f"Hold: {hold_days} days")
            shares = tp.get("suggested_shares", 0)
            pos_pct = tp.get("position_size_pct", 0)
            if shares > 0:
                lines.append(f"Position: {pos_pct*100:.1f}% ({shares} shares)")
            if tp.get("support_level"):
                lines.append(f"Support: {c}{tp['support_level']:,.2f}")
            if tp.get("resistance_level"):
                lines.append(f"Resistance: {c}{tp['resistance_level']:,.2f}")
            if tp.get("trailing_stop_pct"):
                lines.append(f"Trailing Stop: {tp['trailing_stop_pct']*100:.1f}%")

        # ---- Trade Plan (SELL) ----
        if tp and decision == "SELL":
            lines.append("")
            lines.append("<b>━━ ACTION ━━</b>")
            lines.append(f"EXIT @ {c}{current_price:,.2f}")
            lines.append(f"Expected drop: {expected_return*100:+.2f}%")

        # ---- Technicals ----
        tech = result.get("technical", {})
        tech_payload = tech.get("payload", {}) if isinstance(tech, dict) else {}
        tech_sig = tech_payload.get("signal", {})
        tech_raw = tech_payload.get("raw_signal", "")
        if tech_sig:
            lines.append("")
            lines.append("<b>━━ TECHNICALS ━━</b>")
            rsi_val = tech_sig.get("tech_rsi", 0)
            rsi_label = "Oversold" if rsi_val < 0.35 else ("Overbought" if rsi_val > 0.7 else "Neutral")
            lines.append(f"RSI: {rsi_val*100:.0f} ({rsi_label})")
            macd_val = tech_sig.get("tech_macd", 0)
            lines.append(f"MACD: {'BUY' if macd_val > 0.5 else 'SELL'}")
            trend_val = tech_sig.get("tech_trend", 0)
            lines.append(f"Trend: {'Bullish 📈' if trend_val > 0.5 else 'Bearish 📉'}")
            breakout = tech_sig.get("tech_breakout", 0)
            if breakout > 0.5:
                lines.append("⚡ BREAKOUT detected!")
            vol = tech_sig.get("tech_volatility", 0)
            lines.append(f"Volatility: {vol*100:.1f}%")
            support = self._extract_raw_field(tech_raw, "support_level")
            resistance = self._extract_raw_field(tech_raw, "resistance_level")
            if support and resistance:
                lines.append(f"Support: {c}{float(support):,.2f} | Resistance: {c}{float(resistance):,.2f}")
            lines.append(f"Tech Score: {tech_sig.get('tech_score', 0)*100:.0f}/100")

        # ---- Fundamentals ----
        fund = result.get("fundamental", {})
        fund_payload = fund.get("payload", {}) if isinstance(fund, dict) else {}
        fund_sig = fund_payload.get("signal", {})
        fund_raw = fund_payload.get("raw_signal", "")
        if fund_sig:
            lines.append("")
            lines.append("<b>━━ FUNDAMENTALS ━━</b>")
            pe = self._extract_raw_field(fund_raw, "pe_ratio")
            fwd_pe = self._extract_raw_field(fund_raw, "forward_pe")
            val_label = self._extract_raw_field(fund_raw, "valuation_label")
            if pe:
                lines.append(f"PE: {float(pe):.1f}" + (f" (Fwd: {float(fwd_pe):.1f})" if fwd_pe else ""))
            if val_label:
                lines.append(f"Valuation: {val_label.upper()}")
            growth = fund_sig.get("fund_growth", 0)
            health = fund_sig.get("fund_health", 0)
            lines.append(f"Growth: {growth*100:.0f}/100 | Health: {health*100:.0f}/100")
            lines.append(f"Fund Score: {fund_sig.get('fund_score', 0)*100:.0f}/100")

        # ---- Sentiment ----
        sent = result.get("sentiment", {})
        sent_payload = sent.get("payload", {}) if isinstance(sent, dict) else {}
        sent_sig = sent_payload.get("signal", {})
        sent_raw = sent_payload.get("raw_signal", "")
        if sent_sig and sent_sig.get("sent_confidence", 0) > 0:
            lines.append("")
            lines.append("<b>━━ NEWS SENTIMENT ━━</b>")
            pos = self._extract_raw_field(sent_raw, "positive_news_count")
            neg = self._extract_raw_field(sent_raw, "negative_news_count")
            trend = self._extract_raw_field(sent_raw, "sentiment_trend")
            if pos and neg:
                lines.append(f"Positive: {pos} | Negative: {neg}")
            if trend:
                lines.append(f"Trend: {trend}")
            lines.append(f"Sentiment: {sent_sig.get('sent_score', 0.5)*100:.0f}/100")

        # ---- Market Regime ----
        regime = result.get("regime", {})
        if regime:
            lines.append("")
            regime_name = (regime.get("regime") or "?").replace("_", " ").title()
            vol_state = (regime.get("vol_state") or "?").title()
            reg_conf = regime.get("regime_confidence", 0)
            lines.append(f"<b>Regime:</b> {regime_name} | Vol: {vol_state} ({reg_conf*100:.0f}%)")

        # ---- AI Debate ----
        debate = result.get("debate")
        if debate:
            bull = debate.get("bull", {})
            bear = debate.get("bear", {})
            dd = debate.get("debate_decision", {})
            lines.append("")
            lines.append("<b>━━ AI DEBATE ━━</b>")
            lines.append(f"Bull ({bull.get('confidence', 0)*100:.0f}%) vs Bear ({bear.get('confidence', 0)*100:.0f}%)")
            winner = (dd.get("winning_side") or "?").title()
            lines.append(f"Winner: {winner}")
            for pt in bull.get("key_points", [])[:2]:
                lines.append(f"  + {pt[:80]}")
            for pt in bear.get("key_points", [])[:2]:
                lines.append(f"  - {pt[:80]}")

        # ---- Hybrid Final Verdict ----
        hybrid = result.get("hybrid_decision")
        if hybrid:
            lines.append("")
            lines.append("<b>━━ FINAL VERDICT ━━</b>")
            lines.append(f"Rules: {hybrid.get('rule_decision', '?')} ({hybrid.get('rule_confidence', 0)*100:.0f}%)")
            lines.append(f"Debate: {hybrid.get('debate_decision', '?')} ({hybrid.get('debate_confidence', 0)*100:.0f}%)")
            agreement = "✅ Agreed" if hybrid.get("agreement") else "⚠️ Disagreed"
            lines.append(f"Agreement: {agreement}")
            lines.append(f"<b>FINAL: {hybrid.get('final_decision', '?')} ({hybrid.get('final_confidence', 0)*100:.0f}%)</b>")

        # ---- Risk ----
        risk = result.get("risk_assessment")
        if risk:
            risk_level = (risk.get("overall_risk_level") or "?").upper()
            lines.append(f"Risk: {risk_level}")
            if result.get("trade_blocked"):
                lines.append(f"🚫 BLOCKED: {result.get('block_reason', '?')}")

        # ---- Footer ----
        lines.append("")
        lines.append(f"<i>LCF Multi-Agent | {datetime.now().strftime('%H:%M:%S')}</i>")

        title = f"{emoji} {decision} {symbol} ({confidence*100:.0f}%)"
        message = "\n".join(lines)

        return self._send(
            message=message,
            title=title,
            priority=priority,
            sound=sound,
        )
    # ------------------------------------------------------------------
    # News Alerts (from NLP pipeline)
    # ------------------------------------------------------------------

    def send_news_alert(
        self,
        symbol: str,
        headline: str,
        sentiment: str = "",
        confidence: float = 0.0,
        source: str = "",
        url: str = "",
    ) -> bool:
        """Send a significant news alert for a stock.

        Args:
            symbol: Stock ticker
            headline: News headline
            sentiment: "positive", "negative", "neutral"
            confidence: FinBERT confidence score
            source: News source name
            url: Link to the article
        """
        if not self._send_news:
            return False

        emoji_map = {"positive": "📈", "negative": "📉", "neutral": "📰"}
        emoji = emoji_map.get(sentiment, "📰")

        lines = [
            f"{emoji} <b>{symbol}</b> — {sentiment.upper()} ({confidence*100:.0f}%)",
            f"",
            f"<i>{headline}</i>",
        ]
        if source:
            lines.append(f"Source: {source}")

        title = f"{emoji} {symbol} News: {sentiment}"
        message = "\n".join(lines)

        return self._send(
            message=message,
            title=title,
            priority=0,
            url=url,
            url_title="Read article",
        )

    # ------------------------------------------------------------------
    # Daily Summary
    # ------------------------------------------------------------------

    def send_daily_summary(
        self,
        results: List[Dict[str, Any]],
        mode: str = "adaptive",
        market: str = "",
    ) -> bool:
        """Send a daily summary of all analyzed stocks."""
        if not results:
            return False

        market_label = market or self._market
        buy_count = 0
        sell_count = 0
        hold_count = 0
        buy_symbols = []
        sell_symbols = []

        for r in results:
            if "error" in r:
                continue
            jd = r.get("judge_decision", {}).get("payload", {})
            decision = jd.get("decision", "?")
            symbol = r.get("symbol", "?")
            if decision == "BUY":
                buy_count += 1
                buy_symbols.append(symbol)
            elif decision == "SELL":
                sell_count += 1
                sell_symbols.append(symbol)
            else:
                hold_count += 1

        lines = [
            f"📊 <b>{market_label} Daily Summary</b>",
            f"Mode: {mode.upper()}",
            f"Stocks analyzed: {len(results)}",
            f"",
            f"🟢 BUY:  {buy_count}  {'(' + ', '.join(buy_symbols) + ')' if buy_symbols else ''}",
            f"🟡 HOLD: {hold_count}",
            f"🔴 SELL: {sell_count}  {'(' + ', '.join(sell_symbols[:5]) + ')' if sell_symbols else ''}",
            f"",
            f"Time: {datetime.now().strftime('%H:%M:%S')}",
        ]

        return self._send(
            message="\n".join(lines),
            title=f"📊 {market_label} Summary: {buy_count}B/{hold_count}H/{sell_count}S",
            priority=0,
        )

    # ------------------------------------------------------------------
    # Portfolio Alerts
    # ------------------------------------------------------------------

    def send_stop_loss_alert(self, symbol: str, entry_price: float, exit_price: float, pnl: float) -> bool:
        """Alert when a stop loss is triggered."""
        c = self._currency
        pnl_pct = ((exit_price - entry_price) / entry_price * 100) if entry_price else 0
        return self._send(
            message=(
                f"🚨 <b>STOP LOSS TRIGGERED</b>\n\n"
                f"<b>{symbol}</b>\n"
                f"Entry: {c}{entry_price:,.2f}\n"
                f"Exit: {c}{exit_price:,.2f}\n"
                f"P&L: {c}{pnl:+,.2f} ({pnl_pct:+.1f}%)"
            ),
            title=f"🚨 SL Hit: {symbol} ({pnl_pct:+.1f}%)",
            priority=1,
            sound="siren",
        )

    def send_target_hit_alert(self, symbol: str, entry_price: float, exit_price: float, pnl: float) -> bool:
        """Alert when a profit target is hit."""
        c = self._currency
        pnl_pct = ((exit_price - entry_price) / entry_price * 100) if entry_price else 0
        return self._send(
            message=(
                f"🎯 <b>TARGET HIT!</b>\n\n"
                f"<b>{symbol}</b>\n"
                f"Entry: {c}{entry_price:,.2f}\n"
                f"Exit: {c}{exit_price:,.2f}\n"
                f"P&L: {c}{pnl:+,.2f} ({pnl_pct:+.1f}%)"
            ),
            title=f"🎯 Target: {symbol} (+{pnl_pct:.1f}%)",
            priority=0,
            sound="cashregister",
        )

    # ------------------------------------------------------------------
    # Quick test
    # ------------------------------------------------------------------

    def send_test(self) -> bool:
        """Send a test notification to verify setup."""
        return self._send(
            message="✅ LCF Pushover integration is working!\n\nYou will receive stock alerts here.",
            title="✅ LCF Test Notification",
            priority=0,
        )

    # ------------------------------------------------------------------
    # Batch alerts (all stocks in one message)
    # ------------------------------------------------------------------

    def send_batch_trade_alerts(
        self,
        results: List[Dict[str, Any]],
        mode: str = "adaptive",
        market: str = "",
    ) -> bool:
        """Send ALL stock analysis results as ONE combined push notification.

        Collects all BUY/SELL/HOLD results and sends a single rich message
        with a summary header + per-stock detail sections.
        """
        if not results:
            return False

        market_label = market or self._market
        c = self._currency
        buy_count = 0
        sell_count = 0
        hold_count = 0
        sections = []

        for r in results:
            symbol = r.get("symbol", "?")
            if "error" in r:
                sections.append(f"❌ <b>{symbol}</b>: ERROR — {str(r.get('error', ''))[:40]}")
                continue

            jd = r.get("judge_decision", {})
            payload = jd.get("payload", jd)
            decision = payload.get("decision", "?")
            confidence = payload.get("confidence", 0)
            prob_up = payload.get("prob_up_5d", 0)
            expected_return = payload.get("expected_return_5d", 0)
            downside_risk = payload.get("downside_risk_prob", 0)

            if decision == "BUY": buy_count += 1
            elif decision == "SELL": sell_count += 1
            else: hold_count += 1

            tp = r.get("trade_plan") or {}
            current_price = tp.get("current_price") or tp.get("entry_price", 0)

            emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(decision, "⚪")

            # Stock header
            lines = [f"\n{emoji} <b>{decision} — {symbol}</b> ({confidence*100:.0f}%)"]
            lines.append(f"  Price: {c}{current_price:,.2f} | Prob Up: {prob_up*100:.0f}% | Exp: {expected_return*100:+.1f}%")

            # Trade plan for BUY
            if decision == "BUY" and tp:
                sl = tp.get("stop_loss_price", 0)
                target = tp.get("target_price", 0)
                rr = tp.get("risk_reward_ratio", 0)
                parts = []
                if sl > 0:
                    parts.append(f"SL: {c}{sl:,.2f}")
                if target > 0:
                    parts.append(f"Tgt: {c}{target:,.2f}")
                if rr > 0:
                    parts.append(f"R:R=1:{rr:.1f}")
                if parts:
                    lines.append(f"  {' | '.join(parts)}")
                shares = tp.get("suggested_shares", 0)
                if shares > 0:
                    lines.append(f"  {shares} shares ({tp.get('position_size_pct', 0)*100:.1f}%)")

            # Technicals (compact)
            tech = r.get("technical", {})
            tech_sig = (tech.get("payload", {}) if isinstance(tech, dict) else {}).get("signal", {})
            if tech_sig:
                rsi = tech_sig.get("tech_rsi", 0)
                rsi_lbl = "OS" if rsi < 0.35 else ("OB" if rsi > 0.7 else "")
                macd = "Buy" if tech_sig.get("tech_macd", 0) > 0.5 else "Sell"
                trend = "📈" if tech_sig.get("tech_trend", 0) > 0.5 else "📉"
                lines.append(f"  Tech: RSI {rsi*100:.0f}{rsi_lbl} | MACD {macd} | {trend}")

            # Fundamentals (compact)
            fund = r.get("fundamental", {})
            fund_sig = (fund.get("payload", {}) if isinstance(fund, dict) else {}).get("signal", {})
            fund_raw = (fund.get("payload", {}) if isinstance(fund, dict) else {}).get("raw_signal", "")
            if fund_sig:
                pe = self._extract_raw_field(fund_raw, "pe_ratio")
                pe_str = f"PE:{float(pe):.1f}" if pe else ""
                growth = fund_sig.get("fund_growth", 0)
                lines.append(f"  Fund: {pe_str} | Growth {growth*100:.0f} | Score {fund_sig.get('fund_score', 0)*100:.0f}")

            # Regime
            regime = r.get("regime", {})
            if regime:
                lines.append(f"  Regime: {regime.get('regime', '?')} | Vol: {regime.get('vol_state', '?')}")

            # Debate (ultra-compact)
            debate = r.get("debate")
            if debate:
                bull = debate.get("bull", {})
                bear = debate.get("bear", {})
                dd = debate.get("debate_decision", {})
                winner = (dd.get("winning_side") or "?").title()
                lines.append(f"  Debate: Bull {bull.get('confidence', 0)*100:.0f}% vs Bear {bear.get('confidence', 0)*100:.0f}% → {winner}")

            # Hybrid verdict
            hybrid = r.get("hybrid_decision")
            if hybrid:
                agreement = "✅" if hybrid.get("agreement") else "⚠️"
                lines.append(f"  Final: {hybrid.get('final_decision', '?')} ({hybrid.get('final_confidence', 0)*100:.0f}%) {agreement}")

            # Risk
            risk = r.get("risk_assessment")
            if risk:
                risk_level = (risk.get("overall_risk_level") or "?").upper()
                if r.get("trade_blocked"):
                    lines.append(f"  🚫 BLOCKED: {r.get('block_reason', '?')}")
                else:
                    lines.append(f"  Risk: {risk_level}")

            sections.append("\n".join(lines))

        # Build final combined message
        header = [
            f"📊 <b>{market_label} Analysis Report</b>",
            f"Mode: {mode.upper()} | {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"Stocks: {len(results)} | 🟢{buy_count} BUY  🟡{hold_count} HOLD  🔴{sell_count} SELL",
        ]

        message = "\n".join(header) + "\n" + "\n".join(sections)
        message += f"\n\n<i>LCF Multi-Agent System</i>"

        # Determine priority
        priority = 1 if buy_count > 0 else 0
        sound = "cashregister" if buy_count > 0 else ""

        title = f"📊 {market_label}: {buy_count}🟢 {hold_count}🟡 {sell_count}🔴 ({len(results)} stocks)"

        return self._send(
            message=message,
            title=title,
            priority=priority,
            sound=sound,
        )

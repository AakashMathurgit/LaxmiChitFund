"""Message Controller — sends trade alerts via WhatsApp.

Sends formatted trade signals when the pipeline produces BUY/SELL decisions.
Uses pywhatkit for WhatsApp Web automation.

Setup:
    pip install pywhatkit
    - WhatsApp Web must be logged in on your default browser
    - First message may take ~15 seconds (browser opens)

Usage:
    messenger = MessageController(phone="+919876543210")
    messenger.send_trade_alert(result)
    messenger.send_daily_summary(results)
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from ...utils.logger import get_logger
except ImportError:
    import logging
    def get_logger(name):
        return logging.getLogger(name)

logger = get_logger(__name__)


class MessageController:
    """Sends trade alerts to WhatsApp via pywhatkit."""

    def __init__(
        self,
        phone: str = "",
        enabled: bool = True,
        send_hold: bool = False,
    ):
        """Initialize message controller.

        Args:
            phone: WhatsApp number with country code (e.g., "+919876543210")
            enabled: Set False to disable sending (just logs messages)
            send_hold: If True, also sends HOLD decisions (default: only BUY/SELL)
        """
        self._phone = phone
        self._enabled = enabled and bool(phone)
        self._send_hold = send_hold
        self._pywhatkit_available = False

        if self._enabled:
            try:
                import pywhatkit
                self._pywhatkit_available = True
                logger.info(f"MessageController ready: WhatsApp alerts to {phone}")
            except ImportError:
                logger.warning("pywhatkit not installed. Run: pip install pywhatkit")
                self._pywhatkit_available = False

    # ------------------------------------------------------------------
    # Message formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_raw_signal_field(raw_signal, field: str) -> Optional[str]:
        """Extract a field value from a raw_signal (string repr or object)."""
        if raw_signal is None:
            return None
        # If it's an object with the attribute, get it directly
        if hasattr(raw_signal, field):
            val = getattr(raw_signal, field)
            return str(val) if val is not None else None
        # If it's a string representation, parse it
        if isinstance(raw_signal, str):
            import re
            match = re.search(rf"{field}=([^,)]+)", raw_signal)
            return match.group(1).strip("'\"") if match else None
        return None

    @staticmethod
    def format_trade_alert(result: Dict[str, Any]) -> str:
        """Format a single stock analysis result into a rich WhatsApp message."""
        symbol = result.get("symbol", "?")
        date = result.get("date", "")

        # Judge decision
        jd = result.get("judge_decision", {})
        payload = jd.get("payload", jd)
        decision = payload.get("decision", "?")
        prob_up = payload.get("prob_up_5d", 0)
        confidence = payload.get("confidence", 0)
        expected_return = payload.get("expected_return_5d", 0)
        downside_risk = payload.get("downside_risk_prob", 0)

        # Skip HOLD unless configured
        if decision == "HOLD":
            return ""

        lines = []

        # ---- Header ----
        if decision == "BUY":
            lines.append(">>> *BUY SIGNAL* <<<")
        else:
            lines.append(">>> *SELL SIGNAL* <<<")
        lines.append(f"*Stock: {symbol}*")
        lines.append(f"Date: {date}")
        lines.append("")

        # ---- Price & Key Metrics ----
        # Extract current price from trade_plan or technical raw_signal
        tp = result.get("trade_plan") or {}
        current_price = tp.get("current_price") or tp.get("entry_price", 0)
        lines.append(f"*Current Price*: Rs.{current_price:,.2f}")
        lines.append(f"*Confidence*: {confidence*100:.0f}%")
        lines.append(f"*Prob Up (5 days)*: {prob_up*100:.0f}%")
        lines.append(f"*Expected Return*: {expected_return*100:+.2f}%")
        lines.append(f"*Downside Risk*: {downside_risk*100:.0f}%")

        # ---- Trade Plan (BUY) ----
        if tp and decision == "BUY":
            lines.append("")
            lines.append("*--- TRADE PLAN ---*")
            entry_type = (tp.get("entry_type") or "market").upper()
            lines.append(f"  Entry: {entry_type} ORDER @ Rs.{tp.get('entry_price', 0):,.2f}")

            sl = tp.get("stop_loss_price", 0)
            target = tp.get("target_price", 0)
            if sl > 0:
                sl_pct = ((sl - current_price) / current_price * 100) if current_price else 0
                lines.append(f"  Stop Loss: Rs.{sl:,.2f} ({sl_pct:+.1f}%)")
            if target > 0:
                tgt_pct = ((target - current_price) / current_price * 100) if current_price else 0
                lines.append(f"  Target: Rs.{target:,.2f} ({tgt_pct:+.1f}%)")

            rr = tp.get("risk_reward_ratio", 0)
            if rr > 0:
                lines.append(f"  Risk:Reward = 1:{rr:.1f}")

            hold_days = tp.get("expected_holding_days", 0)
            if hold_days > 0:
                lines.append(f"  Hold Period: {hold_days} days")

            shares = tp.get("suggested_shares", 0)
            pos_pct = tp.get("position_size_pct", 0)
            if shares > 0:
                lines.append(f"  Position Size: {pos_pct*100:.1f}% ({shares} shares)")

            if tp.get("support_level"):
                lines.append(f"  Support: Rs.{tp['support_level']:,.2f}")
            if tp.get("resistance_level"):
                lines.append(f"  Resistance: Rs.{tp['resistance_level']:,.2f}")

            if tp.get("trailing_stop_pct"):
                lines.append(f"  Trailing Stop: {tp['trailing_stop_pct']*100:.1f}%")

        # ---- Trade Plan (SELL) ----
        if tp and decision == "SELL":
            lines.append("")
            lines.append("*--- ACTION ---*")
            lines.append(f"  EXIT / SHORT @ Rs.{current_price:,.2f}")
            hold_days = tp.get("expected_holding_days", 0)
            if hold_days > 0:
                lines.append(f"  Expected move: {hold_days} days")
            lines.append(f"  Expected drop: {expected_return*100:+.2f}%")

        # ---- Technical Analysis ----
        tech = result.get("technical", {})
        tech_payload = tech.get("payload", {}) if isinstance(tech, dict) else {}
        tech_sig = tech_payload.get("signal", {})
        tech_raw = tech_payload.get("raw_signal", "")

        if tech_sig:
            lines.append("")
            lines.append("*--- TECHNICALS ---*")
            rsi_val = tech_sig.get("tech_rsi", 0)
            rsi_label = "Oversold" if rsi_val < 0.35 else ("Overbought" if rsi_val > 0.7 else "Neutral")
            lines.append(f"  RSI: {rsi_val*100:.0f} ({rsi_label})")

            macd_val = tech_sig.get("tech_macd", 0)
            lines.append(f"  MACD: {'BUY' if macd_val > 0.5 else 'SELL'} signal")

            trend_val = tech_sig.get("tech_trend", 0)
            lines.append(f"  Trend: {'Bullish' if trend_val > 0.5 else 'Bearish'}")

            breakout = tech_sig.get("tech_breakout", 0)
            if breakout > 0.5:
                lines.append(f"  BREAKOUT detected!")

            vol = tech_sig.get("tech_volatility", 0)
            lines.append(f"  Volatility: {vol*100:.1f}%")

            # Support/Resistance from raw signal
            support = MessageController._extract_raw_signal_field(tech_raw, "support_level")
            resistance = MessageController._extract_raw_signal_field(tech_raw, "resistance_level")
            if support and resistance:
                lines.append(f"  Support: Rs.{float(support):,.2f}")
                lines.append(f"  Resistance: Rs.{float(resistance):,.2f}")

            lines.append(f"  Tech Score: {tech_sig.get('tech_score', 0)*100:.0f}/100")

        # ---- Fundamental Analysis ----
        fund = result.get("fundamental", {})
        fund_payload = fund.get("payload", {}) if isinstance(fund, dict) else {}
        fund_sig = fund_payload.get("signal", {})
        fund_raw = fund_payload.get("raw_signal", "")

        if fund_sig:
            lines.append("")
            lines.append("*--- FUNDAMENTALS ---*")

            pe = MessageController._extract_raw_signal_field(fund_raw, "pe_ratio")
            fwd_pe = MessageController._extract_raw_signal_field(fund_raw, "forward_pe")
            val_label = MessageController._extract_raw_signal_field(fund_raw, "valuation_label")

            if pe:
                lines.append(f"  PE Ratio: {float(pe):.1f}" + (f" (Fwd: {float(fwd_pe):.1f})" if fwd_pe else ""))
            if val_label:
                lines.append(f"  Valuation: {val_label.upper()}")

            growth = fund_sig.get("fund_growth", 0)
            health = fund_sig.get("fund_health", 0)
            lines.append(f"  Growth Score: {growth*100:.0f}/100")
            lines.append(f"  Financial Health: {health*100:.0f}/100")
            lines.append(f"  Fundamental Score: {fund_sig.get('fund_score', 0)*100:.0f}/100")

        # ---- Sentiment ----
        sent = result.get("sentiment", {})
        sent_payload = sent.get("payload", {}) if isinstance(sent, dict) else {}
        sent_sig = sent_payload.get("signal", {})
        sent_raw = sent_payload.get("raw_signal", "")

        if sent_sig and sent_sig.get("sent_confidence", 0) > 0:
            lines.append("")
            lines.append("*--- NEWS SENTIMENT ---*")
            pos = MessageController._extract_raw_signal_field(sent_raw, "positive_news_count")
            neg = MessageController._extract_raw_signal_field(sent_raw, "negative_news_count")
            trend = MessageController._extract_raw_signal_field(sent_raw, "sentiment_trend")
            if pos and neg:
                lines.append(f"  Positive: {pos} | Negative: {neg}")
            if trend:
                lines.append(f"  Trend: {trend}")
            lines.append(f"  Sentiment Score: {sent_sig.get('sent_score', 0.5)*100:.0f}/100")

        # ---- Market Regime ----
        regime = result.get("regime", {})
        if regime:
            lines.append("")
            regime_name = (regime.get("regime") or "unknown").replace("_", " ").title()
            vol_state = (regime.get("vol_state") or "unknown").title()
            reg_conf = regime.get("regime_confidence", 0)
            lines.append(f"*Market Regime*: {regime_name} | Volatility: {vol_state} ({reg_conf*100:.0f}% conf)")

        # ---- AI Debate Summary ----
        debate = result.get("debate")
        if debate:
            bull = debate.get("bull", {})
            bear = debate.get("bear", {})
            dd = debate.get("debate_decision", {})

            lines.append("")
            lines.append("*--- AI DEBATE ---*")
            lines.append(f"  Bull ({bull.get('confidence', 0)*100:.0f}%) vs Bear ({bear.get('confidence', 0)*100:.0f}%)")
            winner = (dd.get("winning_side") or "neutral").title()
            lines.append(f"  Winner: {winner}")

            lines.append("")
            lines.append("  *Bull says:*")
            for pt in bull.get("key_points", [])[:3]:
                lines.append(f"    + {pt[:100]}")

            lines.append("  *Bear says:*")
            for pt in bear.get("key_points", [])[:3]:
                lines.append(f"    - {pt[:100]}")

        # ---- Hybrid Final Decision ----
        hybrid = result.get("hybrid_decision")
        if hybrid:
            lines.append("")
            lines.append("*--- FINAL VERDICT ---*")
            lines.append(f"  Rule-based: {hybrid.get('rule_decision', '?')} ({hybrid.get('rule_confidence', 0)*100:.0f}%)")
            lines.append(f"  AI Debate: {hybrid.get('debate_decision', '?')} ({hybrid.get('debate_confidence', 0)*100:.0f}%)")
            agreement = "YES - Both agree" if hybrid.get("agreement") else "NO - Disagreement"
            lines.append(f"  Agreement: {agreement}")
            lines.append(f"  *FINAL: {hybrid.get('final_decision', '?')} ({hybrid.get('final_confidence', 0)*100:.0f}% confidence)*")

        # ---- Risk Assessment ----
        risk = result.get("risk_assessment")
        if risk:
            lines.append("")
            risk_level = (risk.get("overall_risk_level") or "unknown").upper()
            lines.append(f"*Risk Level*: {risk_level}")
            if result.get("trade_blocked"):
                lines.append(f"  TRADE BLOCKED: {result.get('block_reason', '?')}")
            for w in risk.get("warnings", []):
                lines.append(f"  Warning: {w}")

        # ---- Event Alerts ----
        event = result.get("event", {})
        evt_payload = event.get("payload", {}) if isinstance(event, dict) else {}
        evt_raw = evt_payload.get("raw_signal", "")
        evt_risk = MessageController._extract_raw_signal_field(evt_raw, "event_risk_level")
        if evt_risk and evt_risk not in ("low", "none"):
            lines.append("")
            lines.append(f"*Event Risk*: {evt_risk.upper()}")
            evt_type = MessageController._extract_raw_signal_field(evt_raw, "event_type")
            if evt_type and evt_type != "None":
                lines.append(f"  Type: {evt_type}")

        # ---- Footer ----
        lines.append("")
        lines.append("_LCF Multi-Agent Trading System_")
        lines.append(f"_Generated: {datetime.now().strftime('%H:%M:%S')}_")

        return "\n".join(lines)

    @staticmethod
    def format_daily_summary(results: List[Dict[str, Any]], mode: str = "adaptive") -> str:
        """Format a rich daily summary of all analyzed stocks."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        buy_signals = []
        sell_signals = []
        hold_signals = []

        for r in results:
            jd = r.get("judge_decision", {})
            payload = jd.get("payload", jd)
            decision = payload.get("decision", "HOLD")
            symbol = r.get("symbol", "?")
            confidence = payload.get("confidence", 0)
            prob_up = payload.get("prob_up_5d", 0)
            exp_ret = payload.get("expected_return_5d", 0)

            # Get price
            tp = r.get("trade_plan") or {}
            price = tp.get("current_price") or tp.get("entry_price", 0)

            # Get tech score
            tech = r.get("technical", {})
            tech_score = 0
            if isinstance(tech, dict):
                tech_sig = tech.get("payload", {}).get("signal", {})
                tech_score = tech_sig.get("tech_score", 0)

            entry = {
                "symbol": symbol, "confidence": confidence, "prob_up": prob_up,
                "exp_ret": exp_ret, "price": price, "tech_score": tech_score,
                "result": r,
            }

            if decision == "BUY":
                buy_signals.append(entry)
            elif decision == "SELL":
                sell_signals.append(entry)
            else:
                hold_signals.append(entry)

        lines = []
        lines.append("*LCF DAILY TRADING REPORT*")
        lines.append(f"Mode: {mode.upper()}")
        lines.append(f"Date: {now}")
        lines.append(f"Stocks Analyzed: {len(results)}")

        # Regime
        if results:
            regime = results[0].get("regime", {})
            if regime:
                regime_name = (regime.get("regime") or "?").replace("_", " ").title()
                vol = (regime.get("vol_state") or "?").title()
                lines.append(f"Market: {regime_name} | Volatility: {vol}")
        lines.append("")

        # ---- BUY Signals ----
        if buy_signals:
            lines.append(f"*BUY SIGNALS ({len(buy_signals)})*")
            lines.append("-" * 25)
            for s in sorted(buy_signals, key=lambda x: -x["confidence"]):
                r = s["result"]
                tp = r.get("trade_plan") or {}
                lines.append(f"*{s['symbol']}* @ Rs.{s['price']:,.2f}")
                lines.append(f"  Confidence: {s['confidence']*100:.0f}% | Prob Up: {s['prob_up']*100:.0f}%")
                lines.append(f"  Expected Return: {s['exp_ret']*100:+.2f}%")

                sl = tp.get("stop_loss_price", 0)
                target = tp.get("target_price", 0)
                rr = tp.get("risk_reward_ratio", 0)
                hold = tp.get("expected_holding_days", 0)
                shares = tp.get("suggested_shares", 0)

                if sl > 0:
                    lines.append(f"  Stop Loss: Rs.{sl:,.2f}")
                if target > 0:
                    lines.append(f"  Target: Rs.{target:,.2f}")
                if rr > 0:
                    lines.append(f"  R:R = 1:{rr:.1f}")
                if hold > 0:
                    lines.append(f"  Hold: {hold} days")
                if shares > 0:
                    lines.append(f"  Qty: {shares} shares")

                # Hybrid agreement
                hybrid = r.get("hybrid_decision")
                if hybrid:
                    agr = "Agree" if hybrid.get("agreement") else "Disagree"
                    lines.append(f"  Rule+AI: {agr} ({hybrid.get('final_confidence', 0)*100:.0f}%)")

                lines.append("")

        # ---- SELL Signals ----
        if sell_signals:
            lines.append(f"*SELL SIGNALS ({len(sell_signals)})*")
            lines.append("-" * 25)
            for s in sorted(sell_signals, key=lambda x: -x["confidence"]):
                r = s["result"]
                lines.append(f"*{s['symbol']}* @ Rs.{s['price']:,.2f}")
                lines.append(f"  Confidence: {s['confidence']*100:.0f}% | Prob Up: {s['prob_up']*100:.0f}%")
                lines.append(f"  Expected Drop: {s['exp_ret']*100:+.2f}%")
                lines.append(f"  Tech Score: {s['tech_score']*100:.0f}/100")

                # Why selling - bear key points
                debate = r.get("debate", {})
                bear = debate.get("bear", {}) if debate else {}
                bear_pts = bear.get("key_points", [])
                if bear_pts:
                    lines.append(f"  Reason: {bear_pts[0][:80]}")

                # Risk
                risk = r.get("risk_assessment")
                if risk:
                    lines.append(f"  Risk: {(risk.get('overall_risk_level') or '?').upper()}")

                hybrid = r.get("hybrid_decision")
                if hybrid:
                    agr = "Agree" if hybrid.get("agreement") else "Disagree"
                    lines.append(f"  Rule+AI: {agr} ({hybrid.get('final_confidence', 0)*100:.0f}%)")

                lines.append("")

        # ---- HOLD ----
        if hold_signals:
            hold_names = ", ".join(s["symbol"] for s in hold_signals)
            lines.append(f"*HOLD ({len(hold_signals)})*: {hold_names}")
            for s in hold_signals:
                lines.append(f"  {s['symbol']}: Rs.{s['price']:,.2f} (Prob Up: {s['prob_up']*100:.0f}%)")
            lines.append("")

        # ---- Quick Stats ----
        lines.append("*--- SUMMARY ---*")
        total_buy = len(buy_signals)
        total_sell = len(sell_signals)
        total_hold = len(hold_signals)
        lines.append(f"  BUY: {total_buy} | SELL: {total_sell} | HOLD: {total_hold}")

        if buy_signals:
            avg_conf = sum(s["confidence"] for s in buy_signals) / len(buy_signals)
            lines.append(f"  Avg BUY confidence: {avg_conf*100:.0f}%")
        if sell_signals:
            avg_conf = sum(s["confidence"] for s in sell_signals) / len(sell_signals)
            lines.append(f"  Avg SELL confidence: {avg_conf*100:.0f}%")

        lines.append("")
        lines.append("_LCF Multi-Agent Trading System_")
        lines.append(f"_Report generated: {now}_")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Sending
    # ------------------------------------------------------------------

    def _send_whatsapp(self, message: str) -> bool:
        """Send a message via WhatsApp using pywhatkit."""
        if not self._enabled or not self._pywhatkit_available:
            logger.info(f"[DRY RUN] WhatsApp message:\n{message[:200]}...")
            return False

        try:
            import pywhatkit
            import time

            # sendwhatmsg_instantly opens WhatsApp Web and types the message
            pywhatkit.sendwhatmsg_instantly(
                phone_no=self._phone,
                message=message,
                wait_time=15,
                tab_close=True,
            )

            # Auto-press Enter to actually send the message
            # pywhatkit only types it — we need to press Enter
            time.sleep(2)
            try:
                import pyautogui
                pyautogui.press('enter')
                time.sleep(1)
            except ImportError:
                # pyautogui not installed — try keyboard module
                try:
                    import keyboard
                    keyboard.press_and_release('enter')
                    time.sleep(1)
                except ImportError:
                    logger.warning("Install pyautogui or keyboard for auto-send: pip install pyautogui")

            logger.info(f"WhatsApp sent to {self._phone}: {message[:50]}...")
            return True
        except Exception as e:
            logger.error(f"WhatsApp send failed: {e}")
            return False

    def send_trade_alert(self, result: Dict[str, Any]) -> bool:
        """Send a trade alert for a single stock if it's BUY or SELL.

        Args:
            result: Single stock analysis result from orchestrator

        Returns:
            True if message was sent successfully
        """
        jd = result.get("judge_decision", {})
        payload = jd.get("payload", jd)
        decision = payload.get("decision", "HOLD")

        if decision == "HOLD" and not self._send_hold:
            return False

        message = self.format_trade_alert(result)
        if not message:
            return False

        return self._send_whatsapp(message)

    def send_daily_summary(self, results: List[Dict[str, Any]], mode: str = "adaptive") -> bool:
        """Send a daily summary of all analyzed stocks.

        Args:
            results: List of stock analysis results from orchestrator
            mode: Current trading mode name

        Returns:
            True if message was sent successfully
        """
        if not results:
            return False

        message = self.format_daily_summary(results, mode)
        return self._send_whatsapp(message)

    def send_custom(self, message: str) -> bool:
        """Send a custom message."""
        return self._send_whatsapp(message)

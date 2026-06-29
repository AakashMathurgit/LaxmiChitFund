"""LCF Demo Runner — verbose step-by-step output for recording.

Shows every stage with inputs and outputs clearly formatted.
Usage:
    python news_stock_tracker/run_demo.py
    python news_stock_tracker/run_demo.py --symbol RELIANCE --mode momentum
    python news_stock_tracker/run_demo.py --whatsapp +919351183542
"""

import os
import sys
import json
import argparse
from datetime import datetime

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_LCF_ROOT = os.path.dirname(_SCRIPT_DIR)
if _LCF_ROOT not in sys.path:
    sys.path.insert(0, _LCF_ROOT)

os.environ["LCF_DEBUG"] = "0"  # suppress agent debug prints

CONFIG_PATH = os.path.join(_LCF_ROOT, "config.yaml")
SEP = "=" * 65
THIN = "-" * 65


def header(title, step=None):
    print(f"\n{SEP}")
    prefix = f"  STEP {step}: " if step else "  "
    print(f"{prefix}{title}")
    print(SEP)


def sub(label, value=""):
    if value:
        print(f"  {label:<28} {value}")
    else:
        print(f"  {label}")


def main():
    parser = argparse.ArgumentParser(description="LCF Demo Runner")
    parser.add_argument("--symbol", type=str, default="HDFCBANK", help="Stock to analyze")
    parser.add_argument("--mode", type=str, default="adaptive", help="Trading mode")
    parser.add_argument("--whatsapp", type=str, default="", help="WhatsApp number")
    args = parser.parse_args()

    start = datetime.now()

    print(f"\n{'#' * 65}")
    print(f"#  LCF DEMO — Complete Pipeline Walkthrough")
    print(f"#  Stock: {args.symbol} | Mode: {args.mode.upper()}")
    print(f"#  Time: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 65}")

    # ================================================================
    # STEP 1: Portfolio & Watchlist
    # ================================================================
    header("PORTFOLIO & WATCHLIST", 1)
    from src.main.controllers.portfolio_manager import PortfolioManager
    pm = PortfolioManager()
    pm.load()

    sub("Initial Capital:", f"Rs.{pm._state.initial_capital:,.0f}")
    sub("Current Cash:", f"Rs.{pm._state.cash:,.0f}")
    sub("Open Positions:", f"{len(pm.get_all_holdings())}")
    sub("Watchlist Stocks:", f"{len(pm.get_watchlist())}")
    print(f"  {THIN}")
    for w in pm.get_watchlist():
        sig = f" [{w.last_signal}]" if w.last_signal else ""
        print(f"  [{w.priority[0].upper()}] {w.symbol:<12} {w.reason[:35]}{sig}")

    # ================================================================
    # STEP 2: Initialize Orchestrator
    # ================================================================
    header("INITIALIZE ORCHESTRATOR", 2)
    from src.pipeline.orchestrator import PipelineOrchestrator
    orchestrator = PipelineOrchestrator(config_path=CONFIG_PATH, mode=args.mode)

    sub("Mode:", f"{orchestrator.mode.upper()} — {orchestrator.mode_config.description}")
    sub("Buy Threshold:", f"{orchestrator._tuning.judge.buy_threshold}")
    sub("Sell Threshold:", f"{orchestrator._tuning.judge.sell_threshold}")
    sub("Rule Weight:", f"{orchestrator._tuning.debate.rule_weight}")
    sub("Debate Weight:", f"{orchestrator._tuning.debate.debate_weight}")
    sub("Max Positions:", f"{orchestrator.mode_config.max_positions}")
    sub("Hold Days:", f"{orchestrator.mode_config.hold_days}")
    sub("PatternStore Records:", f"{orchestrator.pattern_store.count}")
    sub("TradeMemory Records:", f"{orchestrator.trade_memory.count}")
    sub("LLM Model:", f"{orchestrator.config.get('llm', {}).get('model', 'gpt-4.1')}")
    sub("Data Providers:", "Yahoo Finance, NSE Corporate, RSS News")

    # ================================================================
    # STEP 3: Fetch Market Regime (ONCE)
    # ================================================================
    header("MARKET REGIME DETECTION", 3)
    print("  Fetching NIFTY 50 index + India VIX...")
    index_data = orchestrator.data_processor.build_index_context(index_symbol="^NSEI")
    regime = orchestrator._detect_regime(index_data)

    sub("Index:", "NIFTY 50 (^NSEI)")
    sub("Index Bars:", f"{len(index_data.historical_ohlc) if index_data and index_data.historical_ohlc else 0}")
    sub("Market Regime:", f"{regime.market_regime.value}")
    sub("Volatility:", f"{regime.volatility_state.value}")
    sub("Confidence:", f"{regime.regime_confidence*100:.0f}%")

    vix_val = getattr(orchestrator, '_last_vix', None)
    if vix_val:
        sub("India VIX:", f"{vix_val:.2f}")

    print(f"\n  Interpretation: ", end="")
    if regime.market_regime.value == "bear_trend":
        print("Market is in a DOWNTREND. System will be DEFENSIVE.")
        print("  -> Higher sell sensitivity, lower buy threshold, reduced position sizes")
    elif regime.market_regime.value == "bull_trend":
        print("Market is TRENDING UP. System will be AGGRESSIVE.")
    else:
        print("Market is SIDEWAYS. System will be BALANCED.")

    # ================================================================
    # STEP 4: Fetch Stock Data
    # ================================================================
    symbol = args.symbol.upper()
    header(f"FETCH DATA FOR {symbol}", 4)
    print(f"  Fetching from Yahoo Finance + NSE + RSS News...")
    stock_ctx = orchestrator.data_processor.build_stock_context(
        symbol=symbol, index_data=index_data,
    )

    sub("Company:", f"{stock_ctx.company_name}")
    sub("Sector:", f"{stock_ctx.sector}")
    sub("Last Close:", f"Rs.{stock_ctx.last_close:,.2f}")
    sub("Previous Close:", f"Rs.{stock_ctx.previous_close:,.2f}" if stock_ctx.previous_close else "N/A")
    day_change = ((stock_ctx.last_close - stock_ctx.previous_close) / stock_ctx.previous_close * 100) if stock_ctx.previous_close else 0
    sub("Day Change:", f"{day_change:+.2f}%")
    bars = getattr(stock_ctx, 'price_data', None) or getattr(stock_ctx, 'historical_ohlc', []) or []
    sub("Historical Bars:", f"{len(bars)}")
    sub("News Articles:", f"{len(getattr(stock_ctx, 'news_items', None) or getattr(stock_ctx, 'news', None) or [])}")
    sub("Events:", f"{len(getattr(stock_ctx, 'event_data', None) or getattr(stock_ctx, 'events', None) or [])}")
    has_fund = hasattr(stock_ctx, 'fundamentals') and stock_ctx.fundamentals
    sub("Has Fundamentals:", f"{'Yes' if has_fund else 'No'}")

    if has_fund:
        fund = stock_ctx.fundamentals
        print(f"  {THIN}")
        sub("PE Ratio:", f"{getattr(fund, 'pe_ratio', 'N/A')}")
        sub("Forward PE:", f"{getattr(fund, 'forward_pe', 'N/A')}")
        sub("Revenue Growth:", f"{getattr(fund, 'revenue_growth_yoy', 'N/A')}")
        sub("Profit Margin:", f"{getattr(fund, 'profit_margin', 'N/A')}")
        sub("D/E Ratio:", f"{getattr(fund, 'debt_to_equity', 'N/A')}")
        sub("ROE:", f"{getattr(fund, 'roe', 'N/A')}")

    # ================================================================
    # STEP 5: Run Analysis Agents (4 parallel)
    # ================================================================
    header(f"ANALYSIS AGENTS (4 parallel)", 5)
    result = orchestrator.analyse_stock_context(stock_ctx, regime=regime)

    # Technical
    tech = result.get("technical", {})
    tech_sig = tech.get("payload", {}).get("signal", {}) if isinstance(tech, dict) else {}
    print(f"\n  TECHNICAL AGENT")
    print(f"  {THIN}")
    sub("RSI (14-day):", f"{tech_sig.get('tech_rsi', 0)*100:.1f}")
    sub("MACD Signal:", f"{'BUY' if tech_sig.get('tech_macd', 0) > 0.5 else 'SELL'}")
    sub("Trend:", f"{'Bullish' if tech_sig.get('tech_trend', 0) > 0.5 else 'Bearish'}")
    sub("Breakout:", f"{'YES' if tech_sig.get('tech_breakout', 0) > 0.5 else 'No'}")
    sub("Volatility:", f"{tech_sig.get('tech_volatility', 0)*100:.1f}%")
    sub("Tech Score:", f"{tech_sig.get('tech_score', 0)*100:.0f}/100")

    # Fundamental
    fund_r = result.get("fundamental", {})
    fund_sig = fund_r.get("payload", {}).get("signal", {}) if isinstance(fund_r, dict) else {}
    fund_raw = fund_r.get("payload", {}).get("raw_signal", "") if isinstance(fund_r, dict) else ""
    print(f"\n  FUNDAMENTAL AGENT")
    print(f"  {THIN}")
    # Extract PE from raw_signal
    import re
    pe_match = re.search(r"pe_ratio=([\d.]+)", str(fund_raw))
    fpe_match = re.search(r"forward_pe=([\d.]+)", str(fund_raw))
    val_match = re.search(r"valuation_label='(\w+)'", str(fund_raw))
    sub("PE Ratio:", f"{float(pe_match.group(1)):.1f}" if pe_match else "N/A")
    sub("Forward PE:", f"{float(fpe_match.group(1)):.1f}" if fpe_match else "N/A")
    sub("Valuation:", f"{val_match.group(1).upper()}" if val_match else "N/A")
    sub("Growth Score:", f"{fund_sig.get('fund_growth', 0)*100:.0f}/100")
    sub("Health Score:", f"{fund_sig.get('fund_health', 0)*100:.0f}/100")
    sub("Fund Score:", f"{fund_sig.get('fund_score', 0)*100:.0f}/100")

    # Sentiment
    sent = result.get("sentiment", {})
    sent_sig = sent.get("payload", {}).get("signal", {}) if isinstance(sent, dict) else {}
    print(f"\n  SENTIMENT AGENT (LLM-powered)")
    print(f"  {THIN}")
    sub("Sentiment Score:", f"{sent_sig.get('sent_score', 0.5)*100:.0f}/100")
    sub("News Trend:", f"{'Improving' if sent_sig.get('sent_trend', 0.5) > 0.6 else 'Stable' if sent_sig.get('sent_trend', 0.5) > 0.4 else 'Deteriorating'}")
    sub("Confidence:", f"{sent_sig.get('sent_confidence', 0)*100:.0f}%")

    # Event
    evt = result.get("event", {})
    evt_sig = evt.get("payload", {}).get("signal", {}) if isinstance(evt, dict) else {}
    evt_raw = evt.get("payload", {}).get("raw_signal", "") if isinstance(evt, dict) else ""
    risk_match = re.search(r"event_risk_level='(\w+)'", str(evt_raw))
    print(f"\n  EVENT AGENT")
    print(f"  {THIN}")
    sub("Event Score:", f"{evt_sig.get('evt_score', 0)*100:.0f}/100")
    sub("Event Risk:", f"{risk_match.group(1).upper()}" if risk_match else "N/A")
    sub("Gap Down:", f"{'YES' if evt_sig.get('evt_gap_down', 0) > 0.5 else 'No'}")
    sub("Gap Up:", f"{'YES' if evt_sig.get('evt_gap_up', 0) > 0.5 else 'No'}")
    sub("Earnings Impact:", f"{'YES' if evt_sig.get('evt_earnings', 0) > 0.5 else 'No'}")

    # ================================================================
    # STEP 6: Rule-Based Decision (JudgeAgent)
    # ================================================================
    header("RULE-BASED DECISION (JudgeAgent)", 6)
    jd = result.get("judge_decision", {})
    jd_payload = jd.get("payload", {}) if isinstance(jd, dict) else {}

    sub("Decision:", f"{jd_payload.get('decision', '?')}")
    sub("Prob Up (5 days):", f"{jd_payload.get('prob_up_5d', 0)*100:.1f}%")
    sub("Expected Return:", f"{jd_payload.get('expected_return_5d', 0)*100:+.2f}%")
    sub("Downside Risk:", f"{jd_payload.get('downside_risk_prob', 0)*100:.1f}%")
    sub("Confidence:", f"{jd_payload.get('confidence', 0)*100:.1f}%")
    sub("Position Size:", f"{jd_payload.get('position_size_pct', 0)*100:.2f}%")

    print(f"\n  Input: 20-dim feature vector from 4 agents")
    print(f"  Method: Weighted sum with mode-specific weights")
    print(f"  Thresholds: BUY > {orchestrator._tuning.judge.buy_threshold:.2f}, "
          f"SELL < {orchestrator._tuning.judge.sell_threshold:.2f}")

    # ================================================================
    # STEP 7: AI Debate (Bull vs Bear)
    # ================================================================
    header("AI DEBATE (Bull vs Bear via GPT-4.1)", 7)
    debate = result.get("debate")
    if debate:
        bull = debate.get("bull", {})
        bear = debate.get("bear", {})
        dd = debate.get("debate_decision", {})

        print(f"\n  BULL AGENT — Argues BUY")
        print(f"  {THIN}")
        sub("Recommendation:", f"{bull.get('recommendation', '?')}")
        sub("Confidence:", f"{bull.get('confidence', 0)*100:.0f}%")
        print(f"  Key Points:")
        for i, pt in enumerate(bull.get("key_points", []), 1):
            print(f"    {i}. + {pt[:90]}")
        print(f"\n  Reasoning: {bull.get('reasoning', '')[:150]}...")

        print(f"\n  BEAR AGENT — Argues SELL")
        print(f"  {THIN}")
        sub("Recommendation:", f"{bear.get('recommendation', '?')}")
        sub("Confidence:", f"{bear.get('confidence', 0)*100:.0f}%")
        print(f"  Key Points:")
        for i, pt in enumerate(bear.get("key_points", []), 1):
            print(f"    {i}. - {pt[:90]}")
        print(f"\n  Reasoning: {bear.get('reasoning', '')[:150]}...")

        print(f"\n  DEBATE RESULT")
        print(f"  {THIN}")
        sub("Winner:", f"{dd.get('winning_side', '?').upper()}")
        sub("Bull Strength:", f"{dd.get('bull_strength', 0)*100:.1f}%")
        sub("Bear Strength:", f"{dd.get('bear_strength', 0)*100:.1f}%")
        sub("Debate Decision:", f"{dd.get('decision', '?')} @ {dd.get('confidence', 0)*100:.0f}%")

        print(f"\n  Input: DebateContext with real PE ratios, headlines, 52-week range,")
        print(f"         RAG similar setups from PatternStore, past trade mistakes")
    else:
        print("  [Debate not available for this stock]")

    # ================================================================
    # STEP 8: Hybrid Decision (Rule + Debate Combined)
    # ================================================================
    header("HYBRID DECISION (Rule + Debate Combined)", 8)
    hybrid = result.get("hybrid_decision")
    if hybrid:
        sub("Rule-Based:", f"{hybrid.get('rule_decision', '?')} @ {hybrid.get('rule_confidence', 0)*100:.0f}%")
        sub("AI Debate:", f"{hybrid.get('debate_decision', '?')} @ {hybrid.get('debate_confidence', 0)*100:.0f}%")
        sub("Agreement:", f"{'YES — Both paths agree' if hybrid.get('agreement') else 'NO — Paths DISAGREE'}")
        print(f"  {THIN}")
        sub("FINAL DECISION:", f"{hybrid.get('final_decision', '?')}")
        sub("FINAL CONFIDENCE:", f"{hybrid.get('final_confidence', 0)*100:.0f}%")
        print(f"\n  Formula: final = {orchestrator._tuning.debate.rule_weight:.3f} x Rule "
              f"+ {orchestrator._tuning.debate.debate_weight:.3f} x Debate")
        print(f"  Reasoning: {hybrid.get('reasoning', '')}")

        if hybrid.get('agreement'):
            print(f"\n  Both paths agree on {hybrid.get('final_decision')} "
                  f"-> confidence BOOSTED to {hybrid.get('final_confidence', 0)*100:.0f}%")
        else:
            print(f"\n  Paths DISAGREE: Rule says {hybrid.get('rule_decision')} but Debate says {hybrid.get('debate_decision')}")
            print(f"  -> Confidence REDUCED to {hybrid.get('final_confidence', 0)*100:.0f}%")

    # ================================================================
    # STEP 9: Trade Plan + Risk Assessment
    # ================================================================
    header("TRADE PLAN & RISK ASSESSMENT", 9)
    tp = result.get("trade_plan")
    risk = result.get("risk_assessment")

    if tp:
        sub("Action:", f"{tp.get('decision', '?')} @ Rs.{tp.get('current_price', 0):,.2f}")
        sub("Entry Type:", f"{(tp.get('entry_type') or 'market').upper()}")
        if tp.get('stop_loss_price', 0) > 0:
            sub("Stop Loss:", f"Rs.{tp['stop_loss_price']:,.2f}")
        if tp.get('target_price', 0) > 0:
            sub("Target:", f"Rs.{tp['target_price']:,.2f}")
        sub("R:R Ratio:", f"1:{tp.get('risk_reward_ratio', 0):.1f}")
        sub("Hold Period:", f"{tp.get('expected_holding_days', 5)} days")
        sub("Expected Return:", f"{jd_payload.get('expected_return_5d', 0)*100:+.2f}%")

    if risk:
        print(f"  {THIN}")
        sub("Risk Level:", f"{(risk.get('overall_risk_level') or '?').upper()}")
        sub("Regime Risk Multiplier:", f"{risk.get('regime_risk_multiplier', 1.0):.2f}x")
        blocked = any(p.get("blocked") for p in risk.get("positions", []))
        sub("Trade Blocked:", f"{'YES' if blocked else 'No'}")
        for w in risk.get("warnings", []):
            print(f"  WARNING: {w}")

    # ================================================================
    # STEP 10: Portfolio Action
    # ================================================================
    header("PORTFOLIO ACTION", 10)
    actions = pm.process_signals([result], auto_enter=False, auto_exit=True, mode=args.mode)

    if actions["entered"]:
        for a in actions["entered"]:
            print(f"  ENTERED: BUY {a['symbol']} {a['shares']} shares @ Rs.{a['price']:,.2f}")
    elif actions["exited"]:
        for a in actions["exited"]:
            print(f"  EXITED: SOLD {a['symbol']} P&L: Rs.{a.get('pnl', 0):+,.2f}")
    elif actions["sl_triggered"]:
        print(f"  STOP LOSS triggered!")
    elif actions["target_hit"]:
        print(f"  TARGET hit!")
    else:
        decision = hybrid.get("final_decision", "?") if hybrid else "?"
        if decision == "SELL":
            print(f"  No position held in {symbol} — SELL signal noted for exit if holding")
        elif decision == "BUY":
            print(f"  BUY signal detected but auto-trade not enabled")
            print(f"  Run with --auto-trade to auto-enter positions")
        else:
            print(f"  HOLD — no action taken")

    sub("Watchlist Updated:", f"{symbol} -> [{hybrid.get('final_decision', '?') if hybrid else '?'}]")
    print(f"\n{pm.portfolio_text_summary()}")

    # ================================================================
    # STEP 11: WhatsApp Alert
    # ================================================================
    if args.whatsapp:
        header("WHATSAPP ALERT", 11)
        from src.main.controllers.message_controller import MessageController
        messenger = MessageController(phone=args.whatsapp)

        print(f"  Sending to: {args.whatsapp}")
        sent = messenger.send_trade_alert(result)
        if sent:
            print(f"  Trade alert SENT")
        else:
            decision = hybrid.get("final_decision", "HOLD") if hybrid else "HOLD"
            if decision == "HOLD":
                print(f"  Skipped — HOLD signals don't trigger alerts")
            else:
                print(f"  Alert sent for {decision} signal")

        messenger.send_daily_summary([result], mode=args.mode)
        print(f"  Daily summary SENT")
    else:
        header("WHATSAPP ALERT (skipped)", 11)
        print(f"  Add --whatsapp +91XXXXXXXXXX to send alerts")

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    header("DEMO COMPLETE")
    elapsed = (datetime.now() - start).total_seconds()

    decision = hybrid.get("final_decision", "?") if hybrid else "?"
    confidence = hybrid.get("final_confidence", 0) if hybrid else 0
    agreement = hybrid.get("agreement", False) if hybrid else False

    print(f"""
  Stock:          {symbol} ({stock_ctx.company_name})
  Price:          Rs.{stock_ctx.last_close:,.2f}
  Mode:           {args.mode.upper()}
  Regime:         {regime.market_regime.value} | {regime.volatility_state.value}

  Technical:      {tech_sig.get('tech_score', 0)*100:.0f}/100 (RSI={tech_sig.get('tech_rsi', 0)*100:.0f})
  Fundamental:    {fund_sig.get('fund_score', 0)*100:.0f}/100 (PE={float(pe_match.group(1)) if pe_match else 0:.1f})
  Sentiment:      {sent_sig.get('sent_score', 0.5)*100:.0f}/100
  Event Risk:     {risk_match.group(1).upper() if risk_match else 'N/A'}

  Rule-Based:     {hybrid.get('rule_decision', '?') if hybrid else '?'} @ {hybrid.get('rule_confidence', 0)*100:.0f if hybrid else 0}%
  AI Debate:      {hybrid.get('debate_decision', '?') if hybrid else '?'} @ {hybrid.get('debate_confidence', 0)*100:.0f if hybrid else 0}%
  Agreement:      {'YES' if agreement else 'NO'}

  >>> FINAL: {decision} @ {confidence*100:.0f}% confidence <<<

  PatternStore:   {orchestrator.pattern_store.count} records (learning)
  TradeMemory:    {orchestrator.trade_memory.count} records (learning)
  Total Time:     {elapsed:.1f}s
""")
    print(SEP)


if __name__ == "__main__":
    main()

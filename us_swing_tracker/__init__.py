"""US swing-trading funnel package.

Scans the same 100-stock universe as the intraday tracker, but evaluates each
candidate on a multi-day horizon (1 week / 1 month) using fundamentals, news,
and the future-prediction agent — answering "is this worth HOLDING?" rather
than "can I profit right now?". Trades into a separate Alpaca paper account.
"""

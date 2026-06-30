"""Quick Pushover connectivity test.

Sends a single test push to verify your Pushover api_token + user_key work,
BEFORE wiring them into the cloud deployment.

Usage (PowerShell):
    $env:PUSHOVER_TOKEN="your_app_api_token"
    $env:PUSHOVER_USER="your_user_key"
    python scripts/test_pushover.py

Or pass as args:
    python scripts/test_pushover.py <api_token> <user_key>

It prints the exact Pushover API response so we can see whether the
credentials are valid (status=1) or rejected (status=0 + errors).
"""

from __future__ import annotations

import os
import sys

import requests

PUSHOVER_API_URL = "https://api.pushover.net/1/messages.json"


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    token = (argv[0] if len(argv) > 0 else os.environ.get("PUSHOVER_TOKEN", "")).strip()
    user = (argv[1] if len(argv) > 1 else os.environ.get("PUSHOVER_USER", "")).strip()

    if not token or not user:
        print("ERROR: provide token + user via args or PUSHOVER_TOKEN / PUSHOVER_USER env vars.")
        return 2

    print(f"Testing Pushover — token ...{token[-4:]}, user ...{user[-4:]}")
    resp = requests.post(
        PUSHOVER_API_URL,
        data={
            "token": token,
            "user": user,
            "title": "LCF Pushover Test",
            "message": "If you can read this, your Pushover credentials work.",
            "priority": "0",
        },
        timeout=10,
    )
    print(f"HTTP {resp.status_code}")
    print(f"Response: {resp.text}")

    try:
        body = resp.json()
    except Exception:
        body = {}

    if resp.status_code == 200 and body.get("status") == 1:
        print("\nSUCCESS — check your phone for the push notification.")
        return 0
    print("\nFAILED — credentials rejected. Common causes:")
    print("  - api_token wrong (must be the *application* token from pushover.net/apps)")
    print("  - user_key wrong (your account user key, not your email/password)")
    print("  - application was deleted")
    return 1


if __name__ == "__main__":
    sys.exit(main())

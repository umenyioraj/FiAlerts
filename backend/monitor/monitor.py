import sys
from pathlib import Path

# Fix path BEFORE importing sibling packages
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import threading
import time
from typing import Literal, Optional

import yfinance as yf
import resend
import psycopg2
import psycopg2.extras
from enviornment.enviornment import RESEND_API_KEY, NEON_URL, NEON_USERNAME, NEON_PASSWORD

resend.api_key = RESEND_API_KEY

# Convert JDBC URL to psycopg2 format
_DB_URL = NEON_URL.replace("jdbc:", "", 1)


def _get_conn():
    return psycopg2.connect(_DB_URL, user=NEON_USERNAME, password=NEON_PASSWORD)


def check_alert(alert: dict) -> bool:
    """Check one alert. Returns True if the target price was hit (alert is done)."""
    try:
        stock = yf.Ticker(alert["ticker"])
        price = stock.fast_info["lastPrice"]
    except Exception as e:
        print(f"Error fetching {alert['ticker']}: {e}")
        return False

    hit = (
        (alert["direction"] == "above" and price >= alert["target_price"])
        or (alert["direction"] == "below" and price <= alert["target_price"])
    )

    if hit:
        msg = (
            f"Alert: {alert['ticker']} has crossed {alert['direction']} "
            f"{alert['target_price']}. Current price: {price}"
        )
        print(msg)

        # Mark as triggered in the database
        with _get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE stock_alerts SET status='triggered', triggered_at=NOW() WHERE id=%s",
                    (alert["id"],)
                )
            conn.commit()

        if alert.get("user_email"):
            resend.Emails.send({
                "from": "onboarding@resend.dev",
                "to": alert["user_email"],
                "subject": f"Stock Alert: {alert['ticker']}",
                "html": f"<p>{msg}</p>",
            })
        return True

    return False


def add_to_monitoring_queue(
    ticker: str,
    target_price: float,
    direction: Literal["above", "below"],
    user_email: Optional[str] = None,
) -> dict:
    """Insert an alert into the database. Returns the created row."""
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """INSERT INTO stock_alerts (ticker, target_price, direction, user_email, status)
                   VALUES (%s, %s, %s, %s, 'active') RETURNING *""",
                (ticker.upper(), target_price, direction, user_email)
            )
            row = dict(cur.fetchone())
        conn.commit()
    return row


def get_active_alerts() -> list[dict]:
    """Fetch all active alerts from the database."""
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM stock_alerts WHERE status='active'")
            return [dict(row) for row in cur.fetchall()]


def cancel_alert(alert_id: str) -> bool:
    """Cancel an alert by ID."""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE stock_alerts SET status='cancelled' WHERE id=%s AND status='active'",
                (alert_id,)
            )
            affected = cur.rowcount
        conn.commit()
    return affected > 0


def _monitor_loop(interval: int = 60):
    """Background loop — fetches active alerts from DB each cycle, checks them."""
    while True:
        try:
            alerts = get_active_alerts()
            for alert in alerts:
                check_alert(alert)
        except Exception as e:
            print(f"Monitor loop error: {e}")
        time.sleep(interval)


def start_monitoring(interval: int = 300):
    """Start the background monitoring thread (call once at server startup)."""
    t = threading.Thread(target=_monitor_loop, args=(interval,), daemon=True)
    t.start()
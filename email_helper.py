"""Tychain — Email helper.

Handles dynamic SMTP delivery for user-specific signal alerts. SMTP settings
are read from environment variables, so the same code works for any provider
(Gmail / SES / Mailgun / Postmark). When SMTP_HOST is not set, send_email
falls back to a "mock" mode that logs the message to stdout — useful for
development.

Public API:
    send_email(to_email, to_name, subject, html_body, text_body=None) -> True | str
    email_signal_alert(to_email, to_name, ticker, signal_type,
                       strength, price, next_price, summary,
                       currency_symbol="₺", dashboard_url=None) -> True | str
"""

from __future__ import annotations

import os
import smtplib
import socket
import ssl
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr, formatdate, make_msgid

# ── Config (resolved at call time, not import time, so tests can monkey-patch) ──

def _cfg() -> dict:
    """Read SMTP / branding config from env. Resolved per-call so unit
    tests and runtime overrides (e.g. setting MAIL_FROM in app.py) work."""
    return {
        "MAIL_FROM": os.environ.get("MAIL_FROM", "noreply@tychain.app"),
        "MAIL_NAME": os.environ.get("MAIL_NAME", "Tychain Alerts"),
        "SMTP_HOST": os.environ.get("SMTP_HOST", ""),
        "SMTP_PORT": int(os.environ.get("SMTP_PORT", 465)),
        "SMTP_USER": os.environ.get("SMTP_USER", ""),
        "SMTP_PASS": os.environ.get("SMTP_PASS", ""),
        "SMTP_TIMEOUT": int(os.environ.get("SMTP_TIMEOUT", 20)),
        "APP_URL":   os.environ.get("APP_URL", "http://localhost:8080"),
    }


# ── Low-level send ─────────────────────────────────────────────────────────────

def _html_to_text(html: str) -> str:
    """Best-effort HTML -> text fallback for clients that disable HTML."""
    try:
        from bs4 import BeautifulSoup  # optional dep
        return BeautifulSoup(html, "html.parser").get_text(separator="\n").strip()
    except Exception:
        # Strip tags crudely so we still produce a readable plain part
        import re
        return re.sub(r"<[^>]+>", "", html).strip()


def send_email(
    to_email,
    to_name,
    subject,
    html_body,
    text_body=None,
):
    """Send a single email. Returns True on success, error-string on failure.

    When SMTP_HOST is empty we run in mock mode (log + return True).
    """
    cfg = _cfg()

    if not cfg["SMTP_HOST"]:
        print(
            f"[email_helper] SMTP_HOST not set - mock-sending to {to_email}\n"
            f"  Subject: {subject}"
        )
        return True

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = formataddr((cfg["MAIL_NAME"], cfg["MAIL_FROM"]))
    msg["To"] = formataddr((to_name or to_email, to_email))
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid(domain=cfg["MAIL_FROM"].split("@")[-1] or "tychain.app")
    msg["X-Tychain-Alert"] = "signal"

    msg.attach(MIMEText(text_body or _html_to_text(html_body), "plain", "utf-8"))
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    try:
        ctx = ssl.create_default_context()
        if cfg["SMTP_PORT"] == 465:
            server = smtplib.SMTP_SSL(
                cfg["SMTP_HOST"], cfg["SMTP_PORT"],
                timeout=cfg["SMTP_TIMEOUT"], context=ctx,
            )
        else:
            server = smtplib.SMTP(
                cfg["SMTP_HOST"], cfg["SMTP_PORT"], timeout=cfg["SMTP_TIMEOUT"]
            )
            server.ehlo()
            try:
                server.starttls(context=ctx)
                server.ehlo()
            except smtplib.SMTPNotSupportedError:
                # Server doesn't support STARTTLS - proceed in plaintext (e.g. local mailhog)
                pass

        if cfg["SMTP_USER"] and cfg["SMTP_PASS"]:
            server.login(cfg["SMTP_USER"], cfg["SMTP_PASS"])

        server.sendmail(cfg["MAIL_FROM"], [to_email], msg.as_string())
        server.quit()
        return True

    except (smtplib.SMTPException, socket.timeout, socket.gaierror, OSError) as e:
        err = f"{type(e).__name__}: {e}"
        print(f"[email_helper] Failed to send to {to_email}: {err}")
        return err


# ── Branded signal-alert email ─────────────────────────────────────────────────

_BUY_SIGNALS = {"BUY", "STRONG BUY"}


def _palette(signal_type):
    """Return (color, emoji, label) tuned per signal type."""
    s = (signal_type or "").upper()
    if s in _BUY_SIGNALS:
        return ("#22C55E", "🚀" if s == "STRONG BUY" else "📈", s)
    if s in {"SELL", "STRONG SELL"}:
        return ("#EF4444", "🔴" if s == "STRONG SELL" else "📉", s)
    return ("#94A3B8", "⏸️", s or "HOLD")


def email_signal_alert(
    to_email,
    to_name,
    ticker,
    signal_type,
    strength,
    price,
    next_price,
    summary,
    currency_symbol="₺",
    dashboard_url=None,
):
    """Send a branded signal-alert email to a tracked stock subscriber.

    The body always includes: ticker, signal type, signal strength %,
    current price, forecast price, and a direct link to the user's
    Tychain dashboard.
    """
    cfg = _cfg()
    color, emoji, label = _palette(signal_type)

    pct_change = 0.0
    try:
        if price and float(price) > 0:
            pct_change = round(((float(next_price) - float(price)) / float(price)) * 100, 2)
    except (TypeError, ValueError):
        pct_change = 0.0
    pct_sign = "+" if pct_change >= 0 else ""

    # Dashboard link - prefer caller-supplied URL, else build from APP_URL
    base = (dashboard_url or cfg["APP_URL"]).rstrip("/")
    dashboard_link = f"{base}/dashboard?ticker={ticker}"
    analysis_link = f"{base}/?stock={ticker}"

    subject = f"{emoji} {label} - {ticker} @ {strength}% - Tychain"
    sent_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    try:
        price_str = f"{float(price):,.2f}"
    except (TypeError, ValueError):
        price_str = str(price)
    try:
        next_price_str = f"{float(next_price):,.2f}"
    except (TypeError, ValueError):
        next_price_str = str(next_price)

    html_body = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{subject}</title>
</head>
<body style="margin:0;padding:0;background:#0D1B2A;font-family:'Segoe UI',Roboto,Arial,sans-serif;color:#ECEFF1;">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#0D1B2A;padding:32px 12px;">
    <tr><td align="center">
      <table role="presentation" width="600" cellpadding="0" cellspacing="0"
             style="max-width:600px;background:#1A2B3C;border-radius:14px;overflow:hidden;border:1px solid #1E3A5F;">
        <tr>
          <td style="padding:26px 30px;background:linear-gradient(135deg,#0D1B2A,#1A2B3C);
                     border-bottom:3px solid {color};">
            <div style="font-size:.75rem;letter-spacing:.18em;color:#78909C;text-transform:uppercase;">
              Tychain - Signal Alert
            </div>
            <h1 style="margin:6px 0 0;font-size:1.45rem;color:#ECEFF1;">
              {emoji} {label} on {ticker}
            </h1>
            <p style="margin:6px 0 0;color:#94A3B8;font-size:.85rem;">
              You're tracking <strong style="color:#ECEFF1;">{ticker}</strong> &middot; {sent_at}
            </p>
          </td>
        </tr>
        <tr>
          <td style="padding:26px 30px;">
            <div style="display:inline-block;background:{color}22;border:1px solid {color};
                        color:{color};padding:9px 22px;border-radius:30px;
                        font-size:1.05rem;font-weight:800;letter-spacing:.05em;margin-bottom:18px;">
              {label} &middot; {strength}% strength
            </div>

            <table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;">
              <tr><td style="padding:11px 0;border-bottom:1px solid #1E3A5F;color:#94A3B8;font-size:.85rem;">Stock ticker</td>
                  <td align="right" style="padding:11px 0;border-bottom:1px solid #1E3A5F;color:#ECEFF1;font-weight:700;">{ticker}</td></tr>
              <tr><td style="padding:11px 0;border-bottom:1px solid #1E3A5F;color:#94A3B8;font-size:.85rem;">Signal strength</td>
                  <td align="right" style="padding:11px 0;border-bottom:1px solid #1E3A5F;color:{color};font-weight:700;">{strength}%</td></tr>
              <tr><td style="padding:11px 0;border-bottom:1px solid #1E3A5F;color:#94A3B8;font-size:.85rem;">Current price</td>
                  <td align="right" style="padding:11px 0;border-bottom:1px solid #1E3A5F;color:#ECEFF1;font-weight:700;font-family:Menlo,monospace;">{currency_symbol}{price_str}</td></tr>
              <tr><td style="padding:11px 0;color:#94A3B8;font-size:.85rem;">Forecast (next day)</td>
                  <td align="right" style="padding:11px 0;color:#ECEFF1;font-weight:700;font-family:Menlo,monospace;">
                    {currency_symbol}{next_price_str}
                    <span style="color:{color};font-weight:600;font-size:.85rem;">({pct_sign}{pct_change}%)</span>
                  </td></tr>
            </table>

            <div style="background:#0D1B2A;border-left:4px solid #29B6F6;
                        border-radius:0 8px 8px 0;padding:14px 18px;margin:22px 0;
                        color:#CFD8DC;font-size:.9rem;line-height:1.6;">
              {summary or "Model summary unavailable."}
            </div>

            <a href="{dashboard_link}"
               style="display:inline-block;background:#29B6F6;color:#0D1B2A;text-decoration:none;
                      padding:13px 28px;border-radius:8px;font-weight:700;margin-top:6px;">
              Open my dashboard &rarr;
            </a>
            <p style="margin:14px 0 0;font-size:.8rem;color:#94A3B8;">
              Or jump straight to <a href="{analysis_link}" style="color:#29B6F6;">{ticker} analysis</a>.
            </p>
          </td>
        </tr>
        <tr>
          <td style="padding:18px 30px;background:#0D1B2A;font-size:.78rem;color:#546E7A;
                     border-top:1px solid #1E3A5F;">
            &#9888; <strong>Disclaimer:</strong> Tychain is an educational AI tool, not financial advice.
            Forecasts can be wrong. Always do your own research before investing.<br><br>
            You received this because <strong>{ticker}</strong> is on your Tychain watchlist.
            Manage alerts from your <a href="{dashboard_link}" style="color:#29B6F6;">dashboard</a>.
          </td>
        </tr>
      </table>
    </td></tr>
  </table>
</body>
</html>"""

    text_body = (
        f"Tychain Signal Alert\n"
        f"{label} - {ticker} - strength {strength}%\n"
        f"Current price: {currency_symbol}{price_str}\n"
        f"Forecast (next day): {currency_symbol}{next_price_str} ({pct_sign}{pct_change}%)\n\n"
        f"{summary or ''}\n\n"
        f"Open dashboard: {dashboard_link}\n"
        f"View analysis: {analysis_link}\n\n"
        f"Tychain is an educational AI tool, not financial advice."
    )

    return send_email(to_email, to_name, subject, html_body, text_body=text_body)

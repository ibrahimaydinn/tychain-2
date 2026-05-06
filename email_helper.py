"""Tychain — Email helper.

Sends user-specific signal alerts via SMTP (Gmail / SES / Mailgun / Postmark /
local mailhog). All credentials are pulled from environment at call time:

    SMTP_HOST       e.g. smtp.gmail.com
    SMTP_PORT       465 (SMTPS) or 587 (STARTTLS) or 25 (plain)
    SMTP_USER       SMTP username
    SMTP_PASS       SMTP password / app password
    SMTP_TIMEOUT    seconds (default 20)
    MAIL_FROM       address used in the From header (default noreply@tychain.app)
    MAIL_NAME       display name used in the From header (default 'Tychain Alerts')
    APP_URL         base URL used to build dashboard links (default localhost)

When SMTP_HOST is not set the helper falls back to mock mode (logs the message
to stdout) so the dispatcher loop is safe to run on a fresh deploy.

Public API:
    send_email(to_email, to_name, subject, html_body, text_body=None) -> True | str
    email_signal_alert(to_email, to_name, ticker, signal_type, strength,
                       price, next_price, summary,
                       currency_symbol="$", dashboard_url=None) -> True | str
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


# ── Config (resolved at call time, not import time, so HF env edits apply) ────

def _cfg() -> dict:
    return {
        "MAIL_FROM":    os.environ.get("MAIL_FROM",  "noreply@tychain.app"),
        "MAIL_NAME":    os.environ.get("MAIL_NAME",  "Tychain Alerts"),
        "SMTP_HOST":    os.environ.get("SMTP_HOST",  ""),
        "SMTP_PORT":    int(os.environ.get("SMTP_PORT", 465)),
        "SMTP_USER":    os.environ.get("SMTP_USER",  ""),
        "SMTP_PASS":    os.environ.get("SMTP_PASS",  ""),
        "SMTP_TIMEOUT": int(os.environ.get("SMTP_TIMEOUT", 20)),
        "APP_URL":      os.environ.get("APP_URL",    "http://localhost:8080"),
    }


# ── Low-level send ─────────────────────────────────────────────────────────────

def _html_to_text(html: str) -> str:
    try:
        from bs4 import BeautifulSoup
        return BeautifulSoup(html, "html.parser").get_text(separator="\n").strip()
    except Exception:
        import re
        return re.sub(r"<[^>]+>", "", html).strip()


def send_email(to_email, to_name, subject, html_body, text_body=None):
    """Send a single email. Returns True on success, error-string on failure."""
    cfg = _cfg()

    if not cfg["SMTP_HOST"]:
        print(
            f"[email_helper] SMTP_HOST not set — mock-sending to {to_email}\n"
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
            server = smtplib.SMTP_SSL(cfg["SMTP_HOST"], cfg["SMTP_PORT"],
                                      timeout=cfg["SMTP_TIMEOUT"], context=ctx)
        else:
            server = smtplib.SMTP(cfg["SMTP_HOST"], cfg["SMTP_PORT"],
                                  timeout=cfg["SMTP_TIMEOUT"])
            server.ehlo()
            try:
                server.starttls(context=ctx)
                server.ehlo()
            except smtplib.SMTPNotSupportedError:
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


# ── Branded signal-alert email (corporate palette, no emojis) ─────────────────

_BUY_SIGNALS  = {"BUY", "STRONG BUY"}
_SELL_SIGNALS = {"SELL", "STRONG SELL"}

# Palette mirrors static/css/theme.css :root so the email looks like the app.
_PALETTE = {
    "bg":            "#0e1419",
    "bg_elevated":   "#161c24",
    "surface":       "#1c2530",
    "border":        "#262e38",
    "border_strong": "#2f3946",
    "text":          "#e6e9ed",
    "text_secondary":"#9aa6b3",
    "text_muted":    "#6b7785",
    "accent":        "#4682b4",
    "positive":      "#4a8264",
    "negative":      "#a74e4e",
}


def _signal_meta(signal_type: str):
    """Return (accent_color, headline) for the given signal."""
    s = (signal_type or "").upper()
    if s in _BUY_SIGNALS:
        return _PALETTE["positive"], s.title()
    if s in _SELL_SIGNALS:
        return _PALETTE["negative"], s.title()
    return _PALETTE["text_muted"], "Hold"


def _signal_message(signal_type: str, ticker: str) -> str:
    """User-facing prose for the signal. Matches the tone agreed with product."""
    s = (signal_type or "").upper()
    if s in _BUY_SIGNALS:
        return (
            f"A strong upward trend has been detected for {ticker} on your "
            f"dashboard. Please review the latest analysis on Tychain."
        )
    if s in _SELL_SIGNALS:
        return (
            f"Your watched stock {ticker} is expected to decrease according "
            f"to Tychain AI analysis. We advise you to be aware of this movement."
        )
    return (
        f"The model regime for {ticker} has shifted. Please review the latest "
        f"analysis on Tychain."
    )


def email_signal_alert(
    to_email,
    to_name,
    ticker,
    signal_type,
    strength,
    price,
    next_price,
    summary,
    currency_symbol="$",
    dashboard_url=None,
):
    """Send a branded signal-alert email to a watchlist subscriber."""
    cfg   = _cfg()
    color, label = _signal_meta(signal_type)
    p     = _PALETTE

    try:
        pct_change = round(((float(next_price) - float(price)) / float(price)) * 100, 2) if float(price) > 0 else 0.0
    except (TypeError, ValueError, ZeroDivisionError):
        pct_change = 0.0
    pct_sign = "+" if pct_change >= 0 else ""

    base = (dashboard_url or cfg["APP_URL"]).rstrip("/")
    dashboard_link = f"{base}/dashboard?ticker={ticker}"
    analysis_link  = f"{base}/?stock={ticker}"

    sent_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    message = _signal_message(signal_type, ticker)
    subject = f"Tychain — {label} signal on {ticker}"

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
<body style="margin:0;padding:0;background:{p['bg']};font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Inter',Arial,sans-serif;color:{p['text']};">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:{p['bg']};padding:32px 12px;">
    <tr><td align="center">
      <table role="presentation" width="600" cellpadding="0" cellspacing="0"
             style="max-width:600px;background:{p['surface']};border:1px solid {p['border']};border-radius:6px;">
        <tr>
          <td style="padding:24px 28px;border-bottom:1px solid {p['border']};">
            <div style="display:inline-block;width:18px;height:18px;background:{p['accent']};border-radius:3px;vertical-align:middle;margin-right:10px;"></div>
            <span style="font-size:15px;font-weight:600;color:{p['text']};letter-spacing:-0.01em;vertical-align:middle;">Tychain</span>
            <div style="font-size:11px;letter-spacing:0.12em;color:{p['text_muted']};text-transform:uppercase;margin-top:14px;">Signal alert</div>
            <h1 style="margin:6px 0 0;font-size:20px;font-weight:600;color:{p['text']};letter-spacing:-0.01em;">
              {label} signal on {ticker}
            </h1>
            <p style="margin:6px 0 0;color:{p['text_muted']};font-size:12px;">{sent_at}</p>
          </td>
        </tr>
        <tr>
          <td style="padding:24px 28px;">
            <p style="margin:0 0 18px;color:{p['text']};font-size:14px;line-height:1.65;">
              {message}
            </p>

            <table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;border:1px solid {p['border']};border-radius:4px;">
              <tr>
                <td style="padding:11px 14px;border-bottom:1px solid {p['border']};color:{p['text_muted']};font-size:11px;letter-spacing:0.05em;text-transform:uppercase;">Ticker</td>
                <td align="right" style="padding:11px 14px;border-bottom:1px solid {p['border']};color:{p['text']};font-weight:600;font-size:13px;">{ticker}</td>
              </tr>
              <tr>
                <td style="padding:11px 14px;border-bottom:1px solid {p['border']};color:{p['text_muted']};font-size:11px;letter-spacing:0.05em;text-transform:uppercase;">Signal</td>
                <td align="right" style="padding:11px 14px;border-bottom:1px solid {p['border']};color:{color};font-weight:600;font-size:13px;">{label}</td>
              </tr>
              <tr>
                <td style="padding:11px 14px;border-bottom:1px solid {p['border']};color:{p['text_muted']};font-size:11px;letter-spacing:0.05em;text-transform:uppercase;">Confidence</td>
                <td align="right" style="padding:11px 14px;border-bottom:1px solid {p['border']};color:{p['text']};font-weight:600;font-size:13px;font-variant-numeric:tabular-nums;">{strength}%</td>
              </tr>
              <tr>
                <td style="padding:11px 14px;border-bottom:1px solid {p['border']};color:{p['text_muted']};font-size:11px;letter-spacing:0.05em;text-transform:uppercase;">Last close</td>
                <td align="right" style="padding:11px 14px;border-bottom:1px solid {p['border']};color:{p['text']};font-weight:600;font-size:13px;font-variant-numeric:tabular-nums;">{currency_symbol}{price_str}</td>
              </tr>
              <tr>
                <td style="padding:11px 14px;color:{p['text_muted']};font-size:11px;letter-spacing:0.05em;text-transform:uppercase;">Forecast (next day)</td>
                <td align="right" style="padding:11px 14px;color:{p['text']};font-weight:600;font-size:13px;font-variant-numeric:tabular-nums;">
                  {currency_symbol}{next_price_str}
                  <span style="color:{color};font-weight:600;font-size:12px;">&nbsp;{pct_sign}{pct_change}%</span>
                </td>
              </tr>
            </table>

            <div style="background:{p['bg_elevated']};border-left:2px solid {p['border_strong']};padding:12px 14px;margin:20px 0;color:{p['text_secondary']};font-size:13px;line-height:1.6;">
              {summary or "Model summary unavailable."}
            </div>

            <a href="{dashboard_link}"
               style="display:inline-block;background:{p['accent']};color:#ffffff;text-decoration:none;
                      padding:9px 18px;border-radius:4px;font-weight:500;font-size:13px;border:1px solid {p['accent']};">
              Open dashboard
            </a>
            <p style="margin:14px 0 0;font-size:12px;color:{p['text_muted']};">
              Or jump directly to the <a href="{analysis_link}" style="color:{p['accent']};text-decoration:none;">{ticker} analysis</a>.
            </p>
          </td>
        </tr>
        <tr>
          <td style="padding:18px 28px;background:{p['bg_elevated']};border-top:1px solid {p['border']};font-size:11px;color:{p['text_muted']};line-height:1.6;">
            For educational purposes only. Tychain forecasts can be wrong; do your own research and never invest more than you can afford to lose.<br><br>
            You received this because <strong style="color:{p['text_secondary']};font-weight:600;">{ticker}</strong> is on your watchlist. Manage alerts from your <a href="{dashboard_link}" style="color:{p['accent']};text-decoration:none;">dashboard</a>.
          </td>
        </tr>
      </table>
    </td></tr>
  </table>
</body>
</html>"""

    text_body = (
        f"Tychain — Signal alert\n"
        f"{label} signal on {ticker} ({strength}% confidence)\n\n"
        f"{message}\n\n"
        f"Last close:           {currency_symbol}{price_str}\n"
        f"Forecast (next day):  {currency_symbol}{next_price_str} ({pct_sign}{pct_change}%)\n\n"
        f"{summary or ''}\n\n"
        f"Open dashboard: {dashboard_link}\n"
        f"View analysis:  {analysis_link}\n\n"
        f"Tychain is an educational tool, not financial advice."
    )

    return send_email(to_email, to_name, subject, html_body, text_body=text_body)

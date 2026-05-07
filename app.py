#!/usr/bin/env python3
"""Tychain — BIST30 Stock Forecasting (Flask web app)"""

import contextlib, json, os, secrets, sqlite3, subprocess, sys, uuid
from datetime import datetime, timedelta
from functools import wraps

from flask import (Flask, jsonify, redirect, render_template,
                   request, session, url_for)
from werkzeug.security import check_password_hash, generate_password_hash
import pandas as pd

import db_config  # Turso (libSQL) in production, local SQLite in dev

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT   = os.path.join(BASE_DIR, 'backend', 'train_model.py')
PYTHON   = sys.executable
# Local-dev fallback path. In production TURSO_DATABASE_URL is set and this is unused.
DB_PATH  = os.environ.get('TYCHAIN_DB_PATH', os.path.join(BASE_DIR, 'tychain.db'))

ALERT_THRESHOLD  = 60          # Send alert when signal strength ≥ this %
ALERT_SIGNAL_TYPES = {         # Only these signal types trigger an email
    'BUY', 'STRONG BUY', 'SELL', 'STRONG SELL',
}
SIGNAL_CACHE_MIN = 60
MAX_TRACKED      = 20
RATE_LIMIT_MAX   = 5
RATE_LIMIT_MIN   = 15

BIST30_STOCKS = {
    'AKBNK':'AKBNK - Akbank',        'ASELS':'ASELS - Aselsan',
    'BIMAS':'BIMAS - BIM',           'DOHOL':'DOHOL - Dogan Holding',
    'EKGYO':'EKGYO - Emlak Konut',   'ENKAI':'ENKAI - Enka Insaat',
    'EREGL':'EREGL - Eregli Demir',  'FROTO':'FROTO - Ford Otosan',
    'GARAN':'GARAN - Garanti BBVA',  'GUBRF':'GUBRF - Gubre Fabrikalari',
    'HALKB':'HALKB - Halkbank',      'ISCTR':'ISCTR - Is Bankasi',
    'KCHOL':'KCHOL - Koc Holding',   'KOZAA':'KOZAA - Koza Anadolu',
    'KOZAL':'KOZAL - Koza Altin',    'KRDMD':'KRDMD - Kardemir',
    'MGROS':'MGROS - Migros',        'ODAS' :'ODAS - Odas Elektrik',
    'PETKM':'PETKM - Petkim',        'PGSUS':'PGSUS - Pegasus',
    'SAHOL':'SAHOL - Sabanci Holding','SASA':'SASA - SASA Polyester',
    'SISE' :'SISE - Sisecam',        'TAVHL':'TAVHL - TAV Havalimanlari',
    'TCELL':'TCELL - Turkcell',      'THYAO':'THYAO - Turkish Airlines',
    'TKFEN':'TKFEN - Tekfen Holding','TOASO':'TOASO - Tofas',
    'TUPRS':'TUPRS - Tupras',        'ULKER':'ULKER - Ulker Biskuvi',
    'VAKBN':'VAKBN - Vakifbank',     'VESTL':'VESTL - Vestel',
    'YKBNK':'YKBNK - Yapi Kredi',
}

def load_sp500_tickers():
    cache_file = os.path.join(BASE_DIR, 'sp500_tickers.json')
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except:
            pass
    try:
        import requests
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        html = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', headers={'User-Agent': 'Mozilla/5.0'}, verify=False).text
        table = pd.read_html(html)[0]
        tickers = {str(row['Symbol']).replace('.', '-'): f"{str(row['Symbol']).replace('.', '-')} - {row['Security']}" for _, row in table.iterrows()}
        with open(cache_file, 'w') as f:
            json.dump(tickers, f)
        return tickers
    except Exception as e:
        print(f"Failed to fetch S&P 500: {e}")
        return {
            'AAPL': 'AAPL - Apple', 'MSFT': 'MSFT - Microsoft',
            'NVDA': 'NVDA - NVIDIA', 'GOOGL': 'GOOGL - Alphabet'
        }

SP500_STOCKS = load_sp500_tickers()
SP500_SECTORS = {'S&P 500': SP500_STOCKS} # Mock sector to keep index.html happy

STOCKS = {**BIST30_STOCKS, **SP500_STOCKS}

TICKER_BAR = [
    ('S&P 500','','up'),    ('NVDA','','up'),    ('GOOGL','','up'),
    ('META','','up'),       ('TSLA','','down'),  ('JPM','','up'),
    ('LLY','','up'),        ('V','','up'),       ('UNH','','down'),
    ('XOM','','up'),        ('MA','','up'),      ('HD','','up'),
]

SIGNAL_COLORS = {
    'STRONG BUY':'#66BB6A','BUY':'#A5D6A7','HOLD':'#FFF176',
    'SELL':'#EF9A9A','STRONG SELL':'#EF5350',
}
SIGNAL_EMOJIS = {
    'STRONG BUY':'🚀','BUY':'📈','HOLD':'⏸️','SELL':'📉','STRONG SELL':'🔴',
}

app = Flask(__name__)

# SECRET_KEY must be stable across workers and restarts. Without one,
# every gunicorn worker generates its own random key, which makes session
# cookies (and the CSRF token they hold) unreadable on cross-worker
# requests — surfacing as "Invalid Token" on signup/login.
_secret = os.environ.get('SECRET_KEY')
if not _secret:
    if db_config.using_turso():
        # Production (Turso configured) — fail loud rather than mint a
        # random per-worker key.
        raise RuntimeError(
            "SECRET_KEY environment variable is required in production. "
            "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\" "
            "and set it as a Hugging Face Space secret."
        )
    # Dev fallback — fine because Flask's dev server is single-process.
    _secret = secrets.token_hex(32)
    print("[app] WARNING: SECRET_KEY not set — using ephemeral key (dev only).")
app.secret_key = _secret

_in_production = bool(os.environ.get('SPACE_ID')) or db_config.using_turso()

# Force HTTPS scheme in production so Secure cookies are always emitted.
class ForceHTTPS:
    def __init__(self, app):
        self.app = app
    def __call__(self, environ, start_response):
        environ['wsgi.url_scheme'] = 'https'
        return self.app(environ, start_response)

if _in_production:
    app.wsgi_app = ForceHTTPS(app.wsgi_app)

# Session cookie settings.
app.config.update(
    SESSION_COOKIE_SAMESITE='None' if _in_production else 'Lax',
    SESSION_COOKIE_SECURE=_in_production,
    SECRET_KEY=os.environ.get('FLASK_SECRET_KEY', 'default_fallback_key'),
    SESSION_COOKIE_HTTPONLY=True,
    PERMANENT_SESSION_LIFETIME=timedelta(days=7),
)

# ── Database ───────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def get_db():
    """Yield a DB connection. Routes through db_config so the same code
    works against Turso (production) and local SQLite (dev)."""
    conn = db_config.get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass  # Some libsql modes don't support rollback after commit
        raise
    finally:
        conn.close()

@contextlib.contextmanager
def get_users_db():
    """Yield a connection specifically for the users database."""
    conn = db_config.get_users_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        conn.close()

_SCHEMA_STATEMENTS_USERS = [
    """CREATE TABLE IF NOT EXISTS users (
        id            TEXT     PRIMARY KEY,
        email         TEXT     UNIQUE NOT NULL,
        password_hash TEXT     NOT NULL,
        created_at    DATETIME DEFAULT CURRENT_TIMESTAMP
    )""",
    "CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)",
    """CREATE TABLE IF NOT EXISTS login_attempts (
        id         INTEGER  PRIMARY KEY AUTOINCREMENT,
        ip         TEXT     NOT NULL,
        email      TEXT     NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )"""
]

# Schema split as discrete statements: required because libsql may not
# expose executescript(). Works identically against sqlite3.
_SCHEMA_STATEMENTS = [
    """CREATE TABLE IF NOT EXISTS tracked_stocks (
        id         INTEGER  PRIMARY KEY AUTOINCREMENT,
        user_id    TEXT     NOT NULL,
        ticker     TEXT     NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(user_id, ticker)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_ts_user   ON tracked_stocks(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_ts_ticker ON tracked_stocks(ticker)",
    """CREATE TABLE IF NOT EXISTS signals (
        id              INTEGER  PRIMARY KEY AUTOINCREMENT,
        ticker          TEXT     NOT NULL,
        signal_type     TEXT     NOT NULL,
        signal_strength INTEGER  NOT NULL,
        score           INTEGER,
        last_price      REAL,
        next_day_price  REAL,
        price_change    REAL,
        rsi             REAL,
        trend           TEXT,
        hmm_label       TEXT,
        summary         TEXT,
        checked_at      DATETIME DEFAULT CURRENT_TIMESTAMP
    )""",
    "CREATE INDEX IF NOT EXISTS idx_sig_ticker ON signals(ticker)",
    """CREATE TABLE IF NOT EXISTS email_log (
        id              INTEGER  PRIMARY KEY AUTOINCREMENT,
        user_id         TEXT     NOT NULL,
        ticker          TEXT     NOT NULL,
        signal_type     TEXT     NOT NULL,
        signal_strength INTEGER  NOT NULL,
        sent_at         DATETIME DEFAULT CURRENT_TIMESTAMP
    )""",
    """CREATE TABLE IF NOT EXISTS performance_analytics (
        ticker TEXT PRIMARY KEY,
        market TEXT NOT NULL,
        price REAL,
        abs_1d REAL,
        pct_1d REAL,
        abs_1w REAL,
        pct_1w REAL,
        abs_1m REAL,
        pct_1m REAL,
        abs_1y REAL,
        pct_1y REAL,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )""",
    # ── Section: fetched market data (refreshed every 3 minutes by a
    # background thread). Holds the latest quote per ticker.
    """CREATE TABLE IF NOT EXISTS market_data (
        ticker      TEXT     PRIMARY KEY,
        last_price  REAL,
        prev_close  REAL,
        change_pct  REAL,
        volume      INTEGER,
        currency    TEXT,
        fetched_at  DATETIME DEFAULT CURRENT_TIMESTAMP
    )""",
    "CREATE INDEX IF NOT EXISTS idx_market_fetched ON market_data(fetched_at)",
    # ── Forum: posts (with optional base64 image) and comments. Logged-in
    # users only — enforced at the route layer.
    """CREATE TABLE IF NOT EXISTS forum_posts (
        id            INTEGER  PRIMARY KEY AUTOINCREMENT,
        user_id       TEXT     NOT NULL,
        author_email  TEXT     NOT NULL,
        title         TEXT     NOT NULL,
        body          TEXT     NOT NULL,
        image_b64     TEXT,
        image_mime    TEXT,
        created_at    DATETIME DEFAULT CURRENT_TIMESTAMP
    )""",
    "CREATE INDEX IF NOT EXISTS idx_forum_posts_created ON forum_posts(created_at DESC)",
    """CREATE TABLE IF NOT EXISTS forum_comments (
        id            INTEGER  PRIMARY KEY AUTOINCREMENT,
        post_id       INTEGER  NOT NULL,
        user_id       TEXT     NOT NULL,
        author_email  TEXT     NOT NULL,
        body          TEXT     NOT NULL,
        created_at    DATETIME DEFAULT CURRENT_TIMESTAMP
    )""",
    "CREATE INDEX IF NOT EXISTS idx_forum_comments_post ON forum_comments(post_id, created_at)",
]


def init_db():
    """Create tables if missing. Safe to run repeatedly. Used by both
    local SQLite (with WAL mode) and Turso."""
    conn = db_config.get_connection()
    try:
        # WAL is a no-op on libsql remote, but speeds up local dev hugely.
        if not db_config.using_turso():
            try:
                conn.execute("PRAGMA journal_mode = WAL")
            except Exception:
                pass
        for stmt in _SCHEMA_STATEMENTS:
            conn.execute(stmt)
        conn.commit()
    finally:
        conn.close()

    uconn = db_config.get_users_connection()
    try:
        if not db_config.using_turso():
            try:
                uconn.execute("PRAGMA journal_mode = WAL")
            except Exception:
                pass
        for stmt in _SCHEMA_STATEMENTS_USERS:
            uconn.execute(stmt)
        uconn.commit()
    finally:
        uconn.close()

def db_create_user(email, password):
    user_id = str(uuid.uuid4())
    try:
        with get_users_db() as conn:
            conn.execute(
                "INSERT INTO users (id, email, password_hash) VALUES (?, ?, ?)",
                (user_id, email, generate_password_hash(password))
            )
            return user_id
    except sqlite3.IntegrityError:
        return None
    except Exception as e:
        # libSQL surfaces UNIQUE violations as a generic error string;
        # fall through only for that specific case so we don't swallow real bugs.
        msg = str(e).lower()
        if 'unique' in msg or 'constraint' in msg:
            return None
        raise

def db_find_user_by_email(email):
    with get_users_db() as conn:
        cur = conn.execute("SELECT * FROM users WHERE email = ?", (email,))
        return db_config.dict_row(cur)

def db_find_user_by_id(user_id):
    with get_users_db() as conn:
        cur = conn.execute(
            "SELECT id, email, created_at FROM users WHERE id = ?", (user_id,)
        )
        return db_config.dict_row(cur)

def db_add_tracked(user_id, ticker):
    try:
        with get_db() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO tracked_stocks (user_id, ticker) VALUES (?, ?)",
                (user_id, ticker.upper())
            )
        return True
    except Exception:
        return False

def db_remove_tracked(user_id, ticker):
    with get_db() as conn:
        conn.execute(
            "DELETE FROM tracked_stocks WHERE user_id = ? AND ticker = ?",
            (user_id, ticker.upper())
        )

def db_get_tracked(user_id):
    with get_db() as conn:
        cur = conn.execute(
            "SELECT ticker, created_at FROM tracked_stocks WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,)
        )
        return db_config.dict_rows(cur)

def db_count_tracked(user_id):
    with get_db() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM tracked_stocks WHERE user_id = ?", (user_id,)
        ).fetchone()[0]

def db_save_signal(d):
    with get_db() as conn:
        conn.execute("DELETE FROM signals WHERE ticker = ?", (d['ticker'],))
        conn.execute("""
            INSERT INTO signals
                (ticker, signal_type, signal_strength, score, last_price,
                 next_day_price, price_change, rsi, trend, hmm_label, summary)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, (
            d['ticker'], d['signal_type'], d['signal_strength'],
            d.get('score'), d.get('last_price'), d.get('next_day_price'),
            d.get('price_change'), d.get('rsi'), d.get('trend'),
            d.get('hmm_label'), d.get('summary'),
        ))

def db_get_signal(ticker):
    with get_db() as conn:
        cur = conn.execute(
            "SELECT * FROM signals WHERE ticker = ? LIMIT 1", (ticker.upper(),)
        )
        return db_config.dict_row(cur)

def db_signal_is_fresh(ticker):
    sig = db_get_signal(ticker)
    if not sig:
        return False
    try:
        checked = datetime.fromisoformat(sig['checked_at'])
        return (datetime.utcnow() - checked).total_seconds() / 60 < SIGNAL_CACHE_MIN
    except Exception:
        return False

def db_rate_limited(ip, email):
    with get_users_db() as conn:
        conn.execute(
            "DELETE FROM login_attempts WHERE created_at < datetime('now', ?)",
            (f'-{RATE_LIMIT_MIN} minutes',)
        )
        count = conn.execute(
            "SELECT COUNT(*) FROM login_attempts WHERE ip = ? AND email = ?",
            (ip, email)
        ).fetchone()[0]
        return count >= RATE_LIMIT_MAX

def db_record_attempt(ip, email):
    with get_users_db() as conn:
        conn.execute(
            "INSERT INTO login_attempts (ip, email) VALUES (?, ?)", (ip, email)
        )

def db_all_unique_tickers():
    """List of distinct tickers anyone has on a watchlist. Portable across
    sqlite3.Row and libsql plain-tuple cursors — never use named-key access
    on rows here."""
    with get_db() as conn:
        rows = conn.execute("SELECT DISTINCT ticker FROM tracked_stocks").fetchall()
        return [(r['ticker'] if hasattr(r, 'keys') else r[0]) for r in rows]

def db_users_tracking(ticker):
    with get_db() as conn:
        rows = conn.execute("SELECT user_id FROM tracked_stocks WHERE ticker = ?", (ticker.upper(),)).fetchall()
        user_ids = [r['user_id'] if hasattr(r, 'keys') else r[0] for r in rows]
    
    if not user_ids:
        return []

    with get_users_db() as uconn:
        placeholders = ','.join('?' * len(user_ids))
        cur = uconn.execute(f"SELECT id, email FROM users WHERE id IN ({placeholders})", tuple(user_ids))
        return db_config.dict_rows(cur)

def db_already_notified(user_id, ticker, signal_type):
    """Cooldown gate: True if (user_id, ticker, signal_type) was emailed in
    the last 12 hours. Prevents the 60-second dispatcher from re-emailing
    every tick while a signal stays above threshold. A signal change
    (BUY → SELL or vice-versa) is treated as a new event because
    signal_type differs."""
    with get_db() as conn:
        count = conn.execute("""
            SELECT COUNT(*) FROM email_log
            WHERE user_id     = ?
              AND ticker      = ?
              AND signal_type = ?
              AND sent_at     > datetime('now', '-12 hours')
        """, (user_id, ticker.upper(), signal_type)).fetchone()[0]
        return count > 0

def db_log_email(user_id, ticker, signal_type, signal_strength):
    with get_db() as conn:
        conn.execute("""
            INSERT INTO email_log (user_id, ticker, signal_type, signal_strength)
            VALUES (?, ?, ?, ?)
        """, (user_id, ticker.upper(), signal_type, signal_strength))

# ── Market data (3-minute refresh) ─────────────────────────────────────────────

def db_upsert_market(ticker, last_price, prev_close, change_pct, volume, currency):
    """Idempotent write of one quote into market_data."""
    with get_db() as conn:
        conn.execute("""
            INSERT INTO market_data (ticker, last_price, prev_close, change_pct,
                                     volume, currency, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(ticker) DO UPDATE SET
              last_price = excluded.last_price,
              prev_close = excluded.prev_close,
              change_pct = excluded.change_pct,
              volume     = excluded.volume,
              currency   = excluded.currency,
              fetched_at = CURRENT_TIMESTAMP
        """, (ticker.upper(), last_price, prev_close, change_pct, volume, currency))

def db_get_market_latest(limit=12):
    """Return the most recently refreshed quotes for the live ticker bar."""
    with get_db() as conn:
        cur = conn.execute(
            "SELECT * FROM market_data ORDER BY fetched_at DESC LIMIT ?",
            (int(limit),)
        )
        return db_config.dict_rows(cur)

def db_get_market_one(ticker):
    with get_db() as conn:
        cur = conn.execute("SELECT * FROM market_data WHERE ticker = ?", (ticker.upper(),))
        return db_config.dict_row(cur)

# ── Forum ──────────────────────────────────────────────────────────────────────

MAX_POST_BYTES  = 12 * 1024            # plain-text body cap
MAX_IMAGE_BYTES = 1_500_000            # ~1.5 MB raw upload
ALLOWED_IMG_MIMES = {'image/png', 'image/jpeg', 'image/gif', 'image/webp'}

def db_create_post(user_id, author_email, title, body, image_b64=None, image_mime=None):
    with get_db() as conn:
        cur = conn.execute("""
            INSERT INTO forum_posts (user_id, author_email, title, body, image_b64, image_mime)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (user_id, author_email, title, body, image_b64, image_mime))
        # libsql exposes lastrowid via the cursor on SQLite path; on Turso we
        # round-trip a quick read to be safe across drivers.
        try:
            return int(cur.lastrowid)
        except Exception:
            row = conn.execute(
                "SELECT id FROM forum_posts WHERE user_id = ? ORDER BY id DESC LIMIT 1",
                (user_id,)
            ).fetchone()
            return int(row[0] if not hasattr(row, 'keys') else row['id'])

def db_list_posts(limit=50, offset=0):
    with get_db() as conn:
        cur = conn.execute("""
            SELECT id, author_email, title, body, image_mime, created_at,
                   (SELECT COUNT(*) FROM forum_comments c WHERE c.post_id = forum_posts.id) AS comment_count,
                   (image_b64 IS NOT NULL) AS has_image
            FROM forum_posts
            ORDER BY created_at DESC, id DESC
            LIMIT ? OFFSET ?
        """, (int(limit), int(offset)))
        return db_config.dict_rows(cur)

def db_get_post(post_id):
    with get_db() as conn:
        cur = conn.execute("SELECT * FROM forum_posts WHERE id = ?", (int(post_id),))
        return db_config.dict_row(cur)

def db_get_post_image(post_id):
    """Returns (b64, mime) or (None, None)."""
    with get_db() as conn:
        cur = conn.execute(
            "SELECT image_b64, image_mime FROM forum_posts WHERE id = ?",
            (int(post_id),)
        )
        row = db_config.dict_row(cur)
        if not row:
            return None, None
        return row.get('image_b64'), row.get('image_mime')

def db_list_comments(post_id):
    with get_db() as conn:
        cur = conn.execute("""
            SELECT id, author_email, body, created_at
            FROM forum_comments WHERE post_id = ?
            ORDER BY created_at ASC, id ASC
        """, (int(post_id),))
        return db_config.dict_rows(cur)

def db_create_comment(post_id, user_id, author_email, body):
    with get_db() as conn:
        conn.execute("""
            INSERT INTO forum_comments (post_id, user_id, author_email, body)
            VALUES (?, ?, ?, ?)
        """, (int(post_id), user_id, author_email, body))

# ── Jinja2 filters ─────────────────────────────────────────────────────────────

@app.template_filter('time_ago')
def time_ago(dt_str):
    try:
        diff = (datetime.utcnow() - datetime.fromisoformat(str(dt_str))).total_seconds()
        if diff < 60:    return 'Just now'
        if diff < 3600:  return f'{int(diff / 60)}m ago'
        if diff < 86400: return f'{int(diff / 3600)}h ago'
        return f'{int(diff / 86400)}d ago'
    except Exception:
        return str(dt_str)

@app.template_filter('format_date')
def format_date(dt_str):
    try:
        return datetime.fromisoformat(str(dt_str)).strftime('%d %b %Y')
    except Exception:
        return str(dt_str)

@app.template_filter('numfmt')
def numfmt(value, decimals=2):
    try:
        return f'{float(value):,.{decimals}f}'
    except (TypeError, ValueError):
        return '--'

@app.template_filter('signal_color')
def signal_color(signal_type):
    for k, v in SIGNAL_COLORS.items():
        if k in (signal_type or ''):
            return v
    return '#78909C'

@app.template_filter('signal_emoji')
def signal_emoji(signal_type):
    for k, v in SIGNAL_EMOJIS.items():
        if k in (signal_type or ''):
            return v
    return '❓'

# ── Helpers ────────────────────────────────────────────────────────────────────

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('auth'))
        return f(*args, **kwargs)
    return decorated

def get_csrf():
    if 'csrf' not in session:
        session['csrf'] = secrets.token_hex(16)
    return session['csrf']

def check_csrf(token):
    return secrets.compare_digest(token or '', session.get('csrf', ''))

def run_analysis(symbol):
    try:
        if symbol in BIST30_STOCKS:
            script_path = SCRIPT
            arg = symbol + '.IS'
        else:
            script_path = os.path.join(BASE_DIR, 'backend', 'train_model_sp500.py')
            arg = symbol

        result = subprocess.run(
            [PYTHON, script_path, arg],
            capture_output=True, text=True, timeout=360, cwd=BASE_DIR
        )
        output = result.stdout
        j0, j1 = output.find('{'), output.rfind('}')
        if j0 == -1 or j1 == -1:
            return {'error': f'No output from model. stderr: {result.stderr[-400:]}'}
        
        data = json.loads(output[j0:j1 + 1])
        data['currency'] = 'TRY' if symbol in BIST30_STOCKS else 'USD'
        data['currency_symbol'] = '₺' if symbol in BIST30_STOCKS else '$'
        return data
    except subprocess.TimeoutExpired:
        return {'error': 'Analysis timed out (>6 min). Please try again.'}
    except json.JSONDecodeError as e:
        return {'error': f'JSON parse error: {e}'}
    except Exception as e:
        return {'error': str(e)}

def persist_signal(symbol, data):
    db_save_signal({
        'ticker':          symbol,
        'signal_type':     data['signal']['action'],
        'signal_strength': data['signal']['strength'],
        'score':           data['signal'].get('score'),
        'last_price':      data.get('last_price'),
        'next_day_price':  data.get('next_day_price'),
        'price_change':    data.get('price_change'),
        'rsi':             data['signal'].get('rsi'),
        'trend':           data['signal'].get('trend'),
        'hmm_label':       data['signal'].get('hmm_label'),
        'summary':         data['signal'].get('summary'),
    })

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html',
        logged_in=('user_id' in session),
        user_email=session.get('email', ''),
        csrf=get_csrf(),
        bist30_stocks=BIST30_STOCKS,
        sp500_stocks=SP500_STOCKS,
        sp500_sectors=SP500_SECTORS,
        ticker_bar=TICKER_BAR,
        stock_param=request.args.get('stock', ''),
    )

@app.route('/auth')
def auth():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('auth.html',
        tab=request.args.get('tab', 'login'),
        csrf=get_csrf(),
    )

@app.route('/dashboard')
@login_required
def dashboard():
    uid     = session['user_id']
    tracked = db_get_tracked(uid)
    
    tracked_bist30 = []
    tracked_sp500 = []
    signals_data = {}
    for t in tracked:
        sig = db_get_signal(t['ticker'])
        if t['ticker'] in BIST30_STOCKS:
            tracked_bist30.append(t)
            if sig:
                sig['currency'] = 'TRY'
                sig['currency_symbol'] = '₺'
        else:
            tracked_sp500.append(t)
            if sig:
                sig['currency'] = 'USD'
                sig['currency_symbol'] = '$'
        signals_data[t['ticker']] = sig

    return render_template('dashboard.html',
        user=db_find_user_by_id(uid),
        tracked=tracked,
        tracked_bist30=tracked_bist30,
        tracked_sp500=tracked_sp500,
        tracked_tickers={t['ticker'] for t in tracked},
        signals=signals_data,
        bist30_stocks=BIST30_STOCKS,
        sp500_stocks=SP500_STOCKS,
        sp500_sectors=SP500_SECTORS,
        csrf=get_csrf(),
        alert_threshold=ALERT_THRESHOLD,
        max_tracked=MAX_TRACKED,
    )

@app.route('/logout', methods=['GET', 'POST'])
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/market-summary')
def market_summary():
    with get_db() as conn:
        bist_gainers  = db_config.dict_rows(conn.execute("SELECT * FROM performance_analytics WHERE market='BIST30' ORDER BY pct_1d DESC LIMIT 10"))
        bist_losers   = db_config.dict_rows(conn.execute("SELECT * FROM performance_analytics WHERE market='BIST30' ORDER BY pct_1d ASC LIMIT 10"))
        sp500_gainers = db_config.dict_rows(conn.execute("SELECT * FROM performance_analytics WHERE market='SP500' ORDER BY pct_1d DESC LIMIT 10"))
        sp500_losers  = db_config.dict_rows(conn.execute("SELECT * FROM performance_analytics WHERE market='SP500' ORDER BY pct_1d ASC LIMIT 10"))

    return render_template('market-summary.html',
        logged_in=('user_id' in session),
        user_email=session.get('email', ''),
        bist_gainers=bist_gainers,
        bist_losers=bist_losers,
        sp500_gainers=sp500_gainers,
        sp500_losers=sp500_losers,
    )

@app.route('/performance')
def performance():
    with get_db() as conn:
        bist_perf  = db_config.dict_rows(conn.execute("SELECT * FROM performance_analytics WHERE market='BIST30' ORDER BY ticker ASC"))
        sp500_perf = db_config.dict_rows(conn.execute("SELECT * FROM performance_analytics WHERE market='SP500' ORDER BY ticker ASC"))

    return render_template('performance.html',
        logged_in=('user_id' in session),
        user_email=session.get('email', ''),
        bist_perf=bist_perf,
        sp500_perf=sp500_perf,
    )

# ── Forum routes ───────────────────────────────────────────────────────────────

@app.route('/forum')
def forum():
    posts = db_list_posts(limit=50)
    return render_template('forum.html',
        posts=posts,
        logged_in=('user_id' in session),
        user_email=session.get('email', ''),
        csrf=get_csrf(),
    )

@app.route('/forum/new', methods=['GET', 'POST'])
@login_required
def forum_new():
    if request.method == 'POST':
        if not check_csrf(request.form.get('csrf')):
            return render_template('forum_new.html',
                error='Session expired — please refresh and try again.',
                csrf=get_csrf(), logged_in=True,
                user_email=session.get('email', ''),
                title=request.form.get('title', ''),
                body=request.form.get('body', '')), 403

        title = (request.form.get('title') or '').strip()[:200]
        body  = (request.form.get('body')  or '').strip()
        if not title or not body:
            return render_template('forum_new.html',
                error='Title and body are required.',
                csrf=get_csrf(), logged_in=True,
                user_email=session.get('email', ''),
                title=title, body=body), 400
        if len(body.encode('utf-8')) > MAX_POST_BYTES:
            return render_template('forum_new.html',
                error=f'Post body is too long (max {MAX_POST_BYTES} bytes).',
                csrf=get_csrf(), logged_in=True,
                user_email=session.get('email', ''),
                title=title, body=body), 400

        image_b64 = None
        image_mime = None
        f = request.files.get('image')
        if f and f.filename:
            blob = f.read()
            if len(blob) > MAX_IMAGE_BYTES:
                return render_template('forum_new.html',
                    error=f'Image is too large (max {MAX_IMAGE_BYTES // 1024} KB).',
                    csrf=get_csrf(), logged_in=True,
                    user_email=session.get('email', ''),
                    title=title, body=body), 400
            mime = (f.mimetype or '').lower()
            if mime not in ALLOWED_IMG_MIMES:
                return render_template('forum_new.html',
                    error='Image must be PNG, JPG, GIF, or WebP.',
                    csrf=get_csrf(), logged_in=True,
                    user_email=session.get('email', ''),
                    title=title, body=body), 400
            import base64 as _b64
            image_b64 = _b64.b64encode(blob).decode('ascii')
            image_mime = mime

        try:
            new_id = db_create_post(
                session['user_id'], session.get('email', ''),
                title, body, image_b64, image_mime,
            )
        except Exception as e:
            return render_template('forum_new.html',
                error=f'Database error: {e}',
                csrf=get_csrf(), logged_in=True,
                user_email=session.get('email', ''),
                title=title, body=body), 500
        return redirect(url_for('forum_post', post_id=new_id))

    # GET
    return render_template('forum_new.html',
        csrf=get_csrf(), logged_in=True,
        user_email=session.get('email', ''),
        title='', body='', error=None,
    )

@app.route('/forum/<int:post_id>')
def forum_post(post_id):
    post = db_get_post(post_id)
    if not post:
        return redirect(url_for('forum'))
    comments = db_list_comments(post_id)
    return render_template('forum_post.html',
        post=post,
        comments=comments,
        logged_in=('user_id' in session),
        user_email=session.get('email', ''),
        csrf=get_csrf(),
    )

@app.route('/forum/<int:post_id>/image')
def forum_post_image(post_id):
    b64, mime = db_get_post_image(post_id)
    if not b64 or not mime:
        return ('', 404)
    import base64 as _b64
    from flask import Response
    try:
        blob = _b64.b64decode(b64)
    except Exception:
        return ('', 500)
    return Response(blob, mimetype=mime, headers={
        'Cache-Control': 'public, max-age=86400',
    })

@app.route('/forum/<int:post_id>/comment', methods=['POST'])
@login_required
def forum_comment(post_id):
    if not check_csrf(request.form.get('csrf')):
        return jsonify({'ok': False, 'error': 'Session expired — please refresh.'}), 403
    body = (request.form.get('body') or '').strip()
    if not body:
        return jsonify({'ok': False, 'error': 'Comment cannot be empty.'}), 400
    if len(body.encode('utf-8')) > MAX_POST_BYTES:
        return jsonify({'ok': False, 'error': 'Comment is too long.'}), 400
    if not db_get_post(post_id):
        return jsonify({'ok': False, 'error': 'Post not found.'}), 404
    try:
        db_create_comment(post_id, session['user_id'], session.get('email', ''), body)
    except Exception as e:
        return jsonify({'ok': False, 'error': f'Database error: {e}'}), 500
    return redirect(url_for('forum_post', post_id=post_id))

# ── API ────────────────────────────────────────────────────────────────────────

@app.route('/api/quotes')
def api_quotes():
    """Public read of the live market_data table — used to power the
    ticker bar on the landing page (auto-refreshes via JS)."""
    rows = db_get_market_latest(limit=24)
    return jsonify({
        'ok': True,
        'count': len(rows),
        'quotes': rows,
    })

@app.route('/api/analyze')
def api_analyze():
    symbol = request.args.get('symbol', '').upper()
    if not symbol or symbol not in STOCKS:
        return jsonify({'error': 'Invalid symbol'}), 400
    data = run_analysis(symbol)
    if 'error' not in data:
        persist_signal(symbol, data)
    return jsonify(data)

@app.route('/api/auth', methods=['POST'])
def api_auth_route():
    action = request.form.get('action')
    form_csrf = request.form.get('csrf')
    sess_csrf = session.get('csrf')

    # Skip CSRF on iframed loads where Safari/Firefox ITP strips the session
    # cookie. Signup/login are unauthenticated endpoints, so CSRF is largely
    # security theatre on them — there's no logged-in state to forge yet.
    # Once a real session exists (from /auth GET), enforce CSRF normally.
    if sess_csrf and not check_csrf(form_csrf):
        return jsonify({'ok': False, 'error': 'Session expired — please refresh and try again.'}), 403

    if action == 'login':
        email = request.form.get('email', '').lower().strip()
        pw    = request.form.get('password', '')
        ip    = request.remote_addr
        try:
            if db_rate_limited(ip, email):
                return jsonify({'ok': False, 'error': 'Too many attempts. Try again later.'})
            user = db_find_user_by_email(email)
            if not user or not check_password_hash(user['password_hash'], pw):
                db_record_attempt(ip, email)
                return jsonify({'ok': False, 'error': 'Invalid email or password.'})
            session['user_id'] = user['id']
            session['email']   = user['email']
            return jsonify({'ok': True, 'redirect': url_for('dashboard')})
        except Exception as e:
            return jsonify({'ok': False, 'error': f'Database connection failed: {str(e)}'}), 500

    if action == 'signup':
        email = request.form.get('email', '').lower().strip()
        pw    = request.form.get('password', '')
        pw2   = request.form.get('password2', '')
        if not email or not pw:
            return jsonify({'ok': False, 'error': 'Email and password are required.'})
        if len(pw) < 8:
            return jsonify({'ok': False, 'error': 'Password must be at least 8 characters.'})
        if pw != pw2:
            return jsonify({'ok': False, 'error': 'Passwords do not match.'})
        try:
            uid = db_create_user(email, pw)
            if uid is None:
                return jsonify({'ok': False, 'error': 'An account with this email already exists.'})
            session['user_id'] = uid
            session['email']   = email
            return jsonify({'ok': True, 'redirect': url_for('dashboard')})
        except Exception as e:
            return jsonify({'ok': False, 'error': f'Database connection failed: {str(e)}'}), 500

    return jsonify({'ok': False, 'error': 'Unknown action'}), 400

@app.route('/api/tracked', methods=['POST'])
@login_required
def api_tracked():
    uid    = session['user_id']
    action = request.form.get('action')
    if not check_csrf(request.form.get('csrf')):
        # Almost always a session/cookie issue: SECRET_KEY mismatch across
        # workers, browser blocking cookies, or a stale form left open
        # across a server restart. Tell the user to refresh.
        return jsonify({'ok': False, 'error': 'Session expired — please refresh the page and try again.'}), 403

    if action == 'add':
        ticker = request.form.get('ticker', '').upper()
        if ticker not in STOCKS:
            return jsonify({'ok': False, 'error': 'Invalid ticker'})
        if db_count_tracked(uid) >= MAX_TRACKED:
            return jsonify({'ok': False, 'error': f'Max {MAX_TRACKED} stocks allowed'})
        db_add_tracked(uid, ticker)
        return jsonify({'ok': True})

    if action == 'remove':
        ticker = request.form.get('ticker', '').upper()
        db_remove_tracked(uid, ticker)
        return jsonify({'ok': True})

    if action == 'refresh':
        ticker = request.form.get('ticker', '').upper()
        if ticker not in STOCKS:
            return jsonify({'ok': False, 'error': 'Invalid ticker'})
        data = run_analysis(ticker)
        if 'error' in data:
            return jsonify({'ok': False, 'error': data['error']})
        persist_signal(ticker, data)
        act = data['signal']['action']
        return jsonify({
            'ok': True,
            'signal': {
                'action':     act,
                'strength':   data['signal']['strength'],
                'color':      SIGNAL_COLORS.get(act, '#78909C'),
                'emoji':      SIGNAL_EMOJIS.get(act, '❓'),
                'last_price': f"{data.get('last_price', 0):.2f}",
                'pch':        round(data.get('price_change', 0), 2),
                'rsi':        round(data['signal'].get('rsi', 0), 1),
                'trend':      data['signal'].get('trend', ''),
                'currency':   data.get('currency', 'TRY'),
                'currency_symbol': data.get('currency_symbol', '₺'),
            }
        })

    if action == 'cron':
        import cron_signals
        result = cron_signals.run_cron()
        return jsonify({
            'ok': True,
            'message': f"Scanned {result['scanned']}, Sent {result['alerts_sent']} alerts"
        })

    return jsonify({'ok': False, 'error': 'Unknown action'}), 400

# ── Background quote refresher (every 3 minutes) ──────────────────────────────

# Tickers we keep fresh in market_data. Trimmed to keep yfinance happy on
# the free HF Space CPU. Adjust to taste.
LIVE_TICKERS = [
    'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA', 'JPM',
    'V', 'UNH', 'XOM', 'MA', 'LLY', 'AVGO', 'HD',
]

REFRESH_INTERVAL_SECONDS = 180  # 3 minutes


def _refresh_quotes_once():
    """Pull last quote for each LIVE_TICKERS symbol via yfinance and upsert."""
    try:
        import yfinance as yf
    except Exception as e:
        print(f"[refresher] yfinance unavailable: {e}")
        return

    for symbol in LIVE_TICKERS:
        try:
            t = yf.Ticker(symbol)
            # 2-row history is the cheapest way to compute change vs. previous close.
            hist = t.history(period='2d', interval='1d')
            if hist is None or hist.empty:
                continue
            last_price = float(hist['Close'].iloc[-1])
            prev_close = float(hist['Close'].iloc[-2]) if len(hist) >= 2 else last_price
            change_pct = ((last_price - prev_close) / prev_close * 100.0) if prev_close else 0.0
            volume = int(hist['Volume'].iloc[-1]) if 'Volume' in hist.columns else 0
            db_upsert_market(symbol, last_price, prev_close, change_pct, volume, 'USD')
        except Exception as e:
            print(f"[refresher] {symbol} failed: {e}")


def _quote_refresher_loop():
    import time
    # Stagger initial run a few seconds so the web server can come up first.
    time.sleep(5)
    while True:
        try:
            _refresh_quotes_once()
        except Exception as e:
            print(f"[refresher] loop error: {e}")
        time.sleep(REFRESH_INTERVAL_SECONDS)


def start_quote_refresher():
    """Start the background thread once per process. A file-based lock
    ensures only one gunicorn worker actually does the fetch even if the
    Space ever scales up to multiple workers."""
    import threading
    import tempfile
    lock_path = os.path.join(tempfile.gettempdir(), 'tychain_refresher.lock')
    try:
        # If another worker grabbed the lock first, skip.
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
    except FileExistsError:
        # Another worker already runs the refresher in this Space.
        return
    thread = threading.Thread(target=_quote_refresher_loop, daemon=True, name='quote-refresher')
    thread.start()
    print("[refresher] started — refreshing every", REFRESH_INTERVAL_SECONDS, "seconds")


# ── Performance fetcher (every 30 min) ────────────────────────────────────────
# Powers /market-summary and /performance. Pulls 2y daily history for every
# BIST30 + S&P 500 ticker, computes 1D/1W/1M/1Y movements, and bulk-writes
# them into the performance_analytics table.

PERFORMANCE_REFRESH_SECONDS = 30 * 60   # 30 minutes
PERFORMANCE_BATCH_SIZE      = 50         # tickers per yfinance request


def _performance_fetch_once():
    """Single pass: fetch quotes for every ticker, recompute the
    performance_analytics table. Idempotent. Errors on individual tickers
    are logged but do not abort the run."""
    try:
        import yfinance as yf
    except Exception as e:
        print(f"[perf] yfinance unavailable: {e}")
        return

    bist_tickers  = list(BIST30_STOCKS.keys())
    sp500_tickers = list(SP500_STOCKS.keys())
    yf_bist  = [t + '.IS' for t in bist_tickers]
    yf_sp500 = sp500_tickers
    all_tickers = yf_bist + yf_sp500

    print(f"[perf] fetch starting for {len(all_tickers)} tickers")

    # Batch the fetches so a single bad symbol can't poison the whole call,
    # and so memory stays bounded on the free HF CPU.
    records = []
    skipped = 0
    for i in range(0, len(all_tickers), PERFORMANCE_BATCH_SIZE):
        chunk = all_tickers[i:i + PERFORMANCE_BATCH_SIZE]
        try:
            data = yf.download(
                " ".join(chunk),
                period="2y",
                interval="1d",
                group_by="ticker",
                threads=True,
                progress=False,
                auto_adjust=True,
            )
        except Exception as e:
            print(f"[perf] batch {i}-{i+len(chunk)} failed: {e}")
            skipped += len(chunk)
            continue

        for yf_tick in chunk:
            is_bist   = yf_tick.endswith('.IS')
            base_tick = yf_tick[:-3] if is_bist else yf_tick
            market    = 'BIST30' if is_bist else 'SP500'
            try:
                if len(chunk) == 1:
                    df = data
                else:
                    if yf_tick not in data.columns.get_level_values(0):
                        skipped += 1
                        continue
                    df = data[yf_tick]
                df = df.dropna(subset=['Close'])
                if len(df) < 2:
                    skipped += 1
                    continue

                last_price = float(df['Close'].iloc[-1])

                def chg(days):
                    if len(df) <= 1:
                        return 0.0, 0.0
                    if len(df) > days:
                        past = float(df['Close'].iloc[-days - 1])
                    else:
                        past = float(df['Close'].iloc[0])
                    if past == 0:
                        return 0.0, 0.0
                    a = last_price - past
                    return a, (a / past) * 100.0

                abs_1d, pct_1d = chg(1)
                abs_1w, pct_1w = chg(5)
                abs_1m, pct_1m = chg(21)
                abs_1y, pct_1y = chg(252)

                records.append((
                    base_tick, market, last_price,
                    abs_1d, pct_1d,
                    abs_1w, pct_1w,
                    abs_1m, pct_1m,
                    abs_1y, pct_1y,
                ))
            except Exception as e:
                skipped += 1
                print(f"[perf] {yf_tick} failed: {e}")

    if not records:
        print(f"[perf] no rows produced (skipped {skipped}); leaving table untouched")
        return

    # libsql doesn't always implement executemany the way sqlite3 does, so
    # use a per-row execute inside one transaction to stay portable.
    try:
        with get_db() as conn:
            conn.execute("DELETE FROM performance_analytics")
            for row in records:
                conn.execute("""
                    INSERT INTO performance_analytics
                      (ticker, market, price, abs_1d, pct_1d,
                       abs_1w, pct_1w, abs_1m, pct_1m, abs_1y, pct_1y)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, row)
        print(f"[perf] wrote {len(records)} rows (skipped {skipped})")
    except Exception as e:
        print(f"[perf] DB write failed: {e}")


def _performance_refresher_loop():
    import time
    # Stagger initial run so the quote refresher and the web server settle first.
    time.sleep(15)
    while True:
        try:
            _performance_fetch_once()
        except Exception as e:
            print(f"[perf] loop error: {e}")
        time.sleep(PERFORMANCE_REFRESH_SECONDS)


def start_performance_refresher():
    """Same one-thread-per-Space file-lock pattern as the quote refresher."""
    import threading
    import tempfile
    lock_path = os.path.join(tempfile.gettempdir(), 'tychain_perf_refresher.lock')
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
    except FileExistsError:
        return
    thread = threading.Thread(target=_performance_refresher_loop,
                              daemon=True, name='perf-refresher')
    thread.start()
    print("[perf] started — refreshing every", PERFORMANCE_REFRESH_SECONDS, "seconds")


# ── Email alert dispatcher (every 60 seconds) ──────────────────────────────────
# Watches the cached `signals` table for tickers that any user is tracking,
# and dispatches one email per (user, ticker, signal_type) when:
#   • signal action ∈ {BUY, STRONG BUY, SELL, STRONG SELL}
#   • signal strength ≥ ALERT_THRESHOLD (60%)
#   • the same (user, ticker, signal_type) hasn't been emailed in the last 12h
#
# The dispatcher reads cached signals only — it does NOT re-train the LSTM
# (that's a 2–5 minute job per ticker and would crush the free HF CPU). The
# `signals` table is filled by the user clicking Analyze, by /api/tracked
# action=refresh, or by the manual /api/tracked action=cron path.

ALERT_DISPATCH_SECONDS = 60


def _alert_dispatch_once():
    """Single tick of the alert dispatcher. Idempotent and DB-driven."""
    try:
        tickers = db_all_unique_tickers()
    except Exception as e:
        print(f"[alert] could not list tracked tickers: {e}")
        return

    if not tickers:
        return  # No watchlists yet — nothing to do.

    sent  = 0
    skipped = 0
    for ticker in tickers:
        try:
            sig = db_get_signal(ticker)
            if not sig:
                continue
            action   = (sig.get('signal_type') or '').upper()
            strength = int(sig.get('signal_strength') or 0)
            if action not in ALERT_SIGNAL_TYPES or strength < ALERT_THRESHOLD:
                continue

            users = db_users_tracking(ticker)
            if not users:
                continue

            # Currency hint based on which catalogue the ticker came from.
            is_bist = ticker in BIST30_STOCKS
            currency_symbol = '₺' if is_bist else '$'

            for u in users:
                uid   = u.get('id') if hasattr(u, 'get') else u['id']
                email = u.get('email') if hasattr(u, 'get') else u['email']
                if not email:
                    continue
                if db_already_notified(uid, ticker, action):
                    skipped += 1
                    continue

                import email_helper
                result = email_helper.email_signal_alert(
                    to_email=email,
                    to_name=email,
                    ticker=ticker,
                    signal_type=action,
                    strength=strength,
                    price=sig.get('last_price') or 0,
                    next_price=sig.get('next_day_price') or 0,
                    summary=sig.get('summary') or '',
                    currency_symbol=currency_symbol,
                    dashboard_url=os.environ.get('APP_URL'),
                )
                if result is True:
                    db_log_email(uid, ticker, action, strength)
                    sent += 1
                else:
                    print(f"[alert] send failed to {email} for {ticker}: {result}")
        except Exception as e:
            print(f"[alert] {ticker} failed: {e}")

    if sent or skipped:
        print(f"[alert] tick: sent {sent}, skipped (cooldown) {skipped}")


def _alert_dispatcher_loop():
    import time
    time.sleep(20)  # let other refreshers settle first
    while True:
        try:
            _alert_dispatch_once()
        except Exception as e:
            print(f"[alert] loop error: {e}")
        time.sleep(ALERT_DISPATCH_SECONDS)


def start_alert_dispatcher():
    import threading
    import tempfile
    lock_path = os.path.join(tempfile.gettempdir(), 'tychain_alert_dispatcher.lock')
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
    except FileExistsError:
        return
    thread = threading.Thread(target=_alert_dispatcher_loop,
                              daemon=True, name='alert-dispatcher')
    thread.start()
    print("[alert] started — dispatching every", ALERT_DISPATCH_SECONDS, "seconds")


# Initialize database tables unconditionally for Gunicorn
init_db()

# Kick off the 3-minute background refresher. Disabled when running tests
# (DISABLE_REFRESHER=1) and when running ad-hoc CLI commands.
if os.environ.get('DISABLE_REFRESHER') != '1':
    try:
        start_quote_refresher()
    except Exception as e:
        # Never block app startup on the refresher.
        print(f"[refresher] failed to start: {e}")
    try:
        start_performance_refresher()
    except Exception as e:
        print(f"[perf] failed to start: {e}")
    try:
        start_alert_dispatcher()
    except Exception as e:
        print(f"[alert] failed to start: {e}")

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=False)

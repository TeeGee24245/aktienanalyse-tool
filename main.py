import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from concurrent.futures import ThreadPoolExecutor
import yfinance as yf
from pathlib import Path

st.set_page_config(page_title="InvestTool", layout="wide")

_CSS = '''
<style>
:root{
    --bg:#0b0f14; --card:#0f1720; --muted:#9aa8b6;
    --accent:#2dd4bf; --good:#16a34a; --bad:#ef4444; --neutral:#f59e0b;
}
html, body, [class*="css"]{background:var(--bg) !important; color:#e6eef8 !important}
.stApp{background:var(--bg)}
.card{background:var(--card);padding:14px;border-radius:8px;margin-bottom:10px}
.card h3{margin:0 0 6px 0}
.metric-label{color:var(--muted);font-size:13px}
.small{font-size:13px;color:var(--muted)}
.kpi{padding:12px;border-radius:8px;background:linear-gradient(90deg,rgba(255,255,255,0.02),transparent)}
.rec-buy{background:#052e14;border-left:6px solid var(--good);padding:16px;border-radius:8px;margin-top:12px}
.rec-hold{background:#2b2410;border-left:6px solid var(--neutral);padding:16px;border-radius:8px;margin-top:12px}
.rec-sell{background:#2a0f0f;border-left:6px solid var(--bad);padding:16px;border-radius:8px;margin-top:12px}
.logo{font-size:20px;font-weight:700;margin-bottom:6px}
</style>
'''
st.markdown(_CSS, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PEER-GRUPPEN
# ═══════════════════════════════════════════════════════════════════════════════

PEERS: dict[str, list[str]] = {
    "AAPL":    ["MSFT", "GOOGL", "META", "AMZN"],
    "MSFT":    ["AAPL", "GOOGL", "AMZN", "CRM"],
    "GOOGL":   ["META", "MSFT", "AAPL", "AMZN"],
    "SIE.DE":  ["SAP.DE", "ABB", "HON", "EMR"],
    "SAP.DE":  ["SIE.DE", "CRM", "ORCL", "MSFT"],
    "NVDA":    ["AMD", "INTC", "QCOM", "AVGO"],
    "ASML.AS": ["AMAT", "LRCX", "KLAC", "TER"],
}

_SECTOR_PEERS: dict[str, list[str]] = {
    "Technology":             ["MSFT", "AAPL", "GOOGL", "META"],
    "Consumer Cyclical":      ["AMZN", "TSLA", "NKE",   "MCD"],
    "Healthcare":             ["JNJ",  "UNH",  "PFE",   "MRK"],
    "Financial Services":     ["JPM",  "BAC",  "WFC",   "GS"],
    "Industrials":            ["HON",  "MMM",  "CAT",   "GE"],
    "Energy":                 ["XOM",  "CVX",  "COP",   "SLB"],
    "Basic Materials":        ["BHP",  "RIO",  "FCX",   "NEM"],
    "Consumer Defensive":     ["KO",   "PEP",  "WMT",   "PG"],
    "Communication Services": ["META", "GOOGL", "NFLX", "DIS"],
    "Real Estate":            ["AMT",  "PLD",  "CCI",   "EQIX"],
    "Utilities":              ["NEE",  "DUK",  "SO",    "AEP"],
    "Semiconductors":         ["NVDA", "AMD",  "INTC",  "QCOM"],
}


# ═══════════════════════════════════════════════════════════════════════════════
#  SHARED UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def human(n):
    try:
        n = float(n)
    except Exception:
        return '—'
    for unit in ['', 'K', 'M', 'B', 'T']:
        if abs(n) < 1000:
            return f"{n:.2f}{unit}"
        n /= 1000.0
    return f"{n:.2f}P"


def pct(val):
    if val is None:
        return '—'
    try:
        v = float(val)
        return f"{v * 100:.2f}%" if abs(v) < 1 else f"{v:.2f}%"
    except Exception:
        return '—'


def fmt2(x):
    try:
        return f"{float(x):.2f}"
    except Exception:
        return '—'


def colored_span(val, color):
    return f"<span style='color:{color};font-weight:700'>{val}</span>"


def safe_get(df, key):
    try:
        if key in df.index:
            return df.loc[key].dropna().iloc[0]
    except Exception:
        pass
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  AKTIENANALYSE — DATA & INDICATORS
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_data(ticker: str) -> dict:
    t = yf.Ticker(ticker)
    return {
        'ticker_obj': t,
        'history':        t.history(period="1y", interval="1d", actions=False),
        'info':           t.info or {},
        'financials':     t.financials     if hasattr(t, 'financials')    else pd.DataFrame(),
        'balance':        t.balance_sheet  if hasattr(t, 'balance_sheet') else pd.DataFrame(),
        'cashflow':       t.cashflow       if hasattr(t, 'cashflow')      else pd.DataFrame(),
        'earnings':       t.earnings       if hasattr(t, 'earnings')      else pd.DataFrame(),
        'recommendations':t.recommendations if hasattr(t, 'recommendations') else pd.DataFrame(),
    }


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta    = series.diff()
    gain     = delta.where(delta > 0, 0.0)
    loss     = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs       = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['EMA20']  = df['Close'].ewm(span=20,  adjust=False).mean()
    df['EMA50']  = df['Close'].ewm(span=50,  adjust=False).mean()
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
    df['RSI']    = _rsi(df['Close'], 14)
    return df


# ── Colour helpers ────────────────────────────────────────────────────────────

def _c_gross(v):
    try: v = float(v)
    except: return 'var(--muted)'
    return 'var(--good)' if v > 0.4 else 'var(--neutral)' if v >= 0.2 else 'var(--bad)'

def _c_net(v):
    try: v = float(v)
    except: return 'var(--muted)'
    return 'var(--good)' if v > 0.15 else 'var(--neutral)' if v >= 0.05 else 'var(--bad)'

def _c_growth(v):
    try: v = float(v)
    except: return 'var(--muted)'
    return 'var(--good)' if v > 0.15 else 'var(--neutral)' if v >= 0.05 else 'var(--bad)'

def _c_debt(v):
    try: v = float(v)
    except: return 'var(--muted)'
    return 'var(--good)' if v < 1 else 'var(--neutral)' if v <= 2 else 'var(--bad)'

def _c_rsi(v):
    try: v = float(v)
    except: return 'var(--muted)'
    return 'var(--bad)' if v > 70 else 'var(--good)' if v >= 30 else 'var(--neutral)'


# ── Scoring ───────────────────────────────────────────────────────────────────

def compute_scores(info: dict, fin: pd.DataFrame, ratios: dict):
    # Quality (100 pts): Bruttomarge 40 + Nettomarge 30 + FCF-Marge 30
    gross = info.get('grossMargins')
    net   = info.get('netMargins')
    fcf_m = None
    try:
        if info.get('totalRevenue'):
            fcf_m = (info.get('freeCashflow') or 0) / info.get('totalRevenue')
    except Exception:
        pass

    def _pts(v, lo, hi, max_pts):
        try:
            v = float(v)
            return max_pts if v > hi else max_pts // 2 if v >= lo else 0
        except Exception:
            return 0

    quality = (
        _pts(gross, 0.20, 0.40, 40) +
        _pts(net,   0.05, 0.15, 30) +
        _pts(fcf_m, 0.05, 0.10, 30)
    )

    # Value (100 pts): PE-based
    pe = info.get('trailingPE') or info.get('forwardPE')
    try:
        p = float(pe)
        value = 80 if p < 25 else 50 if p <= 35 else 30
    except Exception:
        value = 50

    # Momentum (100 pts): RSI + EMA50
    try:
        r    = float(ratios.get('rsi') or 50)
        base = 50 if 40 <= r <= 60 else 70 if r > 60 else 30
    except Exception:
        base = 50
    momentum = base
    try:
        if ratios.get('close') and ratios.get('ema50') and ratios['close'] > ratios['ema50']:
            momentum = min(100, momentum + 20)
    except Exception:
        pass

    return quality, value, momentum


def score_gauge(score, title):
    fig = go.Figure(go.Indicator(
        mode='gauge+number', value=score, title={'text': title},
        gauge={
            'axis':  {'range': [0, 100]},
            'bar':   {'color': '#00bcd4'},
            'steps': [
                {'range': [0,  50], 'color': '#ef4444'},
                {'range': [50, 80], 'color': '#f59e0b'},
                {'range': [80,100], 'color': '#16a34a'},
            ],
        }
    ))
    fig.update_layout(height=220, paper_bgcolor='rgba(0,0,0,0)', font={'color': '#e6eef8'})
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  PEER-DATEN LADEN
# ═══════════════════════════════════════════════════════════════════════════════

def _fetch_one_peer(tk: str) -> dict:
    try:
        info = yf.Ticker(tk).info or {}
        gm = info.get('grossMargins')
        rg = info.get('revenueGrowth')

        pm = info.get('profitMargins') or info.get('netMargins')
        if pm is None:
            ni = info.get('netIncomeToCommon', 0)
            rev = info.get('totalRevenue', 0)
            pm = ni / rev if rev else None
        net_margin = f"{pm * 100:.1f} %" if pm is not None else "—"

        dy = info.get('dividendYield') or 0
        div_yield = f"{dy * 100:.2f} %" if dy else "—"

        return {
            'Ticker':              tk,
            'Unternehmen':         info.get('shortName') or info.get('longName') or tk,
            'Kurs':                info.get('regularMarketPrice') or info.get('currentPrice'),
            'KGV':                 info.get('trailingPE') or info.get('forwardPE'),
            'KUV':                 info.get('priceToSalesTrailing12Months'),
            'Bruttomarge %':       round(float(gm) * 100, 1) if gm is not None else None,
            'Nettomarge %':        net_margin,
            'Umsatzwachstum %':    round(float(rg) * 100, 1) if rg is not None else None,
            'Dividendenrendite %': div_yield,
            'Währung':             info.get('currency', ''),
        }
    except Exception:
        return {'Ticker': tk, 'Unternehmen': tk}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_peers_data(tickers: tuple) -> pd.DataFrame:
    """Lädt Kennzahlen für alle Peer-Ticker parallel (ThreadPoolExecutor)."""
    with ThreadPoolExecutor(max_workers=min(len(tickers), 6)) as ex:
        rows = list(ex.map(_fetch_one_peer, tickers))
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: AKTIENANALYSE
# ═══════════════════════════════════════════════════════════════════════════════

def show_aktienanalyse(ticker: str):
    if not ticker:
        st.error('Bitte einen Ticker eingeben.')
        return

    with st.spinner(f'Lade Daten für {ticker.upper()} …'):
        data = fetch_data(ticker.strip().upper())

    info = data['info']
    hist = data['history']
    fin  = data['financials']
    cash = data['cashflow']
    recs = data['recommendations']

    if hist is None or hist.empty:
        st.error('Keine historischen Daten gefunden. Bitte Ticker prüfen.')
        return

    df = compute_indicators(hist)

    ratios = {
        'rsi':    df['RSI'].iloc[-1]   if 'RSI'   in df.columns else 50,
        'ema20':  df['EMA20'].iloc[-1]  if 'EMA20'  in df.columns else None,
        'ema50':  df['EMA50'].iloc[-1]  if 'EMA50'  in df.columns else None,
        'ema200': df['EMA200'].iloc[-1] if 'EMA200' in df.columns else None,
        'close':  df['Close'].iloc[-1],
        'ret_3m': (df['Close'].iloc[-1] / df['Close'].iloc[-63] - 1) if len(df) > 63 else 0,
    }

    quality, value, momentum = compute_scores(info, fin, ratios)

    # ── Header ────────────────────────────────────────────────────────────────
    c1, c2 = st.columns([3, 1])
    with c1:
        st.subheader(f"{info.get('longName', ticker)} — {ticker.upper()}")
        if info.get('longBusinessSummary'):
            st.markdown(
                f"<div class='small'>{info['longBusinessSummary'][:800]}…</div>",
                unsafe_allow_html=True,
            )
    with c2:
        st.markdown(
            f"<div class='card'><h3>{human(info.get('regularMarketPrice'))}</h3>"
            f"<div class='small'>Aktueller Preis</div></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='card'><h3>{human(info.get('marketCap'))}</h3>"
            f"<div class='small'>Marktkapitalisierung</div></div>",
            unsafe_allow_html=True,
        )

    # ── Scores ────────────────────────────────────────────────────────────────
    st.markdown('**Scores**')
    sc1, sc2, sc3 = st.columns(3)
    sc1.plotly_chart(score_gauge(quality,  'Quality'),  use_container_width=True)
    sc2.plotly_chart(score_gauge(value,    'Value'),    use_container_width=True)
    sc3.plotly_chart(score_gauge(momentum, 'Momentum'), use_container_width=True)

    rsi_val = ratios.get('rsi')
    if rsi_val is not None:
        st.markdown(
            f"<div class='card'><div class='metric-label'>RSI</div>"
            f"<h3 style='color:{_c_rsi(rsi_val)}'>{fmt2(rsi_val)}</h3></div>",
            unsafe_allow_html=True,
        )

    # ── Handlungsempfehlung ───────────────────────────────────────────────────
    total = int(round(quality * 0.4 + value * 0.3 + momentum * 0.3))
    if total > 65:
        rc = 'rec-buy'
        best = max(('Quality', quality), ('Value', value), ('Momentum', momentum), key=lambda x: x[1])
        rt = (f"KAUFEN — Gesamt-Score {total}. Starke {best[0]} ({best[1]}). "
              f"Positive Kombination aus Qualität und Momentum.")
    elif total >= 45:
        rc = 'rec-hold'
        rt = f"HALTEN — Gesamt-Score {total}. Gemischte Signale; prüfe Margen und Bewertung."
    else:
        rc = 'rec-sell'
        rt = f"VERKAUFEN — Gesamt-Score {total}. Schwache Fundamentaldaten oder Momentum; Risiko hoch."

    st.markdown(
        f"<div class='{rc}'><strong style='font-size:18px'>{rt}</strong></div>",
        unsafe_allow_html=True,
    )

    # ── DCF-Bewertungsmodell ──────────────────────────────────────────────────
    st.markdown('---')
    st.markdown('**Bewertungsmodell (DCF)**')
    with st.expander('DCF Details', expanded=True):
        curr = info.get('currency') or 'USD'
        sym  = '$' if curr.upper() == 'USD' else '€' if curr.upper() == 'EUR' else curr

        # Letzter FCF
        fcf_last = None
        try:
            raw = info.get('freeCashflow') or safe_get(cash, 'Free Cash Flow')
            if raw is not None:
                fcf_last = float(raw)
        except Exception:
            pass

        # FCF-Wachstum: Ø der letzten 3 jährlichen Veränderungen
        # Nur positive→positive Übergänge sind aussagekräftig
        fcf_growth    = None
        growth_source = 'Default (8 % – kein historischer FCF verfügbar)'
        try:
            if cash is not None and not cash.empty and 'Free Cash Flow' in cash.index:
                vals    = cash.loc['Free Cash Flow'].dropna().astype(float).values
                changes = []
                for i in range(1, min(len(vals), 4)):
                    prev_v, curr_v = vals[i], vals[i - 1]
                    if prev_v > 0 and curr_v > 0:          # beide positiv → valide
                        changes.append((curr_v / prev_v) - 1)
                if changes:
                    fcf_growth    = float(np.mean(changes))
                    growth_source = f'Historischer FCF-Ø ({len(changes)} Jahreswerte)'
        except Exception:
            pass

        # Fallback: konservativer Default 8 % (kein revenueGrowth – zu volatil)
        if fcf_growth is None:
            fcf_growth = 0.08

        fcf_growth = min(max(fcf_growth, 0.03), 0.25)   # Min 3 %, Max 25 %

        discount_rate   = 0.10
        terminal_growth = 0.03
        shares_out      = info.get('sharesOutstanding')
        price_now       = info.get('regularMarketPrice')

        if not fcf_last:
            st.warning('Nicht genügend Free-Cashflow-Daten für DCF-Berechnung gefunden.')
        else:
            years          = list(range(1, 11))
            proj           = [fcf_last * (1 + fcf_growth) ** y for y in years]
            terminal_value = proj[-1] * (1 + terminal_growth) / (discount_rate - terminal_growth)
            pv_fcfs        = sum(proj[i] / (1 + discount_rate) ** (i + 1) for i in range(10))
            pv_terminal    = terminal_value / (1 + discount_rate) ** 10
            enterprise_val = pv_fcfs + pv_terminal

            fair_per_share = None
            try:
                if shares_out:
                    fair_per_share = enterprise_val / float(shares_out)
            except Exception:
                pass

            # Kennzahlen
            k1, k2, k3 = st.columns(3)
            k1.metric("Fairer Wert (DCF)",
                      f"{sym}{fair_per_share:,.2f}" if fair_per_share else '—')
            k2.metric("Aktueller Kurs",
                      f"{sym}{price_now:,.2f}" if price_now else '—')
            st.caption(
                f"📈 FCF-Wachstum: **{fcf_growth * 100:.1f}% p.a.** — Quelle: {growth_source} "
                f"| Discount-Rate: 10 % | Terminal Growth: 3 %"
            )

            # Upside / Downside
            if fair_per_share and price_now:
                upside = (fair_per_share - float(price_now)) / float(price_now) * 100
                sign   = '+' if upside >= 0 else ''
                uc     = 'var(--good)' if upside > 0 else 'var(--bad)'
                k3.markdown(
                    f"<div class='card'><div class='small'>Upside / Downside</div>"
                    f"<h3 style='color:{uc}'>{sign}{upside:.2f}%</h3></div>",
                    unsafe_allow_html=True,
                )

                # Margin of Safety
                if upside > 20:
                    mos_text  = 'Kaufgelegenheit — Margin of Safety vorhanden (> 20 %)'
                    mos_class = 'rec-buy'
                elif upside >= -20:
                    mos_text  = 'Fair bewertet — kein klarer Sicherheitsabstand'
                    mos_class = 'rec-hold'
                else:
                    mos_text  = 'Überbewertet — Kurs deutlich über fairen Wert'
                    mos_class = 'rec-sell'
                st.markdown(
                    f"<div class='{mos_class}'><strong>{mos_text}</strong></div>",
                    unsafe_allow_html=True,
                )

            # Projizierte FCF-Tabelle
            proj_df = pd.DataFrame({
                'Jahr':             [f'Jahr {y}' for y in years],
                'Projizierter FCF': [f"{sym}{v:,.0f}" for v in proj],
            })
            st.markdown('**Projizierte Free Cash Flows (10 Jahre)**')
            st.table(proj_df)

    # ── Peer-Vergleich ────────────────────────────────────────────────────────
    st.markdown('---')
    st.markdown('**Peer-Vergleich**')

    ticker_up = ticker.upper()
    peer_list = PEERS.get(ticker_up)

    if peer_list is None:
        sector = info.get('sector', '')
        peer_list = [p for p in _SECTOR_PEERS.get(sector, []) if p != ticker_up][:4]
        if peer_list:
            st.caption(
                f"Keine vordefinierten Peers für {ticker_up} — "
                f"zeige branchenverwandte Werte ({sector})"
            )

    if not peer_list:
        st.info('Keine Peer-Daten verfügbar für diesen Ticker.')
    else:
        all_peer_tickers = tuple([ticker_up] + [p for p in peer_list if p != ticker_up])
        with st.spinner('Lade Peer-Daten parallel …'):
            peer_df = fetch_peers_data(all_peer_tickers)

        if peer_df.empty:
            st.warning('Peer-Daten konnten nicht geladen werden.')
        else:
            def _fn(v, dec=1, suf=''):
                try:
                    return f"{float(v):.{dec}f}{suf}" if v is not None else '—'
                except Exception:
                    return '—'

            display = peer_df.copy()
            # Unternehmen (Ticker) zusammenführen und an erste Stelle
            display.insert(
                0, 'Unternehmen (Ticker)',
                display['Unternehmen'].fillna(display['Ticker']) + ' (' + display['Ticker'] + ')',
            )
            display.drop(columns=['Unternehmen', 'Ticker', 'Währung'], errors='ignore', inplace=True)

            display['Kurs']                = display['Kurs'].apply(lambda x: _fn(x, 2))
            display['KGV']                 = display['KGV'].apply(lambda x: _fn(x, 1))
            display['KUV']                 = display['KUV'].apply(lambda x: _fn(x, 2))
            display['Bruttomarge %']       = display['Bruttomarge %'].apply(lambda x: _fn(x, 1, ' %'))
            display['Nettomarge %']        = display['Nettomarge %'].fillna('—').apply(lambda x: '—' if x is None or str(x) == 'None' else x)
            display['Umsatzwachstum %']    = display['Umsatzwachstum %'].apply(lambda x: _fn(x, 1, ' %'))
            display['Dividendenrendite %'] = display['Dividendenrendite %'].fillna('—').apply(lambda x: '—' if x is None or str(x) == 'None' else x)

            # Erste Zeile (analysierte Aktie) hervorheben
            def _highlight_main(row):
                if row.name == 0:
                    return ['background-color:#1a2744; color:#2dd4bf; font-weight:700'] * len(row)
                return [''] * len(row)

            styled = display.style.apply(_highlight_main, axis=1)
            st.dataframe(styled, use_container_width=True, hide_index=True)
            st.caption(
                f"Hervorgehobene Zeile = analysierte Aktie ({ticker_up}). "
                "Kurs in Heimatwährung des jeweiligen Titels."
            )

    # ── Kennzahlen & Fundamentaldaten ─────────────────────────────────────────
    st.markdown('**Kennzahlen & Fundamentaldaten**')

    net_margin = info.get('netMargins')
    if net_margin is None:
        ni  = info.get('netIncomeToCommon') or safe_get(fin, 'Net Income')
        rev = info.get('totalRevenue')       or safe_get(fin, 'Total Revenue')
        try:
            if ni and rev:
                net_margin = float(ni) / float(rev)
        except Exception:
            pass

    debt_raw = info.get('debtToEquity')
    try:
        debt_num = float(debt_raw) / 100.0 if debt_raw is not None else None
    except Exception:
        debt_num = None

    c = st.columns(4)
    pe_val = info.get('trailingPE') or info.get('forwardPE')
    c[0].markdown(f"<div class='kpi'><div class='metric-label'>KGV (PE)</div><h3>{fmt2(pe_val) if pe_val else '—'}</h3></div>", unsafe_allow_html=True)
    c[1].markdown(f"<div class='kpi'><div class='metric-label'>KUV (PS)</div><h3>{fmt2(info.get('priceToSalesTrailing12Months'))}</h3></div>", unsafe_allow_html=True)
    c[2].markdown(f"<div class='kpi'><div class='metric-label'>KBV (PB)</div><h3>{fmt2(info.get('priceToBook'))}</h3></div>", unsafe_allow_html=True)
    c[3].markdown(f"<div class='kpi'><div class='metric-label'>EV/EBITDA</div><h3>{fmt2(info.get('enterpriseToEbitda'))}</h3></div>", unsafe_allow_html=True)

    c2 = st.columns(4)
    g = info.get('grossMargins')
    c2[0].markdown(f"<div class='kpi'><div class='metric-label'>Bruttomarge</div><h3>{colored_span(pct(g), _c_gross(g))}</h3></div>", unsafe_allow_html=True)
    c2[1].markdown(f"<div class='kpi'><div class='metric-label'>Nettomarge</div><h3>{colored_span(pct(net_margin), _c_net(net_margin))}</h3></div>", unsafe_allow_html=True)
    c2[2].markdown(f"<div class='kpi'><div class='metric-label'>EBITDA-Marge</div><h3>{pct(info.get('ebitdaMargins'))}</h3></div>", unsafe_allow_html=True)
    rg = info.get('revenueGrowth')
    c2[3].markdown(f"<div class='kpi'><div class='metric-label'>Umsatzwachstum YoY</div><h3>{colored_span(pct(rg), _c_growth(rg))}</h3></div>", unsafe_allow_html=True)

    c3 = st.columns(4)
    fcf_v = info.get('freeCashflow') or safe_get(cash, 'Free Cash Flow')
    c3[0].markdown(f"<div class='kpi'><div class='metric-label'>Free Cash Flow</div><h3>{human(fcf_v) if fcf_v else '—'}</h3></div>", unsafe_allow_html=True)
    fcf_m2 = None
    try:
        if info.get('totalRevenue'):
            fcf_m2 = (info.get('freeCashflow') or 0) / info['totalRevenue']
    except Exception:
        pass
    c3[1].markdown(f"<div class='kpi'><div class='metric-label'>FCF-Marge</div><h3>{pct(fcf_m2)}</h3></div>", unsafe_allow_html=True)
    c3[2].markdown(f"<div class='kpi'><div class='metric-label'>Verschuldungsgrad D/E</div><h3>{colored_span(fmt2(debt_num) if debt_num is not None else '—', _c_debt(debt_num))}</h3></div>", unsafe_allow_html=True)
    div_raw = info.get('dividendYield')
    c3[3].markdown(f"<div class='kpi'><div class='metric-label'>Dividendenrendite</div><h3>{f'{float(div_raw):.2f}%' if div_raw else '—'}</h3></div>", unsafe_allow_html=True)

    c4 = st.columns(4)
    c4[0].markdown(f"<div class='kpi'><div class='metric-label'>52W Hoch</div><h3>{info.get('fiftyTwoWeekHigh') or '—'}</h3></div>", unsafe_allow_html=True)
    c4[1].markdown(f"<div class='kpi'><div class='metric-label'>52W Tief</div><h3>{info.get('fiftyTwoWeekLow') or '—'}</h3></div>", unsafe_allow_html=True)
    hi = info.get('fiftyTwoWeekHigh')
    mp = info.get('regularMarketPrice')
    if hi and mp:
        c4[2].markdown(f"<div class='kpi'><div class='metric-label'>Abstand 52W Hoch</div><h3>{pct((hi - mp) / hi)}</h3></div>", unsafe_allow_html=True)
    else:
        c4[2].markdown(f"<div class='kpi'><div class='metric-label'>Abstand 52W Hoch</div><h3>—</h3></div>", unsafe_allow_html=True)
    c4[3].markdown(f"<div class='kpi'><div class='metric-label'>Beta</div><h3>{info.get('beta') or '—'}</h3></div>", unsafe_allow_html=True)

    # ── Analystenkonsens ──────────────────────────────────────────────────────
    target       = None
    analyst_text = 'Keine Analystendaten'
    try:
        target = info.get('targetMeanPrice')
        if not recs.empty:
            ly     = recs[recs.index > (pd.Timestamp.today() - pd.Timedelta(days=365))]
            col    = 'To Grade' if 'To Grade' in ly.columns else 'Action'
            counts = ly[col].value_counts()
            analyst_text = f"Analysten (letztes Jahr): {counts.to_dict()}"
        else:
            analyst_text = 'Keine Empfehlungen verfügbar'
    except Exception:
        pass

    st.markdown(
        f"<div class='card'><h3>Analystenkonsens</h3>"
        f"<div class='small'>Durchschn. Kursziel: {human(target) if target else '—'}"
        f"<br>{analyst_text}</div></div>",
        unsafe_allow_html=True,
    )

    # ── Charts ────────────────────────────────────────────────────────────────
    st.markdown('**Charts**')
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.6, 0.15, 0.25], vertical_spacing=0.03,
    )
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                  low=df['Low'], close=df['Close'], name='Preis'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'],  line=dict(color='#f97316', width=1), name='EMA20'),  row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA50'],  line=dict(color='#60a5fa', width=1), name='EMA50'),  row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA200'], line=dict(color='#34d399', width=1), name='EMA200'), row=1, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color='#475569', name='Volumen'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#ffd166'), name='RSI'), row=3, col=1)
    fig.add_hline(y=70, line=dict(color='red',   dash='dash'), row=3, col=1)
    fig.add_hline(y=30, line=dict(color='green', dash='dash'), row=3, col=1)
    fig.update_layout(template='plotly_dark', height=900, margin=dict(l=40, r=20, t=40, b=20))
    fig.update_xaxes(rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # Umsatz & Gewinn
    if not fin.empty:
        try:
            rev_row = (fin.loc['Total Revenue'] if 'Total Revenue' in fin.index
                       else fin.loc['Revenue']  if 'Revenue'       in fin.index else None)
            net_row = (fin.loc['Net Income']    if 'Net Income'    in fin.index
                       else fin.loc['NetIncome'] if 'NetIncome'    in fin.index else None)
            if rev_row is not None and net_row is not None:
                rev_row = rev_row.dropna()
                net_row = net_row.dropna()
                yr_labels = rev_row.index.astype(str).tolist()[:4]
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(x=yr_labels, y=[rev_row[c] for c in yr_labels], name='Umsatz',      marker_color='#60a5fa'))
                fig2.add_trace(go.Bar(x=yr_labels, y=[net_row[c] for c in yr_labels], name='Nettogewinn', marker_color='#34d399'))
                fig2.update_layout(barmode='group', template='plotly_dark',
                                   title='Umsatz & Gewinn (letzte Jahre)')
                st.plotly_chart(fig2, use_container_width=True)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: PORTFOLIO DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

_PORTFOLIO_DATA = [
    ("Alphabet","Tech","USA",204866), ("Nvidia","Halbleiter","USA",180917),
    ("Apple","Tech","USA",172736), ("Meta","Tech","USA",157478),
    ("Microsoft","Tech","USA",146750), ("ASML","Halbleiter","NL",123860),
    ("Visa","Fintech","USA",109515), ("Eli Lilly","Health","USA",90467),
    ("Procter & Gamble","Konsum","USA",89388), ("Applied Materials","Halbleiter","USA",82620),
    ("Johnson & Johnson","Health","USA",78595), ("Mastercard","Fintech","USA",78004),
    ("Roche","Health","CH",76903), ("Caterpillar","Industrie","USA",73006),
    ("Lam Research","Halbleiter","USA",72969), ("Costco","Konsum","USA",63069),
    ("Cisco","Tech","USA",60062), ("Linde","Industrie","IE",57356),
    ("Merck & Co","Health","USA",53409), ("Rio Tinto","Rohstoffe","UK",52512),
    ("BHP","Rohstoffe","AUS",51984), ("Emerson","Industrie","USA",51677),
    ("Walmart","Konsum","USA",49707), ("Coca-Cola","Konsum","USA",48989),
    ("Netflix","Tech","USA",47208), ("GE Aerospace","Industrie","USA",44519),
    ("Analog Devices","Halbleiter","USA",43862), ("ConocoPhillips","Energie","USA",43386),
    ("Novartis","Health","CH",42441), ("Progressive","Finanzen","USA",42046),
    ("DSV","Logistik","DK",41932), ("AstraZeneca","Health","UK",41751),
    ("Atlas Copco","Industrie","SE",40389), ("Eaton","Industrie","USA",40141),
    ("ABB","Industrie","CH",39849), ("Hilton","Konsum","USA",37334),
    ("Citigroup","Finanzen","USA",37300), ("Corning","Industrie","USA",36777),
    ("Ametek","Industrie","USA",36595), ("Schneider Electric","Industrie","FR",36575),
    ("Rheinmetall","Defense","DE",33880), ("Amazon","Tech","USA",33713),
    ("Broadcom","Halbleiter","USA",32379), ("GSK","Health","UK",31716),
    ("Berkshire Hathaway","Holding","USA",31600), ("CME Group","Börse","USA",30650),
    ("L'Oréal","Konsum","FR",29617), ("Wells Fargo","Finanzen","USA",29158),
    ("Novo Nordisk","Health","DK",28332), ("Danaher","Health","USA",28171),
    ("Amphenol","Industrie","USA",28061), ("Vertex","Health","USA",27793),
    ("Safran","Industrie","FR",27672), ("Siemens","Industrie","DE",26928),
    ("Abbott","Health","USA",26302), ("Gallagher","Finanzen","USA",26256),
    ("Nestlé","Konsum","CH",26207), ("Barclays","Finanzen","UK",25990),
    ("Intuitive Surgical","Health","USA",25365), ("Morgan Stanley","Finanzen","USA",22362),
    ("Teradyne","Halbleiter","USA",21794), ("Uber","Tech","USA",21195),
    ("Fastenal","Industrie","USA",20961), ("Snowflake","Tech","USA",20811),
    ("Stryker","Health","USA",20705), ("Palo Alto","Tech","USA",20579),
    ("Ferguson","Industrie","USA",19755), ("Southern Copper","Rohstoffe","USA",18704),
    ("Hermes","Luxus","FR",18540), ("ADP","IT Services","USA",18337),
    ("Intuit","Software","USA",18088), ("Cadence","Software","USA",16737),
    ("IBM","Tech","USA",16512), ("KLA","Halbleiter","USA",16246),
    ("IDEXX","Health","USA",16080), ("Dell","Tech","USA",15281),
    ("Nasdaq","Finanzen","USA",14907), ("Autodesk","Software","USA",14546),
    ("Booking","Travel","USA",14419),
]


def show_portfolio():
    pf = pd.DataFrame(_PORTFOLIO_DATA, columns=["Company", "Sector", "Country", "Value_EUR"])
    pf["Value_EUR"] = pf["Value_EUR"].astype(float)
    total = pf["Value_EUR"].sum()
    pf["Weight_%"] = pf["Value_EUR"] / total * 100

    def _region(c):
        if c == "USA":  return "USA"
        if c in {"NL","CH","IE","UK","SE","FR","DE","DK"}: return "Europe"
        if c == "AUS":  return "Oceania"
        return "Other"

    pf["Region"] = pf["Country"].apply(_region)

    st.title("Portfolio Dashboard")

    # Header metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Gesamtwert Portfolio (EUR)", f"{int(total):,} €")
    m2.metric("Anzahl Positionen", len(pf))
    m3.metric("Anzahl Branchen", pf["Sector"].nunique())

    st.markdown("---")

    # Branchengewichtung + Top 10
    sector_df = pf.groupby("Sector", as_index=False)["Value_EUR"].sum().sort_values("Value_EUR", ascending=False)
    top10     = pf.sort_values("Value_EUR", ascending=False).head(10).copy()

    ca, cb = st.columns(2)
    with ca:
        st.subheader("Branchengewichtung")
        fig = px.pie(sector_df, names="Sector", values="Value_EUR", hole=0.45,
                     color_discrete_sequence=px.colors.qualitative.Dark24)
        fig.update_layout(template="plotly_dark", margin=dict(t=30, b=10, l=10, r=10),
                          legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)
    with cb:
        st.subheader("Top 10 Positionen")
        t10r = top10[::-1]
        fig2 = px.bar(t10r, x="Value_EUR", y="Company", orientation='h',
                      text=t10r["Value_EUR"].map(lambda x: f"{int(x):,} €"),
                      color=t10r["Value_EUR"],
                      color_continuous_scale=px.colors.sequential.Plasma)
        fig2.update_layout(template="plotly_dark", margin=dict(t=30, b=10, l=10, r=10),
                           showlegend=False)
        fig2.update_xaxes(title_text="Wert (EUR)")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # Länder-Verteilung
    st.subheader("Länder-Verteilung")
    region_df = pf.groupby("Region", as_index=False)["Value_EUR"].sum()
    fig3 = px.pie(region_df, names="Region", values="Value_EUR", hole=0.3,
                  color_discrete_sequence=px.colors.sequential.Darkmint)
    fig3.update_layout(template="plotly_dark", margin=dict(t=30, b=10, l=10, r=10))
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")

    # Positions-Tabelle
    st.subheader("Positions-Tabelle")
    display = pf.sort_values("Value_EUR", ascending=False).reset_index(drop=True).copy()
    display["Value_EUR"] = display["Value_EUR"].map(lambda x: f"{int(x):,} €")
    display["Weight_%"]  = display["Weight_%"].map(lambda x: f"{x:.2f}%")
    st.dataframe(
        display.style.set_properties(**{"background-color": "#0f1721", "color": "#e6edf3"}),
        height=500,
    )

    st.markdown("---")

    # Konzentrations-Analyse
    st.subheader("Konzentrations-Analyse")
    sorted_pf = pf.sort_values("Value_EUR", ascending=False)
    top5_sum = sorted_pf.head(5)["Value_EUR"].sum()
    cx, cy = st.columns(2)
    cx.metric("Top 5 Wert (EUR)",           f"{int(top5_sum):,} €")
    cy.metric("Top 5 Anteil am Portfolio",   f"{top5_sum / total * 100:.2f}%")


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR & NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════════

def _search_quotes(query: str) -> list[dict]:
    """Gibt bis zu 5 yfinance-Suchergebnisse zurück (nur handelbare Wertpapiere)."""
    try:
        results = yf.Search(query).quotes
        ALLOWED = {'EQUITY', 'ETF', 'FUND', ''}
        filtered = [
            r for r in results
            if r.get('symbol') and r.get('quoteType', '').upper() in ALLOWED
        ]
        return filtered[:5]
    except Exception:
        return []


with st.sidebar:
    st.markdown("<div class='logo'>📈 InvestTool</div>", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("Navigation", ["Aktienanalyse", "Portfolio"], index=0)
    st.markdown("---")

    if page == "Aktienanalyse":
        # Persistierter aktiver Ticker
        if 'active_ticker' not in st.session_state:
            st.session_state['active_ticker'] = 'AAPL'

        query = st.text_input(
            'Aktie suchen',
            placeholder='Name oder Ticker … z.B. Apple, Siemens, ASML',
            key='search_query',
        )

        selected_ticker = None

        if query and len(query.strip()) >= 2:
            hits = _search_quotes(query.strip())
            if hits:
                labels = [
                    f"{r.get('shortname') or r.get('longname') or r['symbol']}  "
                    f"({r['symbol']})  —  {r.get('exchange', '?')}"
                    for r in hits
                ]
                idx = st.selectbox(
                    'Ergebnis auswählen',
                    range(len(labels)),
                    format_func=lambda i: labels[i],
                )
                selected_ticker = hits[idx]['symbol']
            else:
                st.warning('Keine Ergebnisse — Ticker direkt eingeben?')
                selected_ticker = query.strip().upper()

        # Laden-Button
        st.write('')
        load_disabled = selected_ticker is None
        if st.button('Laden', disabled=load_disabled):
            st.session_state['active_ticker'] = selected_ticker
            st.rerun()

        st.caption(f"Geladen: **{st.session_state['active_ticker']}**")

    st.caption("Streamlit Dashboard — starte mit: streamlit run main.py")

# Aktiver Ticker für die Analyse
ticker = st.session_state.get('active_ticker', 'AAPL') if page == "Aktienanalyse" else ''

st.title("InvestTool — Module Übersicht")

if page == "Aktienanalyse":
    show_aktienanalyse(ticker)
elif page == "Portfolio":
    show_portfolio()

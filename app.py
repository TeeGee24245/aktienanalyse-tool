import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go


st.set_page_config(page_title="Aktienanalyse — Modul 1", layout="wide")

_CSS = '''
<style>
:root{--bg:#0b0f14;--card:#0f1720;--muted:#9aa8b6;--accent:#2dd4bf;--good:#16a34a;--bad:#ef4444;--neutral:#f59e0b}
html, body, [class*="css"]{background:var(--bg) !important; color:#e6eef8 !important}
.stApp{background:var(--bg)}
.card{background:var(--card);padding:14px;border-radius:8px;margin-bottom:10px}
.card h3{margin:0 0 6px 0}
.metric-label{color:var(--muted);font-size:13px}
.score{display:flex;align-items:center;gap:10px}
.small{font-size:13px;color:var(--muted)}
.kpi{padding:12px;border-radius:8px;background:linear-gradient(90deg, rgba(255,255,255,0.02), transparent)}
.rec-buy{background:#052e14;border-left:6px solid var(--good);padding:16px;border-radius:8px;margin-top:12px}
.rec-hold{background:#2b2410;border-left:6px solid var(--neutral);padding:16px;border-radius:8px;margin-top:12px}
.rec-sell{background:#2a0f0f;border-left:6px solid var(--bad);padding:16px;border-radius:8px;margin-top:12px}
</style>
'''

st.markdown(_CSS, unsafe_allow_html=True)


def fetch_data(ticker: str):
    t = yf.Ticker(ticker)
    hist = t.history(period="1y", interval="1d", actions=False)
    info = t.info or {}
    financials = t.financials if hasattr(t, 'financials') else pd.DataFrame()
    balance = t.balance_sheet if hasattr(t, 'balance_sheet') else pd.DataFrame()
    cash = t.cashflow if hasattr(t, 'cashflow') else pd.DataFrame()
    earnings = t.earnings if hasattr(t, 'earnings') else pd.DataFrame()
    recs = t.recommendations if hasattr(t, 'recommendations') else pd.DataFrame()
    return {
        'ticker_obj': t,
        'history': hist,
        'info': info,
        'financials': financials,
        'balance': balance,
        'cashflow': cash,
        'earnings': earnings,
        'recommendations': recs,
    }


def compute_indicators(df: pd.DataFrame):
    df = df.copy()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
    df['RSI'] = rsi(df['Close'], 14)
    return df


def rsi(series: pd.Series, period: int = 14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def human(n):
    try:
        n = float(n)
    except Exception:
        return '—'
    for unit in ['','K','M','B','T']:
        if abs(n) < 1000:
            return f"{n:.2f}{unit}"
        n /= 1000.0
    return f"{n:.2f}P"


def pct(val):
    if val is None:
        return '—'
    try:
        v = float(val)
        if abs(v) < 1:
            return f"{v * 100:.2f}%"
        else:
            return f"{v:.2f}%"
    except:
        return '—'


def fmt_num2(x):
    try:
        return f"{float(x):.2f}"
    except Exception:
        return '—'


def colored_span(val, color):
    return f"<span style='color:{color};font-weight:700'>{val}</span>"


def color_for_gross(v):
    try:
        v = float(v)
    except Exception:
        return 'var(--muted)'
    if v > 0.4:
        return 'var(--good)'
    if v >= 0.2:
        return 'var(--neutral)'
    return 'var(--bad)'


def color_for_net(v):
    try:
        v = float(v)
    except Exception:
        return 'var(--muted)'
    if v > 0.15:
        return 'var(--good)'
    if v >= 0.05:
        return 'var(--neutral)'
    return 'var(--bad)'


def color_for_growth(v):
    try:
        v = float(v)
    except Exception:
        return 'var(--muted)'
    if v > 0.15:
        return 'var(--good)'
    if v >= 0.05:
        return 'var(--neutral)'
    return 'var(--bad)'


def color_for_debt(v):
    try:
        v = float(v)
    except Exception:
        return 'var(--muted)'
    # v expected as ratio (e.g. 1.5)
    if v < 1:
        return 'var(--good)'
    if v <= 2:
        return 'var(--neutral)'
    return 'var(--bad)'


def color_for_rsi(v):
    try:
        v = float(v)
    except Exception:
        return 'var(--muted)'
    if v > 70:
        return 'var(--bad)'
    if v >= 30:
        return 'var(--good)'
    return 'var(--neutral)'


def safe_get(df, key):
    try:
        if key in df.index:
            return df.loc[key].dropna().iloc[0]
    except Exception:
        return None
    return None


def normalize_score(x, good_low=None, good_high=None, invert=False):
    # map x to 0-100 with simple linear caps
    try:
        v = float(x)
    except Exception:
        return 50
    if good_low is None or good_high is None:
        return max(0, min(100, v))
    if invert:
        # lower is better
        if v <= good_low:
            return 100
        if v >= good_high:
            return 0
        return int(100 * (good_high - v) / (good_high - good_low))
    else:
        if v <= good_low:
            return 0
        if v >= good_high:
            return 100
        return int(100 * (v - good_low) / (good_high - good_low))


def compute_scores(info, fin, ratios):
    # New scoring per user spec
    # Quality Score: Bruttomarge (40 pts), Nettomarge (30 pts), FCF-Marge (30 pts)
    gross = info.get('grossMargins')
    net = info.get('netMargins')
    # compute fcf margin as freeCashflow / totalRevenue when possible
    fcf_margin = None
    try:
        if info.get('totalRevenue'):
            fcf_margin = (info.get('freeCashflow') or 0) / info.get('totalRevenue')
    except Exception:
        fcf_margin = None

    gross_pts = 0
    if gross is not None:
        try:
            g = float(gross)
            if g > 0.40:
                gross_pts = 40
            elif g >= 0.20:
                gross_pts = 20
        except Exception:
            gross_pts = 0

    net_pts = 0
    if net is not None:
        try:
            n = float(net)
            if n > 0.15:
                net_pts = 30
            elif n >= 0.05:
                net_pts = 15
        except Exception:
            net_pts = 0

    fcf_pts = 0
    if fcf_margin is not None:
        try:
            f = float(fcf_margin)
            if f > 0.10:
                fcf_pts = 30
            elif f >= 0.05:
                fcf_pts = 15
        except Exception:
            fcf_pts = 0

    quality = int(gross_pts + net_pts + fcf_pts)

    # Value Score: based mainly on KGV per spec
    pe = info.get('trailingPE') or info.get('forwardPE')
    value = 50
    try:
        if pe is None:
            value = 50
        else:
            p = float(pe)
            if p < 25:
                value = 80
            elif 25 <= p <= 35:
                value = 50
            else:
                value = 30
    except Exception:
        value = 50

    # Momentum Score: RSI base and +20 if price > EMA50
    rsi_val = ratios.get('rsi')
    momentum = 0
    try:
        if rsi_val is None:
            base = 50
        else:
            r = float(rsi_val)
            if 40 <= r <= 60:
                base = 50
            elif r > 60:
                base = 70
            else:
                base = 30
    except Exception:
        base = 50

    momentum = base
    try:
        if ratios.get('close') is not None and ratios.get('ema50') is not None:
            if ratios.get('close') > ratios.get('ema50'):
                momentum = min(100, momentum + 20)
    except Exception:
        pass

    return quality, value, momentum


st.title('Aktienanalyse — Modul 1')

with st.sidebar:
    ticker = st.text_input('Ticker', value='AAPL', max_chars=10)
    st.write('')
    if st.button('Laden'):
        st.rerun()

if not ticker:
    st.error('Bitte einen Ticker eingeben.')
    st.stop()

data = fetch_data(ticker.strip().upper())
info = data['info']
hist = data['history']
fin = data['financials']
balance = data['balance']
cash = data['cashflow']
earnings = data['earnings']
recs = data['recommendations']
ticker_obj = data['ticker_obj']

if hist is None or hist.empty:
    st.error('Keine historischen Daten gefunden. Bitte Ticker prüfen.')
    st.stop()

df = compute_indicators(hist)

# quick ratios dict for scoring
ratios = {
    'rsi': df['RSI'].iloc[-1] if 'RSI' in df.columns else 50,
    'ema20': df['EMA20'].iloc[-1] if 'EMA20' in df.columns else None,
    'ema50': df['EMA50'].iloc[-1] if 'EMA50' in df.columns else None,
    'ema200': df['EMA200'].iloc[-1] if 'EMA200' in df.columns else None,
    'close': df['Close'].iloc[-1],
    'ret_3m': (df['Close'].iloc[-1] / df['Close'].iloc[-63] - 1) if len(df) > 63 else 0,
}

quality, value, momentum = compute_scores(info, fin, ratios)

# Header
col1, col2 = st.columns([3,1])
with col1:
    st.subheader(f"{info.get('longName', ticker)} — {ticker}")
    if info.get('longBusinessSummary'):
        st.markdown(f"<div class='small'>{info.get('longBusinessSummary')[:800]}...</div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='card'><h3>{human(info.get('regularMarketPrice'))}</h3><div class='small'>Aktueller Preis</div></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='card'><h3>{human(info.get('marketCap'))}</h3><div class='small'>Marktkapitalisierung</div></div>", unsafe_allow_html=True)

st.markdown('**Scores**')
sc1, sc2, sc3 = st.columns(3)
def score_gauge(score, title):
    colors = [[0, 'red'], [0.5, 'yellow'], [1, 'green']]
    fig = go.Figure(go.Indicator(
        mode='gauge+number', value=score, title={'text': title},
        gauge={'axis':{'range':[0,100]}, 'bar':{'color':'#00bcd4'}, 'steps':[{'range':[0,50],'color':'#ef4444'},{'range':[50,80],'color':'#f59e0b'},{'range':[80,100],'color':'#16a34a'}]}
    ))
    fig.update_layout(height=220, paper_bgcolor='rgba(0,0,0,0)', font={'color':'#e6eef8'})
    return fig

sc1.plotly_chart(score_gauge(quality, 'Quality'), use_container_width=True)
sc2.plotly_chart(score_gauge(value, 'Value'), use_container_width=True)
sc3.plotly_chart(score_gauge(momentum, 'Momentum'), use_container_width=True)

# show colored RSI badge
rsi_val = ratios.get('rsi')
if rsi_val is not None:
    rsi_col = color_for_rsi(rsi_val)
    st.markdown(f"<div class='card'><div class='metric-label'>RSI</div><h3 style='color:{rsi_col}'>{fmt_num2(rsi_val)}</h3></div>", unsafe_allow_html=True)

# Gesamt-Score and recommendation (weighted)
total_score = int(round(quality * 0.4 + value * 0.3 + momentum * 0.3))
rec_text = ''
rec_class = 'rec-hold'
if total_score > 65:
    rec_class = 'rec-buy'
    # choose strongest metric for short justification
    main = max(('Quality', quality), ('Value', value), ('Momentum', momentum), key=lambda x: x[1])
    rec_text = f"KAUFEN — Gesamt-Score {total_score}. Starke {main[0]} ({main[1]}). Positive Kombination aus Qualität und Momentum."
elif total_score >= 45:
    rec_class = 'rec-hold'
    rec_text = f"HALTEN — Gesamt-Score {total_score}. Gemischte Signale; prüfe Margen und Bewertung."
else:
    rec_class = 'rec-sell'
    rec_text = f"VERKAUFEN — Gesamt-Score {total_score}. Schwache Fundamentaldaten oder Momentum; Risiko hoch."

st.markdown(f"<div class='{rec_class}'><strong style='font-size:18px'>{rec_text}</strong></div>", unsafe_allow_html=True)

st.markdown('**Kennzahlen & Fundamentaldaten**')
# compute/normalize a few metrics for display
net_margin = info.get('netMargins')
if net_margin is None:
    # try to compute net margin = netIncomeToCommon / totalRevenue
    net_income = info.get('netIncomeToCommon') or safe_get(fin, 'Net Income')
    total_rev = info.get('totalRevenue') or safe_get(fin, 'Total Revenue')
    try:
        if net_income is not None and total_rev:
            net_margin = float(net_income) / float(total_rev)
    except Exception:
        net_margin = None

# debtToEquity is often given as percentage-like value (e.g. 150.0 meaning 150%)
debt_raw = info.get('debtToEquity')
debt_display = '—'
try:
    if debt_raw is not None:
        debt_display = f"{float(debt_raw)/100:.2f}"
except Exception:
    debt_display = '—'
cards = st.columns(4)
pe_val = info.get('trailingPE') or info.get('forwardPE')
ps_val = info.get('priceToSalesTrailing12Months')
pb_val = info.get('priceToBook')
ev_val = info.get('enterpriseToEbitda')
cards[0].markdown(f"<div class='kpi'><div class='metric-label'>KGV (PE)</div><h3>{fmt_num2(pe_val) if pe_val is not None else '—'}</h3></div>", unsafe_allow_html=True)
cards[1].markdown(f"<div class='kpi'><div class='metric-label'>KUV (PS)</div><h3>{fmt_num2(ps_val) if ps_val is not None else '—'}</h3></div>", unsafe_allow_html=True)
cards[2].markdown(f"<div class='kpi'><div class='metric-label'>KBV (PB)</div><h3>{fmt_num2(pb_val) if pb_val is not None else '—'}</h3></div>", unsafe_allow_html=True)
cards[3].markdown(f"<div class='kpi'><div class='metric-label'>EV/EBITDA</div><h3>{fmt_num2(ev_val) if ev_val is not None else '—'}</h3></div>", unsafe_allow_html=True)

cards2 = st.columns(4)
gross_num = info.get('grossMargins')
gross_col = color_for_gross(gross_num)
cards2[0].markdown(f"<div class='kpi'><div class='metric-label'>Bruttomarge</div><h3>{colored_span(pct(gross_num), gross_col)}</h3></div>", unsafe_allow_html=True)
net_col = color_for_net(net_margin)
cards2[1].markdown(f"<div class='kpi'><div class='metric-label'>Nettomarge</div><h3>{colored_span(pct(net_margin), net_col)}</h3></div>", unsafe_allow_html=True)
cards2[2].markdown(f"<div class='kpi'><div class='metric-label'>EBITDA-Marge</div><h3>{pct(info.get('ebitdaMargins'))}</h3></div>", unsafe_allow_html=True)
rev_num = info.get('revenueGrowth')
rev_col = color_for_growth(rev_num)
cards2[3].markdown(f"<div class='kpi'><div class='metric-label'>Umsatzwachstum YoY</div><h3>{colored_span(pct(rev_num), rev_col)}</h3></div>", unsafe_allow_html=True)

cards3 = st.columns(4)
fcf_val = info.get('freeCashflow') or safe_get(cash, 'Free Cash Flow')
cards3[0].markdown(f"<div class='kpi'><div class='metric-label'>Free Cash Flow</div><h3>{human(fcf_val) if fcf_val is not None else '—'}</h3></div>", unsafe_allow_html=True)
fcf_margin_num = None
if info.get('totalRevenue'):
    try:
        fcf_margin_num = (info.get('freeCashflow') or 0) / info.get('totalRevenue')
    except Exception:
        fcf_margin_num = None
cards3[1].markdown(f"<div class='kpi'><div class='metric-label'>FCF-Marge</div><h3>{pct(fcf_margin_num)}</h3></div>", unsafe_allow_html=True)
# debt_display: numeric and colored
debt_num = None
try:
    if debt_raw is not None:
        debt_num = float(debt_raw) / 100.0
except Exception:
    debt_num = None
debt_col = color_for_debt(debt_num)
cards3[2].markdown(f"<div class='kpi'><div class='metric-label'>Verschuldungsgrad D/E</div><h3>{colored_span(fmt_num2(debt_num) if debt_num is not None else '—', debt_col)}</h3></div>", unsafe_allow_html=True)

div_raw = info.get('dividendYield')
div_display = f"{float(div_raw):.2f}%" if div_raw is not None else '—'
cards3[3].markdown(f"<div class='kpi'><div class='metric-label'>Dividendenrendite</div><h3>{div_display}</h3></div>", unsafe_allow_html=True)

cards4 = st.columns(4)
cards4[0].markdown(f"<div class='kpi'><div class='metric-label'>52W Hoch</div><h3>{info.get('fiftyTwoWeekHigh') or '—'}</h3></div>", unsafe_allow_html=True)
cards4[1].markdown(f"<div class='kpi'><div class='metric-label'>52W Tief</div><h3>{info.get('fiftyTwoWeekLow') or '—'}</h3></div>", unsafe_allow_html=True)
if info.get('fiftyTwoWeekHigh') and info.get('regularMarketPrice'):
    dist_hi = (info.get('fiftyTwoWeekHigh') - info.get('regularMarketPrice'))/info.get('fiftyTwoWeekHigh')
    cards4[2].markdown(f"<div class='kpi'><div class='metric-label'>Abstand zu 52W Hoch</div><h3>{pct(dist_hi)}</h3></div>", unsafe_allow_html=True)
else:
    cards4[2].markdown(f"<div class='kpi'><div class='metric-label'>Abstand zu 52W Hoch</div><h3>—</h3></div>", unsafe_allow_html=True)
cards4[3].markdown(f"<div class='kpi'><div class='metric-label'>Beta</div><h3>{info.get('beta') or '—'}</h3></div>", unsafe_allow_html=True)

# analyst consensus
analyst_text = ''
try:
    target = info.get('targetMeanPrice')
    rec_text = ''
    if not recs.empty:
        last_year = recs[recs.index > (pd.Timestamp.today() - pd.Timedelta(days=365))]
        counts = last_year['To Grade'].value_counts() if 'To Grade' in last_year.columns else last_year['Action'].value_counts()
        analyst_text = f"Analysten (letztes Jahr): {counts.to_dict()}"
    else:
        analyst_text = 'Keine Empfehlungen verfügbar'
except Exception:
    analyst_text = 'Keine Analystendaten'

st.markdown(f"<div class='card'><h3>Analystenkonsens</h3><div class='small'>Durchschn. Kursziel: {human(target) if target else '—'}<br>{analyst_text}</div></div>", unsafe_allow_html=True)

st.markdown('**Charts**')
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6,0.15,0.25], vertical_spacing=0.03)
fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Preis'), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], line=dict(color='#f97316', width=1), name='EMA20'), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['EMA50'], line=dict(color='#60a5fa', width=1), name='EMA50'), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['EMA200'], line=dict(color='#34d399', width=1), name='EMA200'), row=1, col=1)

fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color='#475569', name='Volumen'), row=2, col=1)

fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#ffd166'), name='RSI'), row=3, col=1)
fig.add_hline(y=70, line=dict(color='red', dash='dash'), row=3, col=1)
fig.add_hline(y=30, line=dict(color='green', dash='dash'), row=3, col=1)

fig.update_layout(template='plotly_dark', height=900, margin=dict(l=40,r=20,t=40,b=20))
fig.update_xaxes(rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

# Revenue & Net income last 4 years
if not fin.empty:
    rev = None
    net = None
    try:
        rev = fin.loc['Total Revenue'] if 'Total Revenue' in fin.index else fin.loc['Revenue'] if 'Revenue' in fin.index else None
    except Exception:
        rev = None
    try:
        net = fin.loc['Net Income'] if 'Net Income' in fin.index else fin.loc['NetIncome'] if 'NetIncome' in fin.index else None
    except Exception:
        net = None
    if rev is not None and net is not None:
        rev = rev.dropna()
        net = net.dropna()
        years = rev.index.astype(str).tolist()[:4]
        rev_vals = [rev[col] for col in years]
        net_vals = [net[col] for col in years]
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=years, y=rev_vals, name='Umsatz', marker_color='#60a5fa'))
        fig2.add_trace(go.Bar(x=years, y=net_vals, name='Nettogewinn', marker_color='#34d399'))
        fig2.update_layout(barmode='group', template='plotly_dark', title='Umsatz & Gewinn (letzte Jahre)')
        st.plotly_chart(fig2, use_container_width=True)

st.markdown('---')
st.markdown('App lädt Live-Daten von yfinance. Starte die App mit:')
st.code('streamlit run app.py', language='bash')


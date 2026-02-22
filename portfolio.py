import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Portfolio Dashboard", layout="wide")

_CSS = '''
<style>
body { background-color: #0b1220; color: #e6edf3; }
.stApp {
  background: linear-gradient(180deg,#0b1220 0%, #07101a 100%);
  color: #e6edf3;
}
.card { background: #0f1721; padding: 14px; border-radius: 8px; }
.metric { color: #e6edf3; }
.small { color: #9aa6b2; font-size:12px }
</style>
'''

st.markdown(_CSS, unsafe_allow_html=True)

DATA = [
    ("Alphabet","Tech","USA",204866),
    ("Nvidia","Halbleiter","USA",180917),
    ("Apple","Tech","USA",172736),
    ("Meta","Tech","USA",157478),
    ("Microsoft","Tech","USA",146750),
    ("ASML","Halbleiter","NL",123860),
    ("Visa","Fintech","USA",109515),
    ("Eli Lilly","Health","USA",90467),
    ("Procter & Gamble","Konsum","USA",89388),
    ("Applied Materials","Halbleiter","USA",82620),
    ("Johnson & Johnson","Health","USA",78595),
    ("Mastercard","Fintech","USA",78004),
    ("Roche","Health","CH",76903),
    ("Caterpillar","Industrie","USA",73006),
    ("Lam Research","Halbleiter","USA",72969),
    ("Costco","Konsum","USA",63069),
    ("Cisco","Tech","USA",60062),
    ("Linde","Industrie","IE",57356),
    ("Merck & Co","Health","USA",53409),
    ("Rio Tinto","Rohstoffe","UK",52512),
    ("BHP","Rohstoffe","AUS",51984),
    ("Emerson","Industrie","USA",51677),
    ("Walmart","Konsum","USA",49707),
    ("Coca-Cola","Konsum","USA",48989),
    ("Netflix","Tech","USA",47208),
    ("GE Aerospace","Industrie","USA",44519),
    ("Analog Devices","Halbleiter","USA",43862),
    ("ConocoPhillips","Energie","USA",43386),
    ("Novartis","Health","CH",42441),
    ("Progressive","Finanzen","USA",42046),
    ("DSV","Logistik","DK",41932),
    ("AstraZeneca","Health","UK",41751),
    ("Atlas Copco","Industrie","SE",40389),
    ("Eaton","Industrie","USA",40141),
    ("ABB","Industrie","CH",39849),
    ("Hilton","Konsum","USA",37334),
    ("Citigroup","Finanzen","USA",37300),
    ("Corning","Industrie","USA",36777),
    ("Ametek","Industrie","USA",36595),
    ("Schneider Electric","Industrie","FR",36575),
    ("Rheinmetall","Defense","DE",33880),
    ("Amazon","Tech","USA",33713),
    ("Broadcom","Halbleiter","USA",32379),
    ("GSK","Health","UK",31716),
    ("Berkshire Hathaway","Holding","USA",31600),
    ("CME Group","Börse","USA",30650),
    ("L'Oréal","Konsum","FR",29617),
    ("Wells Fargo","Finanzen","USA",29158),
    ("Novo Nordisk","Health","DK",28332),
    ("Danaher","Health","USA",28171),
    ("Amphenol","Industrie","USA",28061),
    ("Vertex","Health","USA",27793),
    ("Safran","Industrie","FR",27672),
    ("Siemens","Industrie","DE",26928),
    ("Abbott","Health","USA",26302),
    ("Gallagher","Finanzen","USA",26256),
    ("Nestlé","Konsum","CH",26207),
    ("Barclays","Finanzen","UK",25990),
    ("Intuitive Surgical","Health","USA",25365),
    ("Morgan Stanley","Finanzen","USA",22362),
    ("Teradyne","Halbleiter","USA",21794),
    ("Uber","Tech","USA",21195),
    ("Fastenal","Industrie","USA",20961),
    ("Snowflake","Tech","USA",20811),
    ("Stryker","Health","USA",20705),
    ("Palo Alto","Tech","USA",20579),
    ("Ferguson","Industrie","USA",19755),
    ("Southern Copper","Rohstoffe","USA",18704),
    ("Hermes","Luxus","FR",18540),
    ("ADP","IT Services","USA",18337),
    ("Intuit","Software","USA",18088),
    ("Cadence","Software","USA",16737),
    ("IBM","Tech","USA",16512),
    ("KLA","Halbleiter","USA",16246),
    ("IDEXX","Health","USA",16080),
    ("Dell","Tech","USA",15281),
    ("Nasdaq","Finanzen","USA",14907),
    ("Autodesk","Software","USA",14546),
    ("Booking","Travel","USA",14419),
]

df = pd.DataFrame(DATA, columns=["Company","Sector","Country","Value_EUR"])
df["Value_EUR"] = df["Value_EUR"].astype(float)
total_value = df["Value_EUR"].sum()
df["Weight_%"] = df["Value_EUR"] / total_value * 100

def region_from_country(c):
    europe = {"NL","CH","IE","UK","SE","FR","DE","DK"}
    if c == "USA":
        return "USA"
    if c in europe:
        return "Europe"
    if c == "AUS":
        return "Oceania"
    return "Other"

df["Region"] = df["Country"].apply(region_from_country)

st.title("Portfolio Dashboard")

# Header metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.metric("Gesamtwert Portfolio (EUR)", f"{int(total_value):,} €")
    st.markdown("</div>", unsafe_allow_html=True)
with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.metric("Anzahl Positionen", len(df))
    st.markdown("</div>", unsafe_allow_html=True)
with col3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.metric("Anzahl Branchen", df["Sector"].nunique())
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# Charts: Sector donut + Top10
sector_df = df.groupby("Sector", as_index=False)["Value_EUR"].sum().sort_values("Value_EUR", ascending=False)
top10 = df.sort_values("Value_EUR", ascending=False).head(10).copy()

col_a, col_b = st.columns([1,1])
with col_a:
    st.subheader("Branchengewichtung")
    fig = px.pie(sector_df, names="Sector", values="Value_EUR", hole=0.45,
                 color_discrete_sequence=px.colors.qualitative.Dark24)
    fig.update_layout(template="plotly_dark", margin=dict(t=30,b=10,l=10,r=10), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)
with col_b:
    st.subheader("Top 10 Positionen")
    fig2 = px.bar(top10[::-1], x="Value_EUR", y="Company", orientation='h',
                  text=top10[::-1]["Value_EUR"].map(lambda x: f"{int(x):,} €"),
                  color=top10[::-1]["Value_EUR"], color_continuous_scale=px.colors.sequential.Plasma)
    fig2.update_layout(template="plotly_dark", margin=dict(t=30,b=10,l=10,r=10), showlegend=False)
    fig2.update_xaxes(title_text="Wert (EUR)")
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# Regions pie
st.subheader("Länder-Verteilung")
region_df = df.groupby("Region", as_index=False)["Value_EUR"].sum()
fig3 = px.pie(region_df, names="Region", values="Value_EUR", hole=0.3, color_discrete_sequence=px.colors.sequential.Darkmint)
fig3.update_layout(template="plotly_dark", margin=dict(t=30,b=10,l=10,r=10))
st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

# Positions table
st.subheader("Positions-Tabelle")
table_df = df.sort_values("Value_EUR", ascending=False).reset_index(drop=True)
table_df_display = table_df.copy()
table_df_display["Value_EUR"] = table_df_display["Value_EUR"].map(lambda x: f"{int(x):,} €")
table_df_display["Weight_%"] = table_df_display["Weight_%"].map(lambda x: f"{x:.2f}%")
st.dataframe(table_df_display.style.set_properties(**{"background-color":"#0f1721","color":"#e6edf3"}), height=500)

st.markdown("---")

# Concentration analysis
st.subheader("Konzentrations-Analyse")
top5_sum = table_df.head(5)["Value_EUR"].sum()
top5_pct = top5_sum / total_value * 100
colx, coly = st.columns(2)
with colx:
    st.metric("Top 5 Wert (EUR)", f"{int(top5_sum):,} €")
with coly:
    st.metric("Top 5 Anteil am Portfolio", f"{top5_pct:.2f}%")

st.markdown("---")

st.markdown("**Hinweis:** Starte das Dashboard lokal mit: `streamlit run portfolio.py`")

if __name__ == '__main__':
    pass

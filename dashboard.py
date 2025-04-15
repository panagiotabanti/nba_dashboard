import streamlit as st
from datetime import datetime
from standings import get_clean_standings
from teamStats import show_top_teams_by_ppg, show_team_stat_dropdown
from playerComparison import player_comparison_section, show_player_shot_chart
from advancedAnalytics import show_advanced_team_analytics

# --- PAGE CONFIG ---
st.set_page_config(page_title="NBA Dashboard", layout="wide")

# --- SIDEBAR ---
st.sidebar.image("https://cdn.nba.com/logos/nba/nba-logo.svg", width=100)
st.sidebar.title("NBA Dashboard")
st.sidebar.markdown("Compare players, view team stats, track standings, and follow the league.")

# --- ANALYTICS STORIES ---
st.sidebar.markdown("### 🧠 Analytics Stories")

st.sidebar.markdown("""
- 🏀 **High AST% = Efficient Offense**  
  Teams with higher Assist % tend to have better Offensive Ratings, showing the value of ball movement.

- 🚀 **Fast-Paced West**  
  Clustering analysis suggests Western Conference teams play at a higher Pace on average.

- 🧮 **Turnovers Hurt Efficiency**  
  Teams with high Turnover % often rank lower in Net Rating, highlighting the importance of ball security.

- 📈 **Offense Drives Success**  
  Offensive Rating has the strongest correlation with Net Rating compared to other advanced metrics.
""")


theme = st.sidebar.radio("Theme", ["Light", "Dark"], index=0)
season = st.sidebar.selectbox("Select Season:", ["2023-24", "2022-23", "2021-22", "2020-21"])

# --- THEME STYLING ---
if theme == "Light":
    background_color = "#f0f2f6"
    text_color = "#000000"
    cmap = "Blues"
    st.write("<style>body { background-color: #f0f2f6; }</style>", unsafe_allow_html=True)
else:
    background_color = "#1e1e1e"
    text_color = "#ffffff"
    cmap = "Greys"
    st.write("<style>body { background-color: #1e1e1e; color: #ffffff; }</style>", unsafe_allow_html=True)

# --- TITLE & SEASON INFO ---
st.title("🏀 NBA Performance Dashboard")
st.markdown(f"**Season:** {season} | **Theme:** {theme}")
st.markdown("---")

# --- MAIN CONTENT TABS ---
tab1, tab2, tab3 = st.tabs([
    "📈 Team Stats",
    "📊 Advanced Analytics",
    "🤼 Player Comparison"
])

# --- TAB 1: TEAM STATS ---
with tab1:
    st.subheader("📊 NBA Team Standings (By Win %)")
    standings_df = get_clean_standings(season)
    if standings_df.empty:
        st.warning("Could not load standings.")
    else:
        st.dataframe(standings_df, use_container_width=True)

    show_top_teams_by_ppg(season)
    show_team_stat_dropdown(season)

# --- TAB 2: ADVANCED ANALYTICS ---
with tab2:
    show_advanced_team_analytics(season)

# --- TAB 3: PLAYER COMPARISON  ---
with tab3:
    player_comparison_section(theme, suffix="_tab3")
    show_player_shot_chart()

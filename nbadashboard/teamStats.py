from nba_api.stats.endpoints import leaguedashteamstats
import streamlit as st
import matplotlib.pyplot as plt

def show_top_teams_by_ppg(season):
    st.subheader("🔥 Top 5 Teams by Points Per Game")
    try:
        df = leaguedashteamstats.LeagueDashTeamStats(season=season).get_data_frames()[0]
        top5 = df[['TEAM_NAME', 'PTS']].sort_values(by='PTS', ascending=False).head(5)

        fig, ax = plt.subplots(figsize=(10, 6))  # Wider plot
        ax.bar(top5['TEAM_NAME'], top5['PTS'], color='orange')
        ax.set_title(f"Top 5 Teams (Season {season})")
        ax.set_ylabel("Points Per Game")

        # Rotate x-axis labels
        ax.set_xticklabels(top5['TEAM_NAME'], rotation=30, ha='right')

        st.pyplot(fig)
    except:
        st.warning("Could not load top 5 scoring teams.")


def show_team_stat_dropdown(season):
    st.subheader("📈 Team Stats Viewer")
    try:
        df = leaguedashteamstats.LeagueDashTeamStats(season=season).get_data_frames()[0]
        team = st.selectbox("Select a team:", sorted(df['TEAM_NAME'].unique()))
        selected = df[df['TEAM_NAME'] == team].T
        selected.columns = [team]
        st.dataframe(selected, use_container_width=True)
    except:
        st.warning("Could not load team stats.")

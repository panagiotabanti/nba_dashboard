import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from nba_api.stats.static import players
from nba_api.stats.endpoints import playercareerstats

# Get full player list once
player_list = players.get_players()
player_names = sorted([p['full_name'] for p in player_list])

# Fetch regular season player stats
def get_player_stats(name):
    for p in player_list:
        if p['full_name'] == name:
            stats = playercareerstats.PlayerCareerStats(player_id=p['id']).get_data_frames()[0]
            return stats[~stats['SEASON_ID'].str.contains('P')]  # Exclude playoff stats
    return pd.DataFrame()

# Compute career averages
def get_averages(df):
    if df.empty or df['GP'].sum() == 0:
        return None
    totals = df[['GP', 'PTS', 'AST', 'REB', 'FG_PCT', 'FG3_PCT', 'FT_PCT']].sum()
    games = df['GP'].sum()
    return {
        'PTS': totals['PTS'] / games,
        'AST': totals['AST'] / games,
        'REB': totals['REB'] / games,
        'FG%': df['FG_PCT'].mean(skipna=True),
        '3P%': df['FG3_PCT'].mean(skipna=True),
        'FT%': df['FT_PCT'].mean(skipna=True)
    }

# Streamlit section for comparing two players
def player_comparison_section(theme, suffix=""):
    st.markdown("---")
    st.subheader("👥 Compare Player Stats")

    col1, col2 = st.columns(2)
    with col1:
        p1 = st.selectbox("Select Player 1:", player_names, key=f'p1{suffix}')
    with col2:
        p2 = st.selectbox("Select Player 2:", player_names, key=f'p2{suffix}')

    if p1 and p2 and p1 != p2:
        df1 = get_player_stats(p1)
        df2 = get_player_stats(p2)

        avg1 = get_averages(df1)
        avg2 = get_averages(df2)

        if avg1 and avg2:
            st.subheader("📊 Career Averages")
            avg_df = pd.DataFrame({p1: avg1, p2: avg2})
            cmap = "Blues" if theme == "Light" else "Greys"
            st.dataframe(avg_df.style.format("{:.2f}").background_gradient(axis=1, cmap=cmap))

            st.subheader("📉 Stat Comparison")
            labels = list(avg1.keys())
            vals1 = list(avg1.values())
            vals2 = list(avg2.values())

            fig, ax = plt.subplots()
            bar_width = 0.35
            index = range(len(labels))

            ax.bar(index, vals1, bar_width, label=p1)
            ax.bar([i + bar_width for i in index], vals2, bar_width, label=p2)
            ax.set_xticks([i + bar_width / 2 for i in index])
            ax.set_xticklabels(labels)
            ax.set_ylabel("Per Game Average")
            ax.set_title("Career Stat Comparison")
            ax.legend()
            st.pyplot(fig)
        else:
            st.error("Not enough data to compare these players.")
    elif p1 == p2:
        st.info("Please select two different players to compare.")

from nba_api.stats.static import players
from nba_api.stats.endpoints import shotchartdetail
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st

def draw_court(ax=None):
    if ax is None:
        ax = plt.gca()

    # Hoop
    hoop = plt.Circle((0, 0), radius=7.5, linewidth=2, color='black', fill=False)
    ax.add_patch(hoop)

    # Backboard
    ax.plot([-30, 30], [0, 0], linewidth=2, color='black')

    # Paint
    paint = plt.Rectangle((-80, -47.5), 160, 190, linewidth=2, color='black', fill=False)
    ax.add_patch(paint)

    # Free-throw circle
    free_throw = plt.Circle((0, 142.5), 60, linewidth=2, color='black', fill=False)
    ax.add_patch(free_throw)

    # Three-point arc
    three_arc = plt.Circle((0, 0), 237.5, linewidth=2, color='black', fill=False)
    ax.add_patch(three_arc)

    ax.set_xlim(-250, 250)
    ax.set_ylim(-50, 470)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Player Shot Chart")

def show_player_shot_chart():
    st.markdown("---")
    st.subheader("🏀 Player Shot Chart")

    player_name = st.text_input("Enter full player name (e.g., Stephen Curry):", key="shot_input")
    season = st.selectbox("Select Season", ["2023-24", "2022-23", "2021-22", "2020-21"], key="shot_season")

    if player_name:
        player_dict = players.find_players_by_full_name(player_name)
        if player_dict:
            player_id = player_dict[0]['id']

            try:
                response = shotchartdetail.ShotChartDetail(
                    team_id=0,
                    player_id=player_id,
                    season_type_all_star='Regular Season',
                    season_nullable=season
                )
                df = response.get_data_frames()[0]

   


                # 🔍 Debug output
                st.write("Raw Response Columns:", df.columns.tolist())
                st.write("Sample Rows:", df.head())
                st.write("Number of Shots:", len(df))

                if df.empty:
                    st.warning("No shot data available for this player and season.")
                    return

                # 🏀 Plot the shot chart
                fig, ax = plt.subplots(figsize=(6, 6))
                draw_court(ax)
                sns.scatterplot(data=df, x="LOC_X", y="LOC_Y", hue="SHOT_MADE_FLAG", ax=ax, palette={1: "green", 0: "red"}, legend=False)
                ax.set_title(f"{player_name} Shot Chart ({season})")
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Error retrieving shot data: {str(e)}")
        else:
            st.warning("Player not found.")

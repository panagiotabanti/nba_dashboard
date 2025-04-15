from nba_api.stats.endpoints import scoreboardv2
from datetime import datetime
import streamlit as st

def display_live_games():
    try:
        today = datetime.today().strftime('%m/%d/%Y')
        live_games = scoreboardv2.ScoreboardV2(game_date=today).get_normalized_dict()
        games = live_games['GameHeader']
        scores = live_games['LineScore']

        if not games:
            st.info("No games today.")
            return

        for game in games:
            game_id = game['GAME_ID']
            game_status = game['GAME_STATUS_TEXT']
            game_scores = [s for s in scores if s['GAME_ID'] == game_id]
            if len(game_scores) == 2:
                t1, t2 = game_scores
                st.markdown(f"**{t1['TEAM_ABBREVIATION']} {t1['PTS']} - {t2['PTS']} {t2['TEAM_ABBREVIATION']}** — *{game_status}*")

        st.caption(f"Checked on: {today}")
    except:
        st.warning("Live game data not available.")

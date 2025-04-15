from nba_api.stats.endpoints import leaguestandings
import pandas as pd

def get_clean_standings(season='2023-24'):
    try:
        data = leaguestandings.LeagueStandings(season=season).get_data_frames()[0]

        # Debug: print column names once
        print("Standings columns:", data.columns.tolist())

        # Use actual column names — here's the typical correct list:
        cleaned = data[['TeamName', 'Conference', 'WINS', 'LOSSES', 'WinPCT', 'PlayoffRank']]
        cleaned = cleaned.rename(columns={'PlayoffRank': 'ConferenceRank'})
        cleaned.sort_values(by=['Conference', 'ConferenceRank'], inplace=True)
        return cleaned
    except Exception as e:
        print(f"[ERROR] Failed to fetch standings: {e}")
        return pd.DataFrame()

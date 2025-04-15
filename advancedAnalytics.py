# --- Imports ---
from nba_api.stats.endpoints import leaguedashteamstats
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from statsmodels.tsa.holtwinters import ExponentialSmoothing



# ---------------------------------------------
# Get Advanced Team Stats for a Selected Season
# ---------------------------------------------
def get_advanced_team_stats(season='2023-24'):
    response = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        measure_type_detailed_defense='Advanced'
    )
    df = response.get_data_frames()[0]
    try:
        return df[[
            'TEAM_NAME', 'OFF_RATING', 'DEF_RATING', 'NET_RATING',
            'PACE', 'AST_PCT', 'TM_TOV_PCT'
        ]]
    except KeyError:
        st.error("Some expected columns are missing. Check column names above.")
        return df

# ------------------------
# Format Metric Names for UI
# ------------------------
def format_metric_name(metric):
    name_map = {
        'OFF_RATING': 'Offensive Rating',
        'DEF_RATING': 'Defensive Rating',
        'NET_RATING': 'Net Rating',
        'PACE': 'Pace',
        'AST_PCT': 'Assist %',
        'TM_TOV_PCT': 'Turnover %'
    }
    return name_map.get(metric, metric.replace('_', ' ').title())

# --------------------------------
# Trends Across Seasons Chart
# --------------------------------
def plot_metric_trends_across_seasons(metric, selected_teams):
    seasons = ['2020-21', '2021-22', '2022-23', '2023-24']
    trend_data = []

    for season in seasons:
        df = get_advanced_team_stats(season)
        for team in selected_teams:
            row = df[df['TEAM_NAME'] == team]
            if not row.empty:
                trend_data.append({
                    'Season': season,
                    'Team': team,
                    'Value': row.iloc[0][metric]
                })

    trend_df = pd.DataFrame(trend_data)

    st.write(f"### {format_metric_name(metric)} Trends Across Seasons")
    fig, ax = plt.subplots(figsize=(10, 4))
    for team in selected_teams:
        team_data = trend_df[trend_df['Team'] == team]
        ax.plot(team_data['Season'], team_data['Value'], label=team, marker='o')

    # Event Annotations
    events = {
        '2021-22': "Trade: Player X",
        '2022-23': "Injury: Key PG",
        '2023-24': "Coaching Change"
    }
    for season, label in events.items():
        if season in trend_df['Season'].values:
            ax.axvline(season, color='gray', linestyle='--', linewidth=1)
            ax.text(season, ax.get_ylim()[1]*0.95, label, rotation=90, fontsize=8, color='gray', ha='center')

    ax.set_ylabel(format_metric_name(metric))
    ax.set_xlabel('Season')
    ax.legend()
    st.pyplot(fig)

# ------------------------
# Correlation Scatter Plot
# ------------------------
def show_metric_correlation(df, x_metric, y_metric):
    st.write(f"### Correlation: {format_metric_name(x_metric)} vs {format_metric_name(y_metric)}")

    playoff_teams = [
        "Boston Celtics", "Milwaukee Bucks", "Philadelphia 76ers", "New York Knicks",
        "Cleveland Cavaliers", "Indiana Pacers", "Miami Heat", "Orlando Magic",
        "Denver Nuggets", "Minnesota Timberwolves", "Oklahoma City Thunder",
        "Dallas Mavericks", "Los Angeles Clippers", "Phoenix Suns", "Los Angeles Lakers",
        "New Orleans Pelicans", "Sacramento Kings", "Golden State Warriors"
    ]
    use_playoff_teams = st.checkbox("Only show playoff teams", value=False)

    if use_playoff_teams:
        df = df[df['TEAM_NAME'].isin(playoff_teams)]

    if len(df) >= 2:
        corr, _ = pearsonr(df[x_metric], df[y_metric])
        st.caption(f"**Pearson Correlation (r):** {corr:.3f}")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.regplot(x=df[x_metric], y=df[y_metric], ax=ax)
    for i, row in df.iterrows():
        ax.text(row[x_metric], row[y_metric], row['TEAM_NAME'], fontsize=8)
    ax.set_xlabel(format_metric_name(x_metric))
    ax.set_ylabel(format_metric_name(y_metric))
    st.pyplot(fig)

# -------------------------------
# Team Clustering with KMeans
# -------------------------------
def show_team_clustering(df):
    st.markdown("---")
    st.subheader("🧠 Team Style Clustering")

    features = ['PACE', 'OFF_RATING', 'DEF_RATING', 'AST_PCT', 'TM_TOV_PCT']
    X = df[features].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    num_clusters = st.slider("Select number of clusters", 2, 5, 3)

    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    st.dataframe(df[['TEAM_NAME', 'Cluster']].sort_values('Cluster'))

    fig, ax = plt.subplots(figsize=(8, 6))
    for cluster in sorted(df['Cluster'].unique()):
        subset = df[df['Cluster'] == cluster]
        ax.scatter(subset['PACE'], subset['OFF_RATING'], label=f'Cluster {cluster}')
        for _, row in subset.iterrows():
            ax.text(row['PACE'], row['OFF_RATING'], row['TEAM_NAME'], fontsize=7)

    ax.set_xlabel("Pace")
    ax.set_ylabel("Offensive Rating")
    ax.set_title("Team Clusters (based on playing style)")
    ax.legend()
    st.pyplot(fig)

# -----------------------------------------
# Main Entry Function for the Dashboard
# -----------------------------------------
def show_advanced_team_analytics(season):
    st.subheader("📊 Advanced Team Analytics")
    df = get_advanced_team_stats(season)
    st.dataframe(df.sort_values(by='NET_RATING', ascending=False))

    metrics = ['OFF_RATING', 'DEF_RATING', 'NET_RATING', 'PACE', 'AST_PCT', 'TM_TOV_PCT']

    for metric in metrics:
        if metric not in df.columns:
            st.warning(f"Metric '{metric}' not found in data.")
            continue

        st.write(f"### {format_metric_name(metric)} by Team")
        fig, ax = plt.subplots(figsize=(10, 4))
        df_sorted = df.sort_values(by=metric, ascending=False)
        ax.bar(df_sorted['TEAM_NAME'], df_sorted[metric])
        ax.set_ylabel(format_metric_name(metric))
        ax.set_xticklabels(df_sorted['TEAM_NAME'], rotation=45, ha='right')
        st.pyplot(fig)

    # Trend line
    st.markdown("---")
    st.subheader("📈 Compare Metrics Over Seasons")
    all_teams = df['TEAM_NAME'].tolist()
    selected_teams = st.multiselect("Select up to 3 teams to compare:", all_teams, default=all_teams[:2])
    selected_metric = st.selectbox("Choose a metric:", metrics, format_func=format_metric_name)
    if selected_teams:
        plot_metric_trends_across_seasons(selected_metric, selected_teams)

    # Correlation
    st.markdown("---")
    st.subheader("📉 Correlation Between Metrics")
    col1, col2 = st.columns(2)
    with col1:
        x_metric = st.selectbox("X-axis metric", metrics, key="x", format_func=format_metric_name)
    with col2:
        y_metric = st.selectbox("Y-axis metric", metrics, key="y", format_func=format_metric_name)
    if x_metric != y_metric:
        show_metric_correlation(df, x_metric, y_metric)
    else:
        st.info("Select two different metrics to view correlation.")

    
    show_team_clustering(df)

    show_correlation_heatmap(df)

    show_net_rating_prediction(df)

    forecast_off_rating(df)






def show_correlation_heatmap(df):
    st.markdown("---")
    st.subheader("🧠 Full Metric Correlation Heatmap")

    # Select numeric features only
    features = ['OFF_RATING', 'DEF_RATING', 'NET_RATING', 'PACE', 'AST_PCT', 'TM_TOV_PCT']
    corr_matrix = df[features].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True,
                xticklabels=[format_metric_name(f) for f in features],
                yticklabels=[format_metric_name(f) for f in features],
                ax=ax)

    ax.set_title("Correlation Between Advanced Metrics")
    st.pyplot(fig)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

def show_net_rating_prediction(df):
    st.markdown("---")
    st.subheader("📈 Predicting Net Rating with Linear Regression")

    features = ['OFF_RATING', 'DEF_RATING', 'PACE', 'AST_PCT', 'TM_TOV_PCT']
    target = 'NET_RATING'

    X = df[features]
    y = df[target]

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    # Evaluation
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    st.caption(f"**R² (Explained Variance):** {r2:.3f} | **MAE:** {mae:.2f}")

    # Feature weights
    st.markdown("#### 🔍 Feature Importance (Model Coefficients)")
    coef_df = pd.DataFrame({
        'Feature': [format_metric_name(f) for f in features],
        'Coefficient': model.coef_
    }).sort_values(by='Coefficient', ascending=False)
    st.dataframe(coef_df)

    # Plot actual vs predicted
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y, y_pred)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2)
    ax.set_xlabel("Actual Net Rating")
    ax.set_ylabel("Predicted Net Rating")
    ax.set_title("Actual vs Predicted Net Rating")
    st.pyplot(fig)


def forecast_off_rating(df):
    st.markdown("---")
    st.subheader("📊 Forecasting Next Season's Offensive Rating")

    teams = df['TEAM_NAME'].tolist()
    selected_team = st.selectbox("Select a team to forecast:", teams)

    # Build time series across seasons
    seasons = ['2020-21', '2021-22', '2022-23', '2023-24']
    ratings = []

    for season in seasons:
        season_df = get_advanced_team_stats(season)
        row = season_df[season_df['TEAM_NAME'] == selected_team]
        if not row.empty:
            ratings.append(row.iloc[0]['OFF_RATING'])
        else:
            ratings.append(None)

    if len(ratings) < 2 or None in ratings:
        st.warning("Not enough data to generate forecast.")
        return

    series = pd.Series(ratings, index=pd.Index(seasons))

    # Fit ETS model
    model = ExponentialSmoothing(series, trend='add', seasonal=None)
    fit = model.fit()
    forecast = fit.forecast(1)

    # Show results
    all_seasons = seasons + ['2024-25']
    all_values = list(series.values) + list(forecast)

    st.write(f"### Forecasted Offensive Rating for {selected_team} in 2024-25: **{forecast.values[0]:.2f}**")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(all_seasons, all_values, marker='o', label='Offensive Rating')
    ax.axvline(x='2023-24', color='gray', linestyle='--')
    ax.annotate('Forecast', xy=('2024-25', forecast.values[0]), xytext=(-15, 10),
                textcoords='offset points', arrowprops=dict(arrowstyle='->'), fontsize=9)
    ax.set_ylabel("Offensive Rating")
    ax.set_title(f"{selected_team}: Offensive Rating Forecast")
    st.pyplot(fig)

import streamlit as st
import pandas as pd
import numpy as np
from pulp import *
import requests
import joblib
import os
import random

# --- Configuration ---
st.set_page_config(layout="wide", page_title="FPL AI Predictor Live")

# --- Constants & FPL API Endpoints ---
FPL_API_BASE_URL = "https://fantasy.premierleague.com/api/"
FPL_BOOTSTRAP_STATIC = FPL_API_BASE_URL + "bootstrap-static/"
FPL_FIXTURES = FPL_API_BASE_URL + "fixtures/"
FPL_GAMEWEEK_STATUS = FPL_API_BASE_URL + "event-status/"

# --- Data Fetching ---
@st.cache_data(ttl=3600) # Cache data for 1 hour to avoid hitting API too often
def fetch_fpl_data():
    """
    Fetches general FPL data (players, teams, elements types)
    and returns player DataFrame along with a team ID to name mapping.
    """
    try:
        response = requests.get(FPL_BOOTSTRAP_STATIC)
        response.raise_for_status()
        data = response.json()

        elements_df = pd.DataFrame(data['elements'])
        teams_df = pd.DataFrame(data['teams'])
        element_types_df = pd.DataFrame(data['element_types'])

        # Rename the 'name' column in teams_df to 'team_name' *before* merging.
        teams_df_renamed = teams_df.rename(columns={'name': 'team_name'})

        # Merge elements_df with teams_df to get team_name and also keep original team_id
        # The 'team' column in elements_df is the team_id.
        # Use suffixes to clearly differentiate IDs if there were conflicts, but for 'id' and 'team_name' it's fine.
        players_df = elements_df.merge(teams_df_renamed[['id', 'team_name']], left_on='team', right_on='id', suffixes=('_player_id_dummy', '_team_id_dummy'))

        # Merge for position (element_types)
        players_df = players_df.merge(element_types_df[['id', 'singular_name_short']], left_on='element_type', right_on='id', suffixes=('', '_pos'))

        # Select relevant columns and rename for clarity
        # 'id' is the player's FPL ID. 'team' is the team ID. 'team_name' is the team's full name.
        players_df = players_df[[
            'id', # This is the player's unique FPL ID from elements_df
            'first_name', 'second_name', 'web_name', 'now_cost',
            'singular_name_short', 'team_name', 'points_per_game', 'form',
            'status', 'total_points', 'team' # 'team' here is the original team_id from elements_df
        ]]
        players_df.columns = [
            'id', # Renamed to 'id' for player unique ID
            'first_name', 'second_name', 'name', 'cost',
            'position', 'team_name', 'ppg', 'form', # 'team_name' is the full team name
            'status', 'total_points_season', 'team_id' # Renamed 'team' to 'team_id' for clarity
        ]
        
        # Convert cost to millions
        players_df['cost'] = players_df['cost'] / 10.0

        # Convert 'ppg' and 'form' to numeric, handling potential errors
        players_df['ppg'] = pd.to_numeric(players_df['ppg'], errors='coerce').fillna(0.0)
        players_df['form'] = pd.to_numeric(players_df['form'], errors='coerce').fillna(0.0)

        # Filter out unavailable players (injured, suspended, etc.) for initial selection
        players_df['is_available_for_selection'] = ~players_df['status'].isin(['i', 'n', 's'])

        # Create a team ID to name mapping (and name to ID mapping for convenience)
        team_id_to_name = {team['id']: team['name'] for team in data['teams']}
        
        st.success(f"Successfully fetched FPL data for {len(players_df)} players.")
        return players_df[players_df['is_available_for_selection']].copy(), team_id_to_name
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching FPL data: {e}. Please try again later.")
        return pd.DataFrame(), {}

@st.cache_data(ttl=3600)
def fetch_fpl_fixtures():
    """
    Fetches current and future FPL fixtures.
    Returns raw fixture data with team_h and team_a IDs.
    """
    try:
        response = requests.get(FPL_FIXTURES)
        response.raise_for_status()
        fixtures_data = response.json()
        
        st.success(f"Successfully fetched fixture data.")
        return fixtures_data
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching fixture data: {e}. Some chip advice may be unavailable.")
        return []

@st.cache_data(ttl=600) # Cache for 10 minutes
def fetch_current_gameweek():
    """Fetches the current gameweek."""
    try:
        response = requests.get(FPL_GAMEWEEK_STATUS)
        response.raise_for_status()
        data = response.json()
        
        # Find the current gameweek (event) that is 'is_current' or 'is_next'
        current_gw_id = 1
        for event in data.get('events', []):
            if event.get('is_current', False) or event.get('is_next', False):
                current_gw_id = event['id']
                break
        
        st.success(f"Current Gameweek determined: GW {current_gw_id}")
        return current_gw_id
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching gameweek status: {e}. Defaulting to GW 1.")
        return 1

# --- Player Prediction (Simulated ML Model) ---
class SimpleFPLPredictor:
    def predict(self, df):
        df['predicted_points'] = (df['cost'] * 0.5 + df['form'].astype(float) * 1.5 + df['ppg'].astype(float) * 0.8)
        df['predicted_points'] = df['predicted_points'] + np.random.normal(0, 1.5, len(df))
        df['predicted_points'] = df['predicted_points'].apply(lambda x: max(0, min(x, 20))).round(1)
        return df['predicted_points']

@st.cache_resource
def load_prediction_model():
    st.sidebar.info("Using a simplified player prediction model for this demo.")
    return SimpleFPLPredictor()

# --- Team Optimization (PuLP) ---
@st.cache_data(show_spinner="Optimizing your dream team...", experimental_allow_widgets=True)
def optimize_team(players_df, budget=100.0):
    if players_df.empty:
        return pd.DataFrame(), 0, 0, "No Data"

    eligible_players = players_df[players_df['status'].isin(['a', 'd'])].copy()

    if eligible_players.empty:
        return pd.DataFrame(), 0, 0, "No Eligible Players"

    prob = LpProblem("FPL Squad Selection", LpMaximize)
    player_vars = LpVariable.dicts("Player", eligible_players.index, 0, 1, LpBinary)

    prob += lpSum(eligible_players.loc[i, 'predicted_points'] * player_vars[i] for i in eligible_players.index), "Total Predicted Points"

    prob += lpSum(player_vars[i] for i in eligible_players.index) == 15, "Total Players"
    prob += lpSum(eligible_players.loc[i, 'cost'] * player_vars[i] for i in eligible_players.index) <= budget, "Total Cost"

    prob += lpSum(player_vars[i] for i in eligible_players.index if eligible_players.loc[i, 'position'] == 'GKP') == 2, "Goalkeepers"
    prob += lpSum(player_vars[i] for i in eligible_players.index if eligible_players.loc[i, 'position'] == 'DEF') == 5, "Defenders"
    prob += lpSum(player_vars[i] for i in eligible_players.index if eligible_players.loc[i, 'position'] == 'MID') == 5, "Midfielders"
    prob += lpSum(player_vars[i] for i in eligible_players.index if eligible_players.loc[i, 'position'] == 'FWD') == 3, "Forwards"

    for team_name in eligible_players['team_name'].unique():
        prob += lpSum(player_vars[i] for i in eligible_players.index if eligible_players.loc[i, 'team_name'] == team_name) <= 3, f"Max 3 Players from {team_name}"

    prob.solve(PULP_CBC_CMD(msg=0))

    if LpStatus[prob.status] == 'Optimal':
        selected_players_indices = [i for i in eligible_players.index if player_vars[i].varValue == 1]
        selected_team = eligible_players.loc[selected_players_indices].copy()
        
        total_cost = selected_team['cost'].sum()
        total_predicted_points = selected_team['predicted_points'].sum()

        return selected_team, total_cost, total_predicted_points, LpStatus[prob.status]
    else:
        return pd.DataFrame(), 0, 0, LpStatus[prob.status]

# --- Chip Strategy (Heuristic Logic) ---
def predict_team_gw_points(team_df, fpl_fixtures_raw, current_gameweek):
    """
    Estimates team points for a given gameweek based on predicted points.
    Considers if a player's team has a fixture in the given GW.
    """
    total_points = 0
    
    # Filter fixtures for the current gameweek
    current_gw_fixtures_list = [f for f in fpl_fixtures_raw if f.get('event') == current_gameweek]

    teams_playing_in_current_gw_ids = set()
    dgw_teams_in_current_gw_ids = set()

    for fixture in current_gw_fixtures_list:
        if not fixture.get('finished', True): # Only consider unfinished games
            teams_playing_in_current_gw_ids.add(fixture['team_h'])
            teams_playing_in_current_gw_ids.add(fixture['team_a'])
            if fixture.get('double_gameweek', False):
                dgw_teams_in_current_gw_ids.add(fixture['team_h'])
                dgw_teams_in_current_gw_ids.add(fixture['team_a'])
    
    team_df_sorted = team_df.sort_values(by='predicted_points', ascending=False)
    
    for idx, player in team_df_sorted.iterrows():
        player_team_id = player['team_id']
        
        player_has_fixture = player_team_id in teams_playing_in_current_gw_ids
        
        estimated_gw_points = 0
        if player_has_fixture:
            estimated_gw_points = player['predicted_points'] * random.uniform(0.7, 1.3)

            if player_team_id in dgw_teams_in_current_gw_ids:
                estimated_gw_points *= 2 
        
        total_points += estimated_gw_points
            
    return round(total_points, 1)


def recommend_chip_strategy(current_team_df, current_gameweek, fpl_fixtures_raw, team_id_to_name):
    """
    Provides heuristic recommendations for FPL chip usage.
    Uses actual fixture data to identify DGW/BGWs.
    """
    recommendations = []
    
    next_gameweek = current_gameweek + 1
    if next_gameweek > 38:
        recommendations.append("Season is likely over. No more chip recommendations.")
        return recommendations

    # Filter fixtures for the next gameweek and extract participating team IDs
    next_gw_fixtures_list = [f for f in fpl_fixtures_raw if f.get('event') == next_gameweek]

    teams_with_fixture_next_gw_ids = set()
    dgw_teams_next_gw_ids = set()
    for fixture in next_gw_fixtures_list:
        if not fixture.get('finished', True): # Only consider unfinished games
            teams_with_fixture_next_gw_ids.add(fixture['team_h']) # CORRECT: Uses team_h
            teams_with_fixture_next_gw_ids.add(fixture['team_a']) # CORRECT: Uses team_a
            if fixture.get('double_gameweek', False):
                dgw_teams_next_gw_ids.add(fixture['team_h'])
                dgw_teams_next_gw_ids.add(fixture['team_a'])

    current_team_player_team_ids = set(current_team_df['team_id'].unique())
    
    is_next_bgw = len(current_team_player_team_ids.difference(teams_with_fixture_next_gw_ids)) > 0
    
    is_next_dgw = bool(dgw_teams_next_gw_ids)

    # --- Wildcard ---
    avg_predicted_points_current_team = current_team_df['predicted_points'].mean()
    if avg_predicted_points_current_team < 4.5:
        recommendations.append("🚨 **Wildcard:** Your current squad's average predicted performance is low. Consider playing your Wildcard to rebuild for future gameweeks!")

    # --- Free Hit ---
    players_without_fixture_count = len(current_team_df[~current_team_df['team_id'].isin(teams_with_fixture_next_gw_ids)])

    if is_next_bgw and players_without_fixture_count >= 5:
        recommendations.append(f"🔥 **Free Hit:** Gameweek {next_gameweek} is a Blank Gameweek for many of your players ({players_without_fixture_count} without fixture). A Free Hit could maximize your points this GW without permanent changes.")
    elif is_next_dgw:
        players_from_dgw_teams = current_team_df[current_team_df['team_id'].isin(dgw_teams_next_gw_ids)].shape[0]
        if players_from_dgw_teams < 5:
             recommendations.append(f"🔥 **Free Hit:** Gameweek {next_gameweek} is a Double Gameweek. Your current team might not have enough DGW players. A Free Hit could allow you to load up on key DGW assets temporarily.")

    # --- Triple Captain ---
    if is_next_dgw:
        potential_tc_players = current_team_df[current_team_df['team_id'].isin(dgw_teams_next_gw_ids)]
        
        if not potential_tc_players.empty:
            best_tc_candidate = potential_tc_players.sort_values(by='predicted_points', ascending=False).iloc[0]
            recommendations.append(f"⭐ **Triple Captain:** Consider Triple Captaining **{best_tc_candidate['name']} ({best_tc_candidate['team_name']})** in Gameweek {next_gameweek}. They are predicted to score high and have a Double Gameweek!")

    # --- Bench Boost ---
    if is_next_dgw:
        all_players_have_fixture = current_team_df['team_id'].isin(teams_with_fixture_next_gw_ids).all()
        if all_players_have_fixture and current_team_df['predicted_points'].min() > 3.0:
             recommendations.append(f"📈 **Bench Boost:** Gameweek {next_gameweek} is a Double Gameweek and all your 15 players look strong with fixtures. Consider playing your Bench Boost to maximize points!")
    
    if not recommendations:
        recommendations.append("No specific chip recommendations for now. Your team looks good! Keep an eye on player news.")

    return recommendations

# --- Streamlit UI Layout ---
st.title("⚽ FPL AI Predictor & Team Optimizer (Live Demo) ⚽")
st.markdown("""
Welcome to your personal Fantasy Premier League AI assistant! 
This application fetches **live FPL data** to help you select the best 15 players for your squad based on predicted performance
and FPL rules, and provides insights on when to use your valuable chips.

**Disclaimer:** Player predictions are based on a simplified model for demonstration purposes. 
A real production application would require robust, trained machine learning models, regular data updates, and more complex logic.
""")

col1, col2 = st.columns([1, 2])

# --- Sidebar for Data Loading & Settings ---
st.sidebar.header("🛠️ Data & Model")
st.sidebar.info("Data is cached for 1 hour to reduce API calls.")
players_df_full, team_id_to_name_map = fetch_fpl_data()
fpl_fixtures_raw = fetch_fpl_fixtures()
current_fpl_gameweek = fetch_current_gameweek()

if players_df_full.empty:
    st.stop()

# Load the prediction model (or simulate it)
prediction_model = load_prediction_model()
players_df_full['predicted_points'] = prediction_model.predict(players_df_full.copy())

with col1:
    st.header("⚙️ Settings")
    
    st.subheader("Team Building Parameters")
    budget = st.slider("Budget (£M)", min_value=90.0, max_value=105.0, value=100.0, step=0.5)
    
    generate_team_button = st.button("Generate Optimal Team", key="generate_team_btn")
    st.markdown("---")
    
    st.subheader("Chip Strategy Suggestions")
    st.write(f"Current FPL Gameweek: **GW {current_fpl_gameweek}**")
    gameweek_for_advice = st.slider("Gameweek for Chip Advice", min_value=current_fpl_gameweek, max_value=38, value=current_fpl_gameweek)
    
    show_chip_advice_button = st.button("Get Chip Advice", key="get_chip_advice_btn")

with col2:
    st.header("✨ Optimal Team Recommendation")

    selected_team_df = pd.DataFrame() # Initialize empty dataframe
    total_cost, total_predicted_points, status = 0, 0, ""

    if generate_team_button:
        selected_team_df, total_cost, total_predicted_points, status = optimize_team(players_df_full.copy(), budget)
        
        if status == 'Optimal':
            st.success(f"Optimal Team Found! Status: {status}")
            st.metric("Total Predicted Points (Approx.)", f"{total_predicted_points:.1f}")
            st.metric("Total Cost", f"£{total_cost:.1f}M")
            
            st.subheader("Your Recommended 15-Player Squad")
            st.dataframe(selected_team_df[[
                'name', 'team_name', 'position', 'cost', 'predicted_points', 'form', 'ppg', 'status'
            ]].sort_values(by=['position', 'predicted_points'], ascending=[True, False])
            .reset_index(drop=True)
            .style.format({'cost': "£{:.1f}M", 'predicted_points': "{:.1f}", 'ppg': "{:.1f}", 'form': "{:.1f}"}),
            use_container_width=True)
            
            st.info("Remember to manually select your starting XI and captain in the FPL app!")
            
            st.session_state['current_optimal_team'] = selected_team_df

        else:
            st.error(f"Could not find an optimal team. Solver Status: {status}. Try adjusting the budget or check for data issues.")
            st.info("This can happen if constraints are too strict (e.g., not enough eligible players, or budget too low/high).")

    st.markdown("---")
    st.header("💡 Chip Strategy Advice")
    if show_chip_advice_button:
        if 'current_optimal_team' not in st.session_state or st.session_state['current_optimal_team'].empty:
            st.warning("Please generate an optimal team first to get personalized chip advice based on a specific squad.")
        else:
            current_squad_for_advice = st.session_state['current_optimal_team']
            chip_advice = recommend_chip_strategy(current_squad_for_advice, gameweek_for_advice, fpl_fixtures_raw, team_id_to_name_map)
            for advice in chip_advice:
                st.markdown(f"- {advice}")
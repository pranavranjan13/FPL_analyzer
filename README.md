# ⚽ FPL AI Predictor & Team Optimizer (Live Demo) ⚽


## Table of Contents
1.  [About the Application](#about-the-application)
2.  [Features](#features)
3.  [How It Works (Technical Overview)](#how-it-works-technical-overview)
    *   [Data Acquisition](#data-acquisition)
    *   [Player Prediction (AI)](#player-prediction-ai)
    *   [Team Optimization (AI)](#team-optimization-ai)
    *   [Chip Strategy](#chip-strategy)
4.  [Setup and Installation](#setup-and-installation)
    *   [Prerequisites](#prerequisites)
    *   [Local Setup](#local-setup)
    *   [Running the Application](#running-the-application)
5.  [Deployment](#deployment)
    *   [Streamlit Community Cloud (Recommended)](#streamlit-community-cloud-recommended)
    *   [Docker (Advanced)](#docker-advanced)
6.  [Important Notes & Limitations](#important-notes--limitations)
7.  [Future Improvements](#future-improvements)
8.  [Acknowledgments](#acknowledgments)

---

## 1. About the Application

The **FPL AI Predictor & Team Optimizer** is a Streamlit application designed to assist Fantasy Premier League (FPL) managers in making data-driven decisions. It fetches live FPL data to:

*   Predict future FPL points for players based on their current form and statistics.
*   Generate an optimal 15-player squad within a specified budget, adhering to all FPL rules (positional requirements, max players per club).
*   Provide heuristic-based recommendations on when to use FPL chips (Wildcard, Free Hit, Triple Captain, Bench Boost) by analyzing upcoming fixtures (Double Gameweeks, Blank Gameweeks).

**Disclaimer:** This application is a live demonstration and uses a **simplified AI model for player point prediction**. While the team optimization is robust, real FPL prediction is highly complex and would require extensive historical data, advanced machine learning models (e.g., XGBoost, Neural Networks), and continuous retraining.

## 2. Features

*   **Live FPL Data Integration:** Fetches real-time player, team, and fixture data directly from the official FPL API.
*   **Player Point Prediction:** Uses a simple, yet dynamic, model to estimate player points for the upcoming gameweek based on live form, cost, and points-per-game.
*   **Optimal Team Selection:** Leverages Integer Linear Programming (ILP) with `PuLP` to build the best 15-player squad (2 GKP, 5 DEF, 5 MID, 3 FWD) respecting the £100M budget and maximum 3 players per club rule.
*   **Intelligent Chip Advice:** Provides strategic recommendations for Wildcard, Free Hit, Triple Captain, and Bench Boost chips, considering upcoming Double Gameweeks (DGWs) and Blank Gameweeks (BGWs).
*   **Interactive UI:** Built with Streamlit for an intuitive and user-friendly experience, making it easy to input parameters and view recommendations.

## 3. How It Works (Technical Overview)

### Data Acquisition
The application uses the `requests` library to fetch JSON data directly from the unofficial FPL API endpoints:
*   `bootstrap-static/`: For general player, team, and position data.
*   `fixtures/`: For detailed fixture information across all gameweeks.
*   `event-status/`: To determine the current gameweek.
Data is cached using `st.cache_data` to minimize API calls and improve performance.

### Player Prediction (AI)
For this live demo, player point prediction is handled by a `SimpleFPLPredictor` class. This class simulates a pre-trained ML model and calculates predicted points based on:
*   Player's `cost`
*   Player's `form`
*   Player's `points_per_game` (PPG)
*   A small amount of random noise for realism.

**In a production-ready application, this component would be replaced by a sophisticated Machine Learning model (e.g., XGBoost Regressor, LightGBM, or even a time-series model like LSTM) trained on:**
*   Extensive historical FPL data (multiple seasons).
*   Advanced underlying statistics (Expected Goals (xG), Expected Assists (xA), key passes, touches in the box, etc., often sourced from sites like Understat or FBref).
*   Opponent difficulty ratings, home/away advantages, and player availability flags.
*   The trained model would be saved (e.g., as a `.pkl` file) and loaded at runtime.

### Team Optimization (AI)
The core of the team selection logic is an **Integer Linear Programming (ILP)** problem solved using the `PuLP` library.
The solver aims to **maximize the total predicted points** of the 15 selected players, subject to the following FPL constraints:
*   **Squad Size:** Exactly 15 players.
*   **Budget:** Total cost must be less than or equal to £100.0M (adjustable via slider).
*   **Positional Split:** 2 Goalkeepers, 5 Defenders, 5 Midfielders, 3 Forwards.
*   **Club Limit:** Maximum of 3 players from any single Premier League club.
*   **Availability:** Only players currently marked as available or doubtful in the FPL API are considered.

### Chip Strategy
The chip recommendation module employs **heuristic (rule-based)** logic:
*   It analyzes upcoming fixtures (specifically Double and Blank Gameweeks) using the fetched fixture data.
*   **Wildcard:** Suggested if the current team's average predicted points are low, indicating a need for a squad overhaul.
*   **Free Hit:** Recommended for significant Blank Gameweeks (where many players have no fixture) or specific Double Gameweeks where a temporary squad change is highly beneficial.
*   **Triple Captain:** Suggests the highest-predicted scoring player from your current squad who also has a Double Gameweek.
*   **Bench Boost:** Recommended during Double Gameweeks if all 15 players in the selected squad are predicted to have fixtures and decent point returns.

## 4. Setup and Installation

### Prerequisites
*   Python 3.8+
*   Git (for cloning the repository)

### Local Setup
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/fpl_ai_predictor.git
    cd fpl_ai_predictor
    ```
    (Replace `https://github.com/your-username/fpl_ai_predictor.git` with your actual repository URL)

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    ```

3.  **Activate the virtual environment:**
    *   On Windows:
        ```bash
        .\.venv\Scripts\activate
        ```
    *   On macOS/Linux:
        ```bash
        source .venv/bin/activate
        ```

4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application
Once the dependencies are installed and your virtual environment is active:
```bash
streamlit run app.py

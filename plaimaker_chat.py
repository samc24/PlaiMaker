"""
PlaiMaker Chat: Streamlit app for querying basketball stats using OpenAI function calling and a CSV dataset.
Loads data, exposes stat/ranking functions, and interacts with users via a chat interface.
"""

import streamlit as st
import openai
import json
import os
from stats_helpers import StatsHelper
from dotenv import load_dotenv

# --- Set OpenAI API Key ---
load_dotenv()
# openai_api_key = os.environ.get("OPENAI_API_KEY")
openai.api_key = st.secrets["openai"]["api_key"]

# if not openai_api_key:
#     st.error("OPENAI_API_KEY environment variable not set. Please set it in your environment.")
#     st.stop()
# openai.api_key = openai_api_key

# --- Load Data ---
stats_data = StatsHelper('stats_all.csv')
available_stats = stats_data.get_available_stats()
available_stats_text = ", ".join(available_stats)

# --- Column Descriptions Mapping ---
column_descriptions = {
    "Recruiting Profile": "Recruiting Profile (player's recruiting info)",
    "Player": "Player Name",
    "Games": "Games Played",
    "Time": "Minutes Played (hh:mm:ss)",
    "Events": "Total Events (custom stat)",
    "EFG%": "Effective Field Goal Percentage",
    "Pts.": "Points Scored",
    "Opps.": "Scoring Opportunities",
    "Ast": "Assists",
    "TO": "Turnovers",
    "TO%": "Turnover Percentage",
    "Reb": "Total Rebounds",
    "OReb": "Offensive Rebounds",
    "DReb": "Defensive Rebounds",
    "Reb%": "Rebound Percentage",
    "Fouls": "Personal Fouls",
    "Team Pts.": "Team Points Scored",
    "Team Pts. Allowed": "Team Points Allowed",
    "#ERROR!": "Data Error (ignore)",
    "Off. PPP": "Offensive Points Per Possession",
    "Def. PPP": "Defensive Points Per Possession",
    "Net PPP": "Net Points Per Possession (Off - Def)",
    "VPS": "Value Point Score (custom stat)",
    "2 Pt Fouls": "Fouls Drawn on 2-Point Attempts",
    "2Pt": "2-Point Field Goals Made",
    "2Pt A": "2-Point Field Goals Attempted",
    "2Pt Fouled": "Fouled on 2-Point Attempt",
    "2Pt miss": "2-Point Field Goals Missed",
    "2Pt Rate": "2-Point Attempt Rate",
    "2Pt%": "2-Point Field Goal Percentage",
    "3 Pt Fouls": "Fouls Drawn on 3-Point Attempts",
    "3Pt": "3-Point Field Goals Made",
    "3Pt A": "3-Point Field Goals Attempted",
    "3Pt Fouled": "Fouled on 3-Point Attempt",
    "3Pt miss": "3-Point Field Goals Missed",
    "3Pt Rate": "3-Point Attempt Rate",
    "3Pt%": "3-Point Field Goal Percentage",
    "And-1 Fouls": "And-1 Fouls (made basket + foul)",
    "Ast%": "Assist Percentage",
    "ATR %": "Assist-to-Turnover Ratio (Percentage)",
    "ATR Att.": "Assist-to-Turnover Ratio Attempts",
    "ATR Fouled": "Fouled on ATR Play (custom stat)",
    "ATR Made": "ATR Made (custom stat)",
    "ATR Rate": "ATR Rate (custom stat)",
    "ATR/A": "Assist-to-Turnover Ratio (per Attempt)",
    "Blk": "Blocks",
    "Blk%": "Block Percentage",
    "Blk/Foul": "Blocks per Foul",
    "Blkd": "Times Blocked",
    "Chrg": "Charges Taken",
    "Cut": "Cuts (custom stat)",
    "Def Foul": "Defensive Fouls",
    "Def. 2Pt%": "Opponent 2-Point FG% (Defense)",
    "Def. 3Pt%": "Opponent 3-Point FG% (Defense)",
    "Def. Poss.": "Defensive Possessions",
    "Def. Rtg": "Defensive Rating",
    "Def. TO": "Defensive Turnovers Forced",
    "Def. TO%": "Defensive Turnover Percentage",
    "Deflect": "Deflections",
    "FT": "Free Throws Made",
    "FT A": "Free Throws Attempted",
    "FT miss": "Free Throws Missed",
    "FT Trips": "Free Throw Trips (custom stat)",
    "FT Violation": "Free Throw Violations",
    "FT%": "Free Throw Percentage",
    "FTF": "Free Throws per Foul (custom stat)",
    "LB Foul": "Loose Ball Fouls",
    "Long Mid %": "Long Midrange Field Goal Percentage",
    "Long Mid Att.": "Long Midrange Field Goals Attempted",
    "Long Mid Fouled": "Fouled on Long Midrange Shot",
    "Long Mid Made": "Long Midrange Field Goals Made",
    "Long Mid Rate": "Long Midrange Attempt Rate",
    "Long Mid/A": "Long Midrange Attempts per Game",
    "Lost Tie Up": "Lost Tie-Ups (jump balls lost)",
    "MPG": "Minutes Per Game",
    "NS Fouls": "Non-Shooting Fouls",
    "NS Fouls (Bonus)": "Non-Shooting Fouls (Bonus)",
    "Off Foul": "Offensive Fouls",
    "Off. Pace": "Offensive Pace (Possessions per 40 min)",
    "Off. Poss.": "Offensive Possessions",
    "Off. Rtg": "Offensive Rating",
    "Opp DRebs": "Opponent Defensive Rebounds",
    "Opp ORebs": "Opponent Offensive Rebounds",
    "Pace": "Game Pace (Possessions per 40 min)",
    "Poss.": "Total Possessions",
    "PPG": "Points Per Game",
    "Short Mid %": "Short Midrange Field Goal Percentage",
    "Short Mid Att.": "Short Midrange Field Goals Attempted",
    "Short Mid Fouled": "Fouled on Short Midrange Shot",
    "Short Mid Made": "Short Midrange Field Goals Made",
    "Short Mid Rate": "Short Midrange Attempt Rate",
    "Short Mid/A": "Short Midrange Attempts per Game",
    "Shots": "Total Shots Attempted",
    "Stl": "Steals",
    "Stl%": "Steal Percentage",
    "Stl/Foul": "Steals per Foul",
    "Tech Foul": "Technical Fouls",
    "Tie Up": "Tie-Ups (Jump Balls)",
    "TS%": "True Shooting Percentage",
    "Usage %": "Usage Percentage (team possessions used)",
    # Add any additional columns from the CSV here, using best-guess or mark as 'Unknown/Custom Stat' if unclear
}

# --- GPT Function Schema ---
functions = [
    {
        "name": "get_stat_from_csv",
        "description": "Retrieve a specific stat for a player from the CSV data.",
        "parameters": {
            "type": "object",
            "properties": {
                "player_name": {"type": "string", "description": "Player's full name"},
                "stat_type": {"type": "string", "description": "Stat column from the dataset, like 'Pts.' or 'TS%'"},
                "event_context": {"type": "string", "description": "Optional context like 'after substitution'."}
            },
            "required": ["player_name", "stat_type"]
        }
    },
    {
        "name": "rank_players_by_stat",
        "description": "Rank top players by a stat from the CSV data.",
        "parameters": {
            "type": "object",
            "properties": {
                "stat_type": {"type": "string", "description": "Stat column to rank by, like 'Pts.'"},
                "top_n": {"type": "integer", "description": "Number of top players to return, default 5"}
            },
            "required": ["stat_type"]
        }
    },
    {
        "name": "get_all_stats_for_player",
        "description": "Retrieve all available stats for a player from the CSV data.",
        "parameters": {
            "type": "object",
            "properties": {
                "player_name": {"type": "string", "description": "Player's full name"}
            },
            "required": ["player_name"]
        }
    }
]

def summarize_stat(stats_data):
    """
    Format a stat lookup result for display.
    Args:
        stats_data (dict): The result from get_stat_from_csv.
    Returns:
        str: Human-readable summary or error message.
    """
    if "error" in stats_data:
        return stats_data["error"]
    player = stats_data["player"]
    stat = stats_data["stat"]
    value = stats_data["value"]
    context = stats_data["context"]
    context_str = "" if context == "N/A" else f" during {context}"
    return f"{player} recorded {value} {stat}{context_str} according to Hoopalytics data."

def summarize_ranking(ranking_data):
    """
    Format a ranking result for display.
    Args:
        ranking_data (dict): The result from rank_players_by_stat.
    Returns:
        str: Human-readable ranking summary or error message.
    """
    if "error" in ranking_data:
        return ranking_data["error"]
    stat = ranking_data["stat"]
    players = ranking_data["ranking"]
    if not players:
        return "No data available for ranking."
    summary = f"Top {ranking_data.get('top_n', len(players))} players ranked by {stat}:\n"
    for idx, entry in enumerate(players, 1):
        summary += f"{idx}. {entry['Player']} - {entry[stat]}\n"
    return summary.strip()

def summarize_all_stats(all_stats_data):
    """
    Format all stats for a player for display.
    Args:
        all_stats_data (dict): The result from get_all_stats_for_player.
    Returns:
        str: Human-readable summary or error message.
    """
    if "error" in all_stats_data:
        return all_stats_data["error"]
    player = all_stats_data["player"]
    stats = all_stats_data["stats"]
    summary = f"All stats for {player}:\n"
    for k, v in stats.items():
        summary += f"- {k}: {v}\n"
    return summary.strip()

# --- Streamlit App ---
st.title("PlaiMaker Chat 🏀 \nPowered by Hoopsalytics")
st.markdown("""
Ask specific or ranking basketball questions from your dataset.
**Examples:**  
- How many points did AJ Dybansta score?
- Rank the top scorers.
""")

# --- Build column description string for prompt ---
column_desc_str = "\n".join([
    f"- {col}: {desc}" for col, desc in column_descriptions.items()
])

user_query = st.text_input("Your question:")

if user_query:
    with st.spinner("Thinking..."):
        # Call OpenAI to extract function call arguments from the user query
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You help users retrieve basketball stats from a CSV file. "
                        "The available stat columns are listed below with their descriptions. "
                        "When a user asks for a stat, infer the best-fit column from the list, even if the user uses synonyms or natural language. "
                        "Always return the column name that best matches the user's intent.\n" +
                        column_desc_str
                    )
                },
                {"role": "user", "content": user_query}
            ],
            functions=functions,
            function_call="auto"
        )
        message = response.choices[0].message
        if message.function_call:
            function_name = message.function_call.name
            function_args = json.loads(message.function_call.arguments)
            print("FUNCTION ARGS", function_args)
            if function_name == "get_stat_from_csv":
                stats_result = stats_data.get_stat_from_csv(**function_args)
                messages = [
                    {"role": "system", "content": "You answer basketball stat questions strictly based on provided CSV results."},
                    {"role": "user", "content": user_query},
                    {"role": "function", "name": "get_stat_from_csv", "content": json.dumps(stats_result)}
                ]
                final_response = openai.chat.completions.create(
                    model="gpt-4o",
                    messages=messages
                )
                answer = final_response.choices[0].message.content
            elif function_name == "rank_players_by_stat":
                ranking_result = stats_data.rank_players_by_stat(**function_args)
                messages = [
                    {"role": "system", "content": "You answer basketball stat ranking questions based on CSV results."},
                    {"role": "user", "content": user_query},
                    {"role": "function", "name": "rank_players_by_stat", "content": json.dumps(ranking_result)}
                ]
                final_response = openai.chat.completions.create(
                    model="gpt-4o",
                    messages=messages
                )
                answer = final_response.choices[0].message.content
            elif function_name == "get_all_stats_for_player":
                all_stats_result = stats_data.get_all_stats_for_player(**function_args)
                answer = summarize_all_stats(all_stats_result)
        else:
            answer = "I'm sorry, I can't answer that question yet with the current CSV data."
    st.write("### 📈 Answer:")
    st.write(answer)

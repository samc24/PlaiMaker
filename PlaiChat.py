import streamlit as st
import openai
import json
import os
from stats_data import StatsData
from dotenv import load_dotenv

# --- Set OpenAI API Key ---
load_dotenv()
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OPENAI_API_KEY environment variable not set. Please set it in your environment.")
    st.stop()
openai.api_key = openai_api_key

# --- Load Data ---
stats_data = StatsData('stats.csv')
available_stats = stats_data.get_available_stats()
available_stats_text = ", ".join(available_stats)

# --- GPT Function Schema ---
functions = [
    {
        "name": "get_stat_from_csv",
        "description": "Retrieve a specific stat for a player from the CSV data.",
        "parameters": {
            "type": "object",
            "properties": {
                "player_name": {"type": "string", "description": "Player's full name"},
                "stat_type": {"type": "string", "description": "Use exactly one of these columns: " + available_stats_text},
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
    }
]

# --- Answer Formatting ---
def summarize_stat(stats_data):
    if "error" in stats_data:
        return stats_data["error"]
    player = stats_data["player"]
    stat = stats_data["stat"]
    value = stats_data["value"]
    context = stats_data["context"]
    context_str = "" if context == "N/A" else f" during {context}"
    return f"{player} recorded {value} {stat}{context_str} according to Hoopalytics data."

def summarize_ranking(ranking_data):
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

# --- Streamlit App ---
st.title("PlaiChat 🏀 (With OpenAI Function Calling)")
st.markdown("""
Ask specific or ranking basketball questions from your dataset.
**Examples:**
- How many points did AJ Dybansta score?
- Rank the top scorers.
""")

user_query = st.text_input("Your question:")

if user_query:
    with st.spinner("Thinking..."):
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You help users retrieve basketball stats from CSV. "
                        "The available stat columns you must choose from exactly are: " + available_stats_text + ". "
                        "Only use these names for 'stat_type'. Do not guess other variants like 'Assists' if 'Ast' is present. "
                        "Examples of valid stat types include: " + available_stats_text + "."
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
        else:
            answer = "I'm sorry, I can't answer that question yet with the current CSV data."
    st.write("### 📊 Answer:")
    st.write(answer)

import streamlit as st
import pandas as pd
import re
import openai
import json


# --- Set OpenAI API Key ---
openai.api_key = "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"


# --- Load and Clean CSV ---
df = pd.read_csv('stats.csv')

def clean_player_name(name):
    cleaned = re.sub(r'#\d+\s*', '', str(name))
    cleaned = cleaned.split('\n')[0].strip()
    return cleaned


df['Player'] = df['Player'].apply(clean_player_name)


available_stats = [col for col in df.columns if col not in ['Player', 'Recruiting\nProfile', 'Unnamed: 1']]
available_stats_text = ", ".join(available_stats)
print("available_stats_text", available_stats_text)

# --- Functions Exposed to GPT ---
def get_stat_from_csv(player_name: str, stat_type: str, event_context: str = None) -> dict:
    filtered = df[df['Player'].str.lower() == player_name.lower()]

    if filtered.empty:
        return {"error": f"Player '{player_name}' not found."}

    row = filtered.iloc[0]
    print("ROW", row)
    print("stat_type", stat_type)
     # TODO ensure stat_type is normalized by here 
    if stat_type not in df.columns:
        return {"error": f"Stat '{stat_type}' not found in dataset columns."}

    stat_value = row.get(stat_type)

    if pd.isna(stat_value):
        return {"error": f"No data available for {stat_type} for {player_name}."}

    stat_value = str(stat_value)

    return {
        "player": player_name,
        "stat": stat_type,
        "value": stat_value,
        "context": event_context or "N/A"
    }



def rank_players_by_stat(stat_type: str, top_n: int = 5) -> dict:
    if stat_type not in df.columns:
        return {"error": f"Stat '{stat_type}' not found in dataset columns."}

    sorted_df = df.sort_values(by=stat_type, ascending=False)
    sorted_df = sorted_df[['Player', stat_type]].head(top_n)

    ranking = sorted_df.to_dict(orient='records')

    return {
        "stat": stat_type,
        "top_n": top_n,
        "ranking": ranking
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

        # Call GPT to extract function call arguments
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
                stats_data = get_stat_from_csv(**function_args)
                print("json.dumps(stats_data)", json.dumps(stats_data))

                messages = [
                    {"role": "system", "content": "You answer basketball stat questions strictly based on provided CSV results."},
                    {"role": "user", "content": user_query},
                    {"role": "function", "name": "get_stat_from_csv", "content": json.dumps(stats_data)}
                ]

                final_response = openai.chat.completions.create(
                    model="gpt-4o",
                    messages=messages
                )
                answer = final_response.choices[0].message.content

            elif function_name == "rank_players_by_stat":
                ranking_data = rank_players_by_stat(**function_args)

                messages = [
                    {"role": "system", "content": "You answer basketball stat ranking questions based on CSV results."},
                    {"role": "user", "content": user_query},
                    {"role": "function", "name": "rank_players_by_stat", "content": json.dumps(ranking_data)}
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

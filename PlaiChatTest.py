import streamlit as st
import pandas as pd
import re


# --- Load and Clean CSV ---
df = pd.read_csv('stats.csv')

def clean_player_name(name):
    cleaned = re.sub(r'#\d+\s*', '', str(name))
    cleaned = cleaned.split('\n')[0].strip()
    return cleaned

df['Player'] = df['Player'].apply(clean_player_name)


# --- Lookup Function ---
def get_stat_from_csv(player_name: str, stat_type: str, event_context: str = None) -> dict:
    filtered = df[df['Player'].str.lower() == player_name.lower()]

    if filtered.empty:
        return {"error": f"Player '{player_name}' not found."}

    row = filtered.iloc[0]
    if stat_type not in df.columns:
        return {"error": f"Stat '{stat_type}' not found in dataset columns."}

    stat_value = row.get(stat_type)

    return {
        "player": player_name,
        "stat": stat_type,
        "value": stat_value,
        "context": event_context or "N/A"
    }


# --- Mock GPT Routing for Prototype Question ---
def mock_gpt_router(user_question: str):
    # Hardcode mapping for your prototype question
    if "aj dybansta" in user_question.lower() and "points" in user_question.lower():
        return {
            "player_name": "AJ Dybansta",
            "stat_type": "Pts.",
            "event_context": None
        }
    else:
        # fallback example if you want more
        return {
            "player_name": "Donovan Williams, Jr.",
            "stat_type": "EFG%",
            "event_context": None
        }


# --- Mock GPT Answer Formatting ---
def generate_mock_answer(stats_data):
    if "error" in stats_data:
        return stats_data["error"]

    player = stats_data["player"]
    stat = stats_data["stat"]
    value = stats_data["value"]
    context = stats_data["context"]

    context_str = "" if context == "N/A" else f" during {context}"

    return f"{player} recorded {value} {stat}{context_str} according to Hoopalytics data."


# --- Streamlit App ---
st.title("PlaiChat (Mocked) 🏀")

st.markdown(
    "Ask a basketball performance question. Prototype example: **How many points did AJ Dybansta score?**"
)

user_query = st.text_input("Your question:")

if user_query:
    with st.spinner("Thinking..."):
        function_args = mock_gpt_router(user_query)
        stats_data = get_stat_from_csv(**function_args)
        answer = generate_mock_answer(stats_data)

    st.write("### 📊 Answer:")
    st.write(answer)

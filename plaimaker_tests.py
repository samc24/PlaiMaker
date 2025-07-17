import pandas as pd
import re
import pytest

# --- Functions to Test ---
def clean_player_name(name):
    cleaned = re.sub(r'#\d+\s*', '', str(name))
    cleaned = cleaned.split('\n')[0].strip()
    return cleaned

def get_stat_from_csv(df, player_name: str, stat_type: str, event_context: str = None) -> dict:
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

def mock_gpt_router(user_question: str):
    if "aj dybansta" in user_question.lower() and "points" in user_question.lower():
        return {
            "player_name": "AJ Dybansta",
            "stat_type": "Pts.",
            "event_context": None
        }
    else:
        return {
            "player_name": "Donovan Williams, Jr.",
            "stat_type": "EFG%",
            "event_context": None
        }

# --- Fixtures ---
@pytest.fixture
def sample_df():
    data = {
        'Player': [clean_player_name('AJ Dybansta'), clean_player_name('Donovan Williams, Jr.')],
        'Pts.': [25, 18],
        'EFG%': [0.55, 0.62]
    }
    return pd.DataFrame(data)

# --- Tests ---
def test_clean_player_name():
    assert clean_player_name('AJ Dybansta #123') == 'AJ Dybansta'
    assert clean_player_name('Donovan Williams, Jr.\n') == 'Donovan Williams, Jr.'
    assert clean_player_name('John Doe') == 'John Doe'

def test_get_stat_from_csv_success(sample_df):
    result = get_stat_from_csv(sample_df, 'AJ Dybansta', 'Pts.')
    assert result['player'] == 'AJ Dybansta'
    assert result['stat'] == 'Pts.'
    assert result['value'] == 25
    assert result['context'] == 'N/A'

def test_get_stat_from_csv_player_not_found(sample_df):
    result = get_stat_from_csv(sample_df, 'Nonexistent Player', 'Pts.')
    assert 'error' in result
    assert "not found" in result['error']

def test_get_stat_from_csv_stat_not_found(sample_df):
    result = get_stat_from_csv(sample_df, 'AJ Dybansta', 'Ast')
    assert 'error' in result
    assert "not found" in result['error']

def test_mock_gpt_router_points():
    args = mock_gpt_router('How many points did AJ Dybansta score?')
    assert args['player_name'] == 'AJ Dybansta'
    assert args['stat_type'] == 'Pts.'

def test_mock_gpt_router_default():
    args = mock_gpt_router('Show me EFG for Donovan Williams, Jr.')
    assert args['player_name'] == 'Donovan Williams, Jr.'
    assert args['stat_type'] == 'EFG%'

import pandas as pd
import re
from typing import Optional, Dict, Any, List
import difflib

class StatsData:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self.clean_data()
        self.available_stats = [col for col in self.df.columns if col not in ['Player', 'Recruiting\nProfile', 'Unnamed: 1']]

    def clean_player_name(self, name: str) -> str:
        cleaned = re.sub(r'#\d+\s*', '', str(name))
        cleaned = cleaned.split('\n')[0].strip()
        return cleaned

    def clean_data(self):
        self.df['Player'] = self.df['Player'].apply(self.clean_player_name)

    def normalize_stat_type(self, stat_type: str) -> Optional[str]:
        # Fuzzy match stat_type to available columns (case-insensitive, ignore punctuation)
        candidates = self.available_stats
        stat_type_clean = re.sub(r'[^a-zA-Z0-9]', '', stat_type).lower()
        for col in candidates:
            col_clean = re.sub(r'[^a-zA-Z0-9]', '', col).lower()
            if stat_type_clean == col_clean:
                return col
        # Fallback: use difflib for closest match
        matches = difflib.get_close_matches(stat_type, candidates, n=1, cutoff=0.7)
        if matches:
            return matches[0]
        return None

    def get_stat_from_csv(self, player_name: str, stat_type: str, event_context: Optional[str] = None) -> Dict[str, Any]:
        norm_stat = self.normalize_stat_type(stat_type)
        if not norm_stat:
            return {"error": f"Stat '{stat_type}' not found in dataset columns."}
        filtered = self.df[self.df['Player'].str.lower() == player_name.lower()]
        if filtered.empty:
            return {"error": f"Player '{player_name}' not found."}
        row = filtered.iloc[0]
        stat_value = row.get(norm_stat)
        if pd.isna(stat_value):
            return {"error": f"No data available for {norm_stat} for {player_name}."}
        return {
            "player": player_name,
            "stat": norm_stat,
            "value": str(stat_value),
            "context": event_context or "N/A"
        }

    def rank_players_by_stat(self, stat_type: str, top_n: int = 5) -> Dict[str, Any]:
        norm_stat = self.normalize_stat_type(stat_type)
        if not norm_stat:
            return {"error": f"Stat '{stat_type}' not found in dataset columns."}
        sorted_df = self.df.sort_values(by=norm_stat, ascending=False)
        sorted_df = sorted_df[['Player', norm_stat]].head(top_n)
        ranking = sorted_df.to_dict(orient='records')
        return {
            "stat": norm_stat,
            "top_n": top_n,
            "ranking": ranking
        }

    def get_available_stats(self) -> List[str]:
        return self.available_stats 
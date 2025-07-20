"""
StatsHelper module for working with basketball statistics data.
Loads, cleans, and queries player stats from a CSV file.
"""

import pandas as pd
import re
from typing import Optional, Dict, Any, List
import difflib

class StatsHelper:
    """
    Helper class for working with basketball player statistics from a CSV file.
    Handles data loading, cleaning, and querying.
    """
    def __init__(self, csv_path: str):
        """
        Initialize StatsHelper with the path to a CSV file.
        Loads the data and prepares available stats.
        """
        self.df = pd.read_csv(csv_path)
        self.clean_data()
        self.available_stats = [
            col for col in self.df.columns if col not in ['Player', 'Recruiting\nProfile', 'Unnamed: 1']
        ]

    def clean_player_name(self, name: str) -> str:
        """
        Clean a player name by removing jersey numbers and trailing whitespace/newlines.
        """
        cleaned = re.sub(r'#\d+\s*', '', str(name))
        cleaned = cleaned.split('\n')[0].strip()
        return cleaned

    def clean_data(self):
        """
        Clean all player names in the DataFrame.
        """
        self.df['Player'] = self.df['Player'].apply(self.clean_player_name)

    def normalize_stat_type(self, stat_type: str) -> Optional[str]:
        """
        Normalize a stat type string to the closest matching column in the DataFrame.
        Uses case-insensitive and punctuation-insensitive matching, with fuzzy fallback.
        """
        candidates = self.available_stats
        stat_type_clean = re.sub(r'[^a-zA-Z0-9]', '', stat_type).lower()
        for col in candidates:
            col_clean = re.sub(r'[^a-zA-Z0-9]', '', col).lower()
            if stat_type_clean == col_clean:
                return col
        # Fallback to fuzzy matching if no exact match
        matches = difflib.get_close_matches(stat_type, candidates, n=1, cutoff=0.7)
        if matches:
            return matches[0]
        return None

    def get_stat_from_csv(self, player_name: str, stat_type: str, event_context: Optional[str] = None) -> Dict[str, Any]:
        """
        Get a specific stat for a player from the DataFrame.
        """
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
        """
        Rank the top players by a given stat.
        """
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
        """
        Get a list of available stat columns in the DataFrame.
        """
        return self.available_stats

    def get_all_stats_for_player(self, player_name: str) -> dict:
        """
        Get all available stats for a player from the DataFrame, excluding recruiting/profile columns and NaN values.
        """
        filtered = self.df[self.df['Player'].str.lower() == player_name.lower()]
        if filtered.empty:
            return {"error": f"Player '{player_name}' not found."}
        row = filtered.iloc[0].to_dict()
        # Exclude unwanted columns and NaN values
        exclude_cols = {'Recruiting Profile', 'Unnamed: 1'}
        stats = {k: v for k, v in row.items() if k not in exclude_cols and pd.notna(v)}
        return {"player": player_name, "stats": stats} 
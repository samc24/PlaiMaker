"""
DubaiStatsHelper module for working with Dubai basketball statistics data.
Loads, cleans, and queries player stats from the Dubai CSV file.
"""

import pandas as pd
import re
from typing import Optional, Dict, Any, List
import difflib

class DubaiStatsHelper:
    """
    Helper class for working with Dubai basketball player statistics from a CSV file.
    Handles data loading, cleaning, and querying for the Dubai tournament dataset.
    """
    def __init__(self, csv_path: str):
        """
        Initialize DubaiStatsHelper with the path to a CSV file.
        Loads the data and prepares available stats.
        """
        self.df = pd.read_csv(csv_path)
        self.clean_data()
        self.available_stats = [
            col for col in self.df.columns if col not in ['player_name', 'team_name']
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
        self.df['player_name'] = self.df['player_name'].apply(self.clean_player_name)

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
        filtered = self.df[self.df['player_name'].str.lower() == player_name.lower()]
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
        sorted_df = sorted_df[['player_name', 'team_name', norm_stat]].head(top_n)
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
        Get all available stats for a player from the DataFrame, excluding team_name and NaN values.
        """
        filtered = self.df[self.df['player_name'].str.lower() == player_name.lower()]
        if filtered.empty:
            return {"error": f"Player '{player_name}' not found."}
        row = filtered.iloc[0].to_dict()
        # Exclude unwanted columns and NaN values
        exclude_cols = {'team_name'}
        stats = {k: v for k, v in row.items() if k not in exclude_cols and pd.notna(v)}
        return {"player": player_name, "stats": stats}

    def get_team_stats(self, team_name: str) -> Dict[str, Any]:
        """
        Get all players and their stats for a specific team.
        """
        filtered = self.df[self.df['team_name'].str.lower() == team_name.lower()]
        if filtered.empty:
            return {"error": f"Team '{team_name}' not found."}
        
        team_stats = []
        for index, row in filtered.iterrows():
            player_stats = {
                'player_name': row['player_name'],
                'team_name': row['team_name']
            }
            # Add all other stats
            for col in self.available_stats:
                if pd.notna(row[col]):
                    player_stats[col] = row[col]
            team_stats.append(player_stats)
        
        return {
            "team": team_name,
            "player_count": len(team_stats),
            "players": team_stats
        }

    def get_quarter_performance(self, quarter: str) -> Dict[str, Any]:
        """
        Get performance statistics for a specific quarter.
        quarter should be '1st', '2nd', '3rd', or '4th'
        """
        quarter_mapping = {
            '1st': 'avg_1st_quarter_pts',
            '2nd': 'avg_2nd_quarter_pts', 
            '3rd': 'avg_3rd_quarter_points',
            '4th': 'avg_4th_quarter_points'
        }
        
        if quarter not in quarter_mapping:
            return {"error": f"Invalid quarter '{quarter}'. Use '1st', '2nd', '3rd', or '4th'."}
        
        col = quarter_mapping[quarter]
        if col not in self.df.columns:
            return {"error": f"Quarter data not available for {quarter} quarter."}
        
        # Get top performers for this quarter
        sorted_df = self.df.sort_values(by=col, ascending=False)
        top_performers = sorted_df[['player_name', 'team_name', col]].head(10)
        
        return {
            "quarter": quarter,
            "top_performers": top_performers.to_dict(orient='records'),
            "average_points": self.df[col].mean(),
            "max_points": self.df[col].max()
        }

    def get_foul_analysis(self) -> Dict[str, Any]:
        """
        Get foul analysis across all players and quarters.
        """
        foul_columns = [
            'avg_1st_quarter_fouls', 'avg_2nd_quarter_fouls', 
            'avg_3rd_quarter_fouls', 'avg_4th_quarter_fouls', 'avg_total_fouls'
        ]
        
        foul_stats = {}
        for col in foul_columns:
            if col in self.df.columns:
                foul_stats[col] = {
                    'average': self.df[col].mean(),
                    'max': self.df[col].max(),
                    'min': self.df[col].min()
                }
        
        # Get players with most fouls
        if 'avg_total_fouls' in self.df.columns:
            most_fouls = self.df.nlargest(5, 'avg_total_fouls')[['player_name', 'team_name', 'avg_total_fouls']]
            foul_stats['most_fouls'] = most_fouls.to_dict(orient='records')
        
        # Get players with least fouls
        if 'avg_total_fouls' in self.df.columns:
            least_fouls = self.df.nsmallest(5, 'avg_total_fouls')[['player_name', 'team_name', 'avg_total_fouls']]
            foul_stats['least_fouls'] = least_fouls.to_dict(orient='records')
        
        return foul_stats 
"""
PlaiMaker Dubai - Basketball Stats Chat

A Streamlit app that lets you ask questions about Dubai basketball tournament stats in plain English.
Uses GPT-4o to convert your questions into data queries automatically.
Focused on quarter-by-quarter performance and fouls analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import openai
import json
import re
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dubai_stats_helpers import DubaiStatsHelper

# Set up OpenAI API key from Streamlit secrets
openai_api_key = st.secrets["openai"]["api_key"]
openai.api_key = openai_api_key

# Load the Dubai basketball stats data
@st.cache_data
def load_dubai_data():
    """Load the Dubai basketball statistics dataset."""
    stats_data = DubaiStatsHelper('stats_dubai.csv')
    return stats_data

# Helper functions
def clean_generated_code(code):
    """
    Clean up the pandas code that GPT generates.
    Fixes common issues like markdown formatting and per-game calculations.
    """
    if not code:
        return code
    
    # Remove markdown code blocks
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0].strip()
    elif "```" in code:
        code = code.split("```")[1].split("```")[0].strip()
    
    # Fix loop variable naming issues
    code = re.sub(r"for\s+(\w+),\s+(\w+)\s+in\s+(\w+)\.iterrows\(\):", r"for \1, \2 in \3.iterrows():", code)
    
    # Fix f-string newline issues
    code = code.replace("\\n", "\n")
    
    # Remove trailing newlines in f-strings
    code = re.sub(r'f"([^"]*)\\n"', r'f"\1"', code)
    
    return code

def format_query_result(result_text, query):
    """
    Format the query results into a nice table.
    Tries to parse player data and create a DataFrame if possible.
    """
    if not result_text or result_text.strip() == "":
        return "No results found for your query."
    
    # Clean up the result text
    result_text = result_text.strip()
    
    # Try to parse and format as a table
    try:
        # Split by lines
        lines = result_text.split("\n")
        players_data = []
        
        # Pattern 1: "Player - Points: X, Rebounds: Y" (most common)
        for line in lines:
            line = line.strip()
            if line and " - " in line and ":" in line:
                parts = line.split(" - ")
                if len(parts) == 2:
                    player = parts[0].strip()
                    stats = parts[1].strip()
                    
                    # Parse stats into a dict
                    stats_dict = {"player_name": player}
                    # Split by comma and handle each stat
                    stat_pairs = [pair.strip() for pair in stats.split(",")]
                    for pair in stat_pairs:
                        if ":" in pair:
                            stat_parts = pair.split(":", 1)
                            if len(stat_parts) == 2:
                                stat_name = stat_parts[0].strip()
                                stat_value = stat_parts[1].strip()
                                stats_dict[stat_name] = stat_value
                    
                    # Only add if we have at least one stat
                    if len(stats_dict) > 1:
                        players_data.append(stats_dict)
        
        # Pattern 2: "Player: Name" followed by stat lines
        if not players_data and "Player:" in result_text:
            lines = result_text.split("\n")
            table_data = []
            current_player = None
            current_stats = {}
            
            for line in lines:
                line = line.strip()
                if line and ":" in line:
                    if "Player:" in line:
                        # Save previous player if exists
                        if current_player and current_stats:
                            current_stats["player_name"] = current_player
                            table_data.append(current_stats)
                        
                        # Start new player
                        current_player = line.split("Player:")[1].strip()
                        current_stats = {}
                    else:
                        # This is a stat line
                        stat_parts = line.split(":", 1)
                        if len(stat_parts) == 2:
                            stat_name = stat_parts[0].strip()
                            stat_value = stat_parts[1].strip()
                            current_stats[stat_name] = stat_value
            
            # Add last player
            if current_player and current_stats:
                current_stats["player_name"] = current_player
                table_data.append(current_stats)
            
            if table_data:
                players_data = table_data
        
        # If we found structured data, create a DataFrame
        if players_data:
            # Create a DataFrame and return it for st.dataframe
            df_result = pd.DataFrame(players_data)
            return df_result
        
        # If no structured data found, return formatted text
        return result_text.replace("  ", "\n  ").replace(" - ", "\n  ")
    
    except Exception as e:
        # If formatting fails, return the original with basic cleanup
        return result_text.replace("  ", "\n  ").replace(" - ", "\n  ")

def create_nice_output(result_data, query):
    """
    Create a nicely formatted output with the query context.
    """
    # Create the output header
    output_header = f"""
## 🏀 Dubai Basketball Stats Results

**Your Question:** {query}

"""
    
    return output_header, result_data

def create_player_report(player_name, stats_data):
    """
    Create a comprehensive player report with visualizations and rankings for Dubai data.
    """
    # Find the player in the dataset
    player_data = stats_data.df[stats_data.df['player_name'].str.lower() == player_name.lower()]
    
    if player_data.empty:
        return None, None, None
    
    player_row = player_data.iloc[0]
    
    # Get key stats for the player (Dubai-specific columns)
    key_stats = {
        'Team': player_row.get('team_name', 'N/A'),
        'Total Points': player_row.get('avg_total_points', 0),
        'Total Fouls': player_row.get('avg_total_fouls', 0),
        '1st Quarter Points': player_row.get('avg_1st_quarter_pts', 0),
        '2nd Quarter Points': player_row.get('avg_2nd_quarter_pts', 0),
        '3rd Quarter Points': player_row.get('avg_3rd_quarter_points', 0),
        '4th Quarter Points': player_row.get('avg_4th_quarter_points', 0),
        '1st Quarter Fouls': player_row.get('avg_1st_quarter_fouls', 0),
        '2nd Quarter Fouls': player_row.get('avg_2nd_quarter_fouls', 0),
        '3rd Quarter Fouls': player_row.get('avg_3rd_quarter_fouls', 0),
        '4th Quarter Fouls': player_row.get('avg_4th_quarter_fouls', 0)
    }
    
    # Calculate rankings
    rankings = {}
    total_players = len(stats_data.df)
    
    # Total Points ranking
    if 'avg_total_points' in stats_data.df.columns:
        points_rank = stats_data.df['avg_total_points'].rank(ascending=False, method='min')
        rankings['Total Points'] = int(points_rank[player_data.index[0]])
    
    # Total Fouls ranking (lower is better)
    if 'avg_total_fouls' in stats_data.df.columns:
        fouls_rank = stats_data.df['avg_total_fouls'].rank(ascending=True, method='min')
        rankings['Total Fouls'] = int(fouls_rank[player_data.index[0]])
    
    # Quarter-by-quarter points rankings
    quarter_cols = ['avg_1st_quarter_pts', 'avg_2nd_quarter_pts', 'avg_3rd_quarter_points', 'avg_4th_quarter_points']
    quarter_names = ['1st Quarter Points', '2nd Quarter Points', '3rd Quarter Points', '4th Quarter Points']
    
    for col, name in zip(quarter_cols, quarter_names):
        if col in stats_data.df.columns:
            rank = stats_data.df[col].rank(ascending=False, method='min')
            rankings[name] = int(rank[player_data.index[0]])
    
    return key_stats, rankings, total_players

def create_player_visualizations(player_name, key_stats, rankings, total_players):
    """
    Create visualizations for the Dubai player report.
    """
    # 1. Quarter-by-Quarter Points Performance
    quarters = ['1st Quarter', '2nd Quarter', '3rd Quarter', '4th Quarter']
    quarter_points = [
        key_stats['1st Quarter Points'],
        key_stats['2nd Quarter Points'], 
        key_stats['3rd Quarter Points'],
        key_stats['4th Quarter Points']
    ]
    
    fig_quarters = go.Figure(data=[
        go.Bar(
            x=quarters,
            y=quarter_points,
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
            text=[f"{val:.1f}" for val in quarter_points],
            textposition='auto'
        )
    ])
    fig_quarters.update_layout(
        title=f"{player_name}'s Quarter-by-Quarter Points",
        yaxis_title="Average Points",
        showlegend=False
    )
    
    # 2. Quarter-by-Quarter Fouls Performance
    quarter_fouls = [
        key_stats['1st Quarter Fouls'],
        key_stats['2nd Quarter Fouls'],
        key_stats['3rd Quarter Fouls'], 
        key_stats['4th Quarter Fouls']
    ]
    
    fig_fouls = go.Figure(data=[
        go.Bar(
            x=quarters,
            y=quarter_fouls,
            marker_color=['#ff9999', '#ffcc99', '#99ff99', '#99ccff'],
            text=[f"{val:.1f}" for val in quarter_fouls],
            textposition='auto'
        )
    ])
    fig_fouls.update_layout(
        title=f"{player_name}'s Quarter-by-Quarter Fouls",
        yaxis_title="Average Fouls",
        showlegend=False
    )
    
    # 3. Points vs Fouls Radar Chart
    radar_stats = ['Total Points', 'Total Fouls']
    radar_values = [key_stats['Total Points'], key_stats['Total Fouls']]
    
    # Normalize values to 0-100 scale for radar chart
    max_points = stats_data.df['avg_total_points'].max()
    max_fouls = stats_data.df['avg_total_fouls'].max()
    
    normalized_values = []
    if max_points > 0:
        normalized_values.append((key_stats['Total Points'] / max_points) * 100)
    else:
        normalized_values.append(0)
    
    if max_fouls > 0:
        normalized_values.append((key_stats['Total Fouls'] / max_fouls) * 100)
    else:
        normalized_values.append(0)
    
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=normalized_values,
        theta=radar_stats,
        fill='toself',
        name=player_name,
        line_color='#1f77b4'
    ))
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=False,
        title=f"{player_name}'s Performance Profile"
    )
    
    # 4. Ranking Percentiles
    ranking_stats = list(rankings.keys())
    ranking_percentiles = [(total_players - rank + 1) / total_players * 100 for rank in rankings.values()]
    
    fig_rankings = go.Figure(data=[
        go.Bar(
            x=ranking_stats,
            y=ranking_percentiles,
            marker_color='#ff7f0e',
            text=[f"#{rank}" for rank in rankings.values()],
            textposition='auto'
        )
    ])
    fig_rankings.update_layout(
        title=f"{player_name}'s Rankings (Percentile)",
        yaxis_title="Percentile",
        yaxis=dict(range=[0, 100]),
        showlegend=False
    )
    
    return fig_quarters, fig_fouls, fig_radar, fig_rankings

def display_player_report(player_name, stats_data):
    """
    Display the complete player report with visualizations for Dubai data.
    """
    key_stats, rankings, total_players = create_player_report(player_name, stats_data)
    
    if key_stats is None:
        st.error(f"Player '{player_name}' not found in the Dubai dataset.")
        return
    
    # Header
    st.markdown(f"## 🏀 {player_name}'s Dubai Tournament Report")
    st.markdown("---")
    
    # Key Stats Overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Points", f"{key_stats['Total Points']:.1f}", f"#{rankings.get('Total Points', 'N/A')}")
    with col2:
        st.metric("Total Fouls", f"{key_stats['Total Fouls']:.1f}", f"#{rankings.get('Total Fouls', 'N/A')}")
    with col3:
        st.metric("Team", f"{key_stats['Team']}")
    with col4:
        st.metric("Best Quarter", f"{max(key_stats['1st Quarter Points'], key_stats['2nd Quarter Points'], key_stats['3rd Quarter Points'], key_stats['4th Quarter Points']):.1f} pts")
    
    # Create visualizations
    fig_quarters, fig_fouls, fig_radar, fig_rankings = create_player_visualizations(player_name, key_stats, rankings, total_players)
    
    # Display charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(fig_quarters, use_container_width=True)
    
    with col2:
        st.plotly_chart(fig_fouls, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(fig_radar, use_container_width=True)
    
    with col2:
        st.plotly_chart(fig_rankings, use_container_width=True)
    
    # Detailed Stats Table
    st.markdown("### 📊 Detailed Statistics")
    
    # Create a DataFrame for the detailed stats
    detailed_stats = []
    for stat, value in key_stats.items():
        if stat in rankings:
            rank = rankings[stat]
            percentile = ((total_players - rank + 1) / total_players) * 100
            detailed_stats.append({
                'Statistic': stat,
                'Value': f"{value:.1f}" if isinstance(value, (int, float)) else value,
                'Rank': f"#{rank}",
                'Percentile': f"{percentile:.1f}%"
            })
        else:
            detailed_stats.append({
                'Statistic': stat,
                'Value': f"{value:.1f}" if isinstance(value, (int, float)) else value,
                'Rank': 'N/A',
                'Percentile': 'N/A'
            })
    
    df_detailed = pd.DataFrame(detailed_stats)
    st.dataframe(df_detailed, use_container_width=True, hide_index=True)
    
    # Summary insights
    st.markdown("### 💡 Key Insights")
    
    # Find best and worst rankings
    if rankings:
        best_stat = min(rankings, key=rankings.get)
        worst_stat = max(rankings, key=rankings.get)
        
        best_rank = rankings[best_stat]
        worst_rank = rankings[worst_stat]
        
        best_percentile = ((total_players - best_rank + 1) / total_players) * 100
        worst_percentile = ((total_players - worst_rank + 1) / total_players) * 100
        
        st.info(f"**Strongest Area:** {best_stat} (#{best_rank}, {best_percentile:.1f}th percentile)")
        st.warning(f"**Area for Improvement:** {worst_stat} (#{worst_rank}, {worst_percentile:.1f}th percentile)")
    
    # Quarter performance analysis
    quarter_points = [
        key_stats['1st Quarter Points'],
        key_stats['2nd Quarter Points'],
        key_stats['3rd Quarter Points'],
        key_stats['4th Quarter Points']
    ]
    
    best_quarter_idx = quarter_points.index(max(quarter_points))
    worst_quarter_idx = quarter_points.index(min(quarter_points))
    quarter_names = ['1st', '2nd', '3rd', '4th']
    
    st.success(f"**Best Quarter:** {quarter_names[best_quarter_idx]} Quarter ({quarter_points[best_quarter_idx]:.1f} avg points)")
    st.info(f"**Needs Improvement:** {quarter_names[worst_quarter_idx]} Quarter ({quarter_points[worst_quarter_idx]:.1f} avg points)")
    
    # Foul analysis
    total_fouls = key_stats['Total Fouls']
    if total_fouls > 2:
        st.warning(f"**High Foul Rate:** {total_fouls:.1f} average fouls per game - consider improving defensive positioning")
    elif total_fouls < 0.5:
        st.success(f"**Clean Play:** {total_fouls:.1f} average fouls per game - excellent defensive discipline")

# --- Streamlit App ---
st.set_page_config(
    page_title="🏀 Dubai Basketball Stats Chat",
    page_icon="🏀",
    layout="wide"
)

st.title("🏀 PlaiMaker Dubai Basketball Stats")
st.markdown("Ask questions about Dubai basketball tournament statistics in natural language!")
st.markdown("*Analyze quarter-by-quarter performance, fouls, and team statistics*")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load data
stats_data = load_dubai_data()

# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about Dubai basketball stats..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Check if this is a player report request
        player_report_patterns = [
            r"show me (.+?)'s report",
            r"report for (.+?)(?:\s*$)",
            r"player report (.+?)(?:\s*$)",
            r"(.+?)'s report",
            r"(.+?) player report"
        ]
        
        player_name = None
        for pattern in player_report_patterns:
            # Use case-insensitive matching but preserve original case
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                player_name = match.group(1).strip()
                # Additional validation: player name should be at least 2 characters
                if len(player_name) >= 2:
                    break
                else:
                    player_name = None
        
        if player_name:
            # Display player report
            display_player_report(player_name, stats_data)
            st.session_state.messages.append({"role": "assistant", "content": f"Generated player report for {player_name}"})
        else:
            # Use pandas code generation approach for regular queries
            all_columns = stats_data.df.columns.tolist()
            numeric_columns = stats_data.df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Process the query
            try:
                # Generate pandas code
                pandas_prompt = f"""
                Convert this Dubai basketball query to pandas code:
                Query: {prompt}

                DataFrame 'df' has these columns: {all_columns}
                Numeric columns: {numeric_columns}

                IMPORTANT DATA TYPE INFORMATION:
                - All columns contain numeric values (no percentage strings like in other datasets)
                - Quarter columns represent average performance per game
                - Team names are in 'team_name' column
                - Player names are in 'player_name' column

                Generate pandas code that:
                1. Filters the data based on the query conditions
                2. Selects the relevant columns (ALWAYS include player_name and team_name)
                3. Returns the result in a readable format

                CRITICAL OUTPUT FORMAT RULES:
                - Return a pandas DataFrame with the filtered results
                - ALWAYS include player_name and team_name columns for identification
                - Store final result in 'result' variable as a DataFrame
                - DO NOT format as strings - return the DataFrame directly

                CRITICAL RULES: 
                - Use exact column names from the list above
                - ALWAYS include player_name and team_name columns for identification
                - For any "more than X [stat]" use: df['[ColumnName]'] > X
                - For any "averaged more than X [stat]" use: df['[ColumnName]'] > X
                - Use the exact column names from the DataFrame (case-sensitive)
                - For player name filtering, use case-insensitive matching: df[df['player_name'].str.lower() == 'player_name'.lower()]
                - For team name filtering, use case-insensitive matching: df[df['team_name'].str.lower() == 'team_name'.lower()]

                DUBAI BASKETBALL TERMINOLOGY INFERENCE:
                - "points" refers to avg_total_points column
                - "fouls" refers to avg_total_fouls column
                - "1st quarter" refers to avg_1st_quarter_pts and avg_1st_quarter_fouls
                - "2nd quarter" refers to avg_2nd_quarter_pts and avg_2nd_quarter_fouls
                - "3rd quarter" refers to avg_3rd_quarter_points and avg_3rd_quarter_fouls
                - "4th quarter" refers to avg_4th_quarter_points and avg_4th_quarter_fouls
                - "quarter points" refers to the specific quarter points columns
                - "quarter fouls" refers to the specific quarter fouls columns
                - "team" refers to team_name column
                - "player" refers to player_name column

                CORRECT Examples:
                - "players with more than 5 points" → df['avg_total_points'] > 5
                - "players with more than 2 fouls" → df['avg_total_fouls'] > 2
                - "players with more than 3 points in 1st quarter" → df['avg_1st_quarter_pts'] > 3
                - "players with more than 1 foul in 4th quarter" → df['avg_4th_quarter_fouls'] > 1
                - "players from Canada team" → df[df['team_name'].str.lower() == 'canada open white (19+)'.lower()]
                - "how many points did Adam Karmali score" → df[df['player_name'].str.lower() == 'adam karmali'.lower()][['player_name', 'team_name', 'avg_total_points']]

                EXAMPLE CODE FORMAT:
                # For "players with more than 5 points":
                result = df[df['avg_total_points'] > 5][['player_name', 'team_name', 'avg_total_points']]

                # For "players with more than 2 fouls":
                result = df[df['avg_total_fouls'] > 2][['player_name', 'team_name', 'avg_total_fouls']]

                # For "players with more than 3 points in 1st quarter":
                result = df[df['avg_1st_quarter_pts'] > 3][['player_name', 'team_name', 'avg_1st_quarter_pts']]

                # For "players from Canada team":
                result = df[df['team_name'].str.lower() == 'canada open white (19+)'.lower()][['player_name', 'team_name', 'avg_total_points']]

                Generate the pandas code:
                """

                response = openai.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": pandas_prompt}],
                    max_tokens=500,
                    temperature=0.1
                )

                generated_code = response.choices[0].message.content.strip()
                
                # Extract code from markdown if present
                if "```python" in generated_code:
                    generated_code = generated_code.split("```python")[1].split("```")[0].strip()
                elif "```" in generated_code:
                    generated_code = generated_code.split("```")[1].split("```")[0].strip()

                # Clean the generated code
                cleaned_code = clean_generated_code(generated_code)
                
                # Execute the code
                try:
                    # Create execution context with DataFrame
                    exec_globals = {
                        'df': stats_data.df,
                        'pd': pd,
                        'np': np,
                        'result': None
                    }
                    exec(cleaned_code, exec_globals)
                    if 'result' in exec_globals:
                        answer = exec_globals['result']
                        # If result is a DataFrame, convert to string format for compatibility
                        if isinstance(answer, pd.DataFrame):
                            if len(answer) > 0:
                                result_str = ""
                                for index, row in answer.iterrows():
                                    result_str += f"{row['player_name']} ({row['team_name']})"
                                    for col in answer.columns:
                                        if col not in ['player_name', 'team_name']:
                                            result_str += f" - {col}: {row[col]:.1f}"
                                    result_str += "\n"
                                answer = result_str
                            else:
                                answer = "No players found matching your criteria."
                        else:
                            raise Exception("No result variable found")
                        
                except Exception as e:
                    # Fallback: try a simpler approach
                    simple_prompt = f"""
                    For query: {prompt}
                    DataFrame columns: {all_columns}
                    
                    Write simple pandas code to get the data. 
                    - Always include player_name and team_name columns
                    - Store result in 'result' variable as a DataFrame
                    - For any "more than X [stat]" use: df['[ColumnName]'] > X
                    - Use exact column names from the DataFrame
                    - For player name filtering, use case-insensitive matching: df[df['player_name'].str.lower() == 'player_name'.lower()]
                    - For team name filtering, use case-insensitive matching: df[df['team_name'].str.lower() == 'team_name'.lower()]
                    
                    Example: 
                    # For "players with more than 5 points":
                    result = df[df['avg_total_points'] > 5][['player_name', 'team_name', 'avg_total_points']]
                    
                    Generate the pandas code:
                    """

                    fallback_response = openai.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": simple_prompt}],
                        max_tokens=300,
                        temperature=0.1
                    )

                    fallback_code = fallback_response.choices[0].message.content.strip()
                    
                    if "```python" in fallback_code:
                        fallback_code = fallback_code.split("```python")[1].split("```")[0].strip()
                    elif "```" in fallback_code:
                        fallback_code = fallback_code.split("```")[1].split("```")[0].strip()

                    try:
                        # Create execution context with DataFrame
                        exec_globals = {
                            'df': stats_data.df,
                            'pd': pd,
                            'np': np,
                            'result': None
                        }
                        exec(fallback_code, exec_globals)
                        if 'result' in exec_globals:
                            answer = exec_globals['result']
                            # If result is a DataFrame, convert to string format for compatibility
                            if isinstance(answer, pd.DataFrame):
                                if len(answer) > 0:
                                    result_str = ""
                                    for index, row in answer.iterrows():
                                        result_str += f"{row['player_name']} ({row['team_name']})"
                                        for col in answer.columns:
                                            if col not in ['player_name', 'team_name']:
                                                result_str += f" - {col}: {row[col]:.1f}"
                                        result_str += "\n"
                                    answer = result_str
                                else:
                                    answer = "No players found matching your criteria."
                        else:
                            raise Exception("No result variable found in fallback")
                        
                    except Exception as fallback_error:
                        # Final fallback: pattern matching
                        
                        # Extract numbers and keywords from query
                        numbers = re.findall(r'\d+', prompt)
                        threshold = int(numbers[0]) if numbers else 5
                        
                        # Find the most relevant column by looking at the DataFrame columns
                        query_lower = prompt.lower()
                        column = None
                        
                        # Look for any column that might match the query
                        for col in stats_data.df.columns:
                            col_lower = col.lower()
                            # Check if any word from the query appears in the column name
                            query_words = query_lower.split()
                            for word in query_words:
                                if len(word) > 2 and word in col_lower:
                                    column = col
                                    break
                            if column:
                                break
                        
                        if column:
                            # Check if the column exists and has numeric data
                            if column in stats_data.df.columns:
                                # Check if column is numeric
                                if column in numeric_columns:
                                    filtered = stats_data.df[stats_data.df[column] > threshold]
                                    if len(filtered) > 0:
                                        result = f"Players with more than {threshold} {column}:\n\n"
                                        for index, row in filtered.iterrows():
                                            result += f"{row['player_name']} ({row['team_name']}) - {column}: {row[column]:.1f}\n"
                                        
                                        result += f"\nTotal: {len(filtered)} players found"
                                        answer = result
                                    else:
                                        answer = f"No players found with more than {threshold} {column}."
                                else:
                                    answer = f"Sorry, I couldn't understand that stat. Try asking about points, fouls, or quarter-by-quarter performance."
                            else:
                                answer = f"Sorry, I couldn't find that stat in our data. Try asking about points, fouls, or quarter-by-quarter performance."
                        else:
                            answer = "I'm not sure which stat you're asking about. Try being more specific, like 'players with more than 10 points' or 'players with more than 2 fouls'."

            except Exception as e:
                answer = "Sorry, I'm having trouble understanding your question. Try asking about specific basketball stats like points, fouls, or quarter-by-quarter performance."
            
            # Format the output nicely
            formatted_result = format_query_result(answer, prompt)
            
            # Display the result
            if isinstance(formatted_result, pd.DataFrame):
                # If it's a DataFrame, use st.dataframe for interactive table
                output_header, _ = create_nice_output(None, prompt)
                st.markdown(output_header)
                
                # Display the interactive dataframe
                st.dataframe(
                    formatted_result,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "player_name": st.column_config.TextColumn(
                            "Player",
                            width="medium",
                            help="Player name"
                        ),
                        "team_name": st.column_config.TextColumn(
                            "Team",
                            width="medium",
                            help="Team name"
                        )
                    }
                )
                
                # Show summary stats
                st.info(f"📊 Found {len(formatted_result)} players matching your criteria")
                
            else:
                # If it's regular text, use markdown
                output_header, _ = create_nice_output(None, prompt)
                st.markdown(output_header)
                st.markdown(formatted_result)
            
            st.session_state.messages.append({"role": "assistant", "content": answer}) 
"""
PlaiMaker - Basketball Stats Chat

A Streamlit app that lets you ask questions about basketball stats in plain English.
Uses GPT-4o to convert your questions into data queries automatically.
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
from stats_helpers import StatsHelper

# Set up OpenAI API key from Streamlit secrets
openai_api_key = st.secrets["openai"]["api_key"]
openai.api_key = openai_api_key

# Load the basketball stats data
@st.cache_data
def load_data():
    """Load the basketball statistics dataset."""
    stats_data = StatsHelper('stats_all.csv')
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
    
    # Fix per-game calculations - we want total stats, not per-game
    code = re.sub(r"df\['([^']+)'\]\s*/\s*df\['Games'\]", r"df['\1']", code)
    
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
                    stats_dict = {"Player": player}
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
                            current_stats["Player"] = current_player
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
                current_stats["Player"] = current_player
                table_data.append(current_stats)
            
            if table_data:
                players_data = table_data
        
        # Pattern 3: Look for any structured data with player names and stats
        if not players_data:
            # Try to find any pattern that looks like player data
            for line in lines:
                line = line.strip()
                if line and len(line) > 10:  # Reasonable length for player data
                    # Look for patterns like "Name - Stat: Value" or "Name Stat: Value"
                    if " - " in line or ":" in line:
                        # Try to extract player name and stats
                        if " - " in line:
                            parts = line.split(" - ")
                            if len(parts) >= 2:
                                player = parts[0].strip()
                                stats_part = " - ".join(parts[1:])
                                
                                stats_dict = {"Player": player}
                                # Look for stat patterns
                                stat_matches = re.findall(r'([^:]+):\s*([^,]+)', stats_part)
                                for stat_name, stat_value in stat_matches:
                                    stats_dict[stat_name.strip()] = stat_value.strip()
                                
                                if len(stats_dict) > 1:  # Has player + at least one stat
                                    players_data.append(stats_dict)
        
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
## 🏀 Basketball Stats Results

**Your Question:** {query}

"""
    
    return output_header, result_data

def create_player_report(player_name, stats_data):
    """
    Create a comprehensive player report with visualizations and rankings.
    """
    # Find the player in the dataset
    player_data = stats_data.df[stats_data.df['Player'].str.lower() == player_name.lower()]
    
    if player_data.empty:
        return None, None, None
    
    player_row = player_data.iloc[0]
    
    # Get key stats for the player
    key_stats = {
        'Points': player_row.get('Pts.', 0),
        'Rebounds': player_row.get('Reb', 0),
        'Assists': player_row.get('Ast', 0),
        'Steals': player_row.get('Stl', 0),
        'Blocks': player_row.get('Blk', 0),
        'Turnovers': player_row.get('TO', 0),
        'Fouls': player_row.get('Fouls', 0),
        'Games': player_row.get('Games', 0),
        'Minutes': player_row.get('Time', '0:00:00'),
        'EFG%': player_row.get('EFG%', '0%'),
        'FT%': player_row.get('FT%', '0%'),
        '3Pt%': player_row.get('3Pt%', '0%'),
        'Usage %': player_row.get('Usage %', '0%'),
        'Off RTG': player_row.get('Off. Rtg', 0),
        'Def RTG': player_row.get('Def. Rtg', 0),
        'Net RTG': player_row.get('Net PPP', 0)
    }
    
    # Calculate rankings
    rankings = {}
    total_players = len(stats_data.df)
    
    # Points ranking
    if 'Pts.' in stats_data.df.columns:
        points_rank = stats_data.df['Pts.'].rank(ascending=False, method='min')
        rankings['Points'] = int(points_rank[player_data.index[0]])
    
    # Rebounds ranking
    if 'Reb' in stats_data.df.columns:
        reb_rank = stats_data.df['Reb'].rank(ascending=False, method='min')
        rankings['Rebounds'] = int(reb_rank[player_data.index[0]])
    
    # Assists ranking
    if 'Ast' in stats_data.df.columns:
        ast_rank = stats_data.df['Ast'].rank(ascending=False, method='min')
        rankings['Assists'] = int(ast_rank[player_data.index[0]])
    
    # EFG% ranking (handle percentage strings)
    if 'EFG%' in stats_data.df.columns:
        efg_numeric = pd.to_numeric(stats_data.df['EFG%'].str.replace('%', ''), errors='coerce')
        efg_rank = efg_numeric.rank(ascending=False, method='min')
        rankings['EFG%'] = int(efg_rank[player_data.index[0]])
    
    # Offensive Rating ranking
    if 'Off. Rtg' in stats_data.df.columns:
        off_rtg_rank = stats_data.df['Off. Rtg'].rank(ascending=False, method='min')
        rankings['Off RTG'] = int(off_rtg_rank[player_data.index[0]])
    
    # Defensive Rating ranking
    if 'Def. Rtg' in stats_data.df.columns:
        def_rtg_rank = stats_data.df['Def. Rtg'].rank(ascending=True, method='min')  # Lower is better
        rankings['Def RTG'] = int(def_rtg_rank[player_data.index[0]])
    
    return key_stats, rankings, total_players

def create_player_visualizations(player_name, key_stats, rankings, total_players):
    """
    Create visualizations for the player report.
    """
    # 1. Key Stats Radar Chart
    radar_stats = ['Points', 'Rebounds', 'Assists', 'Steals', 'Blocks']
    radar_values = [key_stats[stat] for stat in radar_stats]
    
    # Normalize values to 0-100 scale for radar chart
    max_values = {
        'Points': stats_data.df['Pts.'].max(),
        'Rebounds': stats_data.df['Reb'].max(),
        'Assists': stats_data.df['Ast'].max(),
        'Steals': stats_data.df['Stl'].max(),
        'Blocks': stats_data.df['Blk'].max()
    }
    
    normalized_values = []
    for stat, value in zip(radar_stats, radar_values):
        if max_values[stat] > 0:
            normalized_values.append((value / max_values[stat]) * 100)
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
    
    # 2. Ranking Percentiles
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
    
    # 3. Shooting Percentages
    shooting_stats = ['EFG%', 'FT%', '3Pt%']
    shooting_values = []
    for stat in shooting_stats:
        value = key_stats[stat]
        if isinstance(value, str) and '%' in value:
            shooting_values.append(float(value.replace('%', '')))
        else:
            shooting_values.append(0)
    
    fig_shooting = go.Figure(data=[
        go.Bar(
            x=shooting_stats,
            y=shooting_values,
            marker_color=['#2ca02c', '#d62728', '#9467bd'],
            text=[f"{val:.1f}%" for val in shooting_values],
            textposition='auto'
        )
    ])
    fig_shooting.update_layout(
        title=f"{player_name}'s Shooting Percentages",
        yaxis_title="Percentage",
        yaxis=dict(range=[0, 100]),
        showlegend=False
    )
    
    return fig_radar, fig_rankings, fig_shooting

def display_player_report(player_name, stats_data):
    """
    Display the complete player report with visualizations.
    """
    key_stats, rankings, total_players = create_player_report(player_name, stats_data)
    
    if key_stats is None:
        st.error(f"Player '{player_name}' not found in the dataset.")
        return
    
    # Header
    st.markdown(f"## 🏀 {player_name}'s Player Report")
    st.markdown("---")
    
    # Key Stats Overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Points", f"{key_stats['Points']}", f"#{rankings.get('Points', 'N/A')}")
    with col2:
        st.metric("Rebounds", f"{key_stats['Rebounds']}", f"#{rankings.get('Rebounds', 'N/A')}")
    with col3:
        st.metric("Assists", f"{key_stats['Assists']}", f"#{rankings.get('Assists', 'N/A')}")
    with col4:
        st.metric("Games", f"{key_stats['Games']}")
    
    # Create visualizations
    fig_radar, fig_rankings, fig_shooting = create_player_visualizations(player_name, key_stats, rankings, total_players)
    
    # Display charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(fig_radar, use_container_width=True)
    
    with col2:
        st.plotly_chart(fig_rankings, use_container_width=True)
    
    st.plotly_chart(fig_shooting, use_container_width=True)
    
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
                'Value': value,
                'Rank': f"#{rank}",
                'Percentile': f"{percentile:.1f}%"
            })
        else:
            detailed_stats.append({
                'Statistic': stat,
                'Value': value,
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
    
    # Usage and efficiency
    if key_stats['Usage %'] != '0%':
        usage = float(key_stats['Usage %'].replace('%', ''))
        if usage > 25:
            st.success(f"**High Usage Player:** {usage:.1f}% usage rate indicates primary offensive option")
        elif usage < 15:
            st.info(f"**Role Player:** {usage:.1f}% usage rate suggests supporting role")
    
    # Net rating analysis
    if key_stats['Net RTG'] != 0:
        net_rtg = key_stats['Net RTG']
        if net_rtg > 0.1:
            st.success(f"**Positive Impact:** +{net_rtg:.3f} net rating shows positive team contribution")
        elif net_rtg < -0.1:
            st.warning(f"**Negative Impact:** {net_rtg:.3f} net rating indicates room for improvement")
        else:
            st.info(f"**Neutral Impact:** {net_rtg:.3f} net rating shows balanced performance")

# --- Streamlit App ---
st.set_page_config(
    page_title="🏀 Basketball Stats Chat",
    page_icon="🏀",
    layout="wide"
)

# No custom CSS needed - using native Streamlit components

st.title("🏀 PlaiMaker Basketball Stats")
st.markdown("Ask questions about basketball statistics in natural language!")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load data
stats_data = load_data()

# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about basketball stats..."):
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
                Convert this basketball query to pandas code:
                Query: {prompt}

                DataFrame 'df' has these columns: {all_columns}
                Numeric columns: {numeric_columns}

                IMPORTANT DATA TYPE INFORMATION:
                - Percentage columns (like EFG%, FT%, 2Pt%, 3Pt%, etc.) contain string values like "50.2%" 
                - These need to be converted to numeric before comparison
                - Use: pd.to_numeric(df['ColumnName'].str.replace('%', ''), errors='coerce') > threshold

                Generate pandas code that:
                1. Filters the data based on the query conditions
                2. Selects the relevant columns (ALWAYS include Player and any stats mentioned)
                3. Returns the result in a readable format

                CRITICAL OUTPUT FORMAT RULES:
                - Return a pandas DataFrame with the filtered results
                - ALWAYS include Player column and any stats mentioned in the query
                - Store final result in 'result' variable as a DataFrame
                - DO NOT format as strings - return the DataFrame directly

                CRITICAL RULES: 
                - Use exact column names from the list above
                - ALWAYS include Player column for identification
                - NEVER calculate per-game averages unless explicitly asked for "per game" or "average per game"
                - For any "more than X [stat]" use: df['[ColumnName]'] > X (NOT df['[ColumnName]'] / df['Games'] > X)
                - For any "averaged more than X [stat]" use: df['[ColumnName]'] > X (NOT df['[ColumnName]'] / df['Games'] > X)
                - The word "averaged" in queries means total stats, not per-game averages
                - Use the exact column names from the DataFrame (case-sensitive)
                - For player name filtering, use case-insensitive matching: df[df['Player'].str.lower() == 'player_name'.lower()]

                BASKETBALL TERMINOLOGY INFERENCE:
                - "EFG%" refers to Effective Field Goal Percentage column
                - "assists" refers to "Ast" column
                - "points" refers to "Pts." column
                - "rebounds" refers to "Reb" column
                - "steals" refers to "Stl" column
                - "blocks" refers to "Blk" column
                - "turnovers" refers to "TO" column
                - "fouls" refers to "Fouls" column
                - "free throws" refers to "FT" column
                - "shots" refers to "Shots" column
                - "field goals" refers to "EFG%" column
                - "goals" refers to "EFG%" column
                - "throws" refers to "FT%" column

                FOR PERCENTAGE COLUMNS (EFG%, FT%, 2Pt%, 3Pt%, etc.):
                - These contain strings like "50.2%" 
                - Convert to numeric first: pd.to_numeric(df['EFG%'].str.replace('%', ''), errors='coerce') > 50
                - For "more than 50 EFG%" use: pd.to_numeric(df['EFG%'].str.replace('%', ''), errors='coerce') > 50

                CORRECT Examples:
                - "players with more than 5 assists" → df['Ast'] > 5
                - "players with more than 20 points" → df['Pts.'] > 20  
                - "players with more than 10 rebounds" → df['Reb'] > 10
                - "players with more than 3 steals" → df['Stl'] > 3
                - "players with more than 2 blocks" → df['Blk'] > 2
                - "players with more than 5 turnovers" → df['TO'] > 5
                - "players with more than 8 fouls" → df['Fouls'] > 8
                - "players with more than 10 free throws" → df['FT'] > 10
                - "players with more than 15 shots" → df['Shots'] > 15
                - "players with more than 50 EFG%" → pd.to_numeric(df['EFG%'].str.replace('%', ''), errors='coerce') > 50

                WRONG Examples (don't do this):
                - df['Ast'] / df['Games'] > 5  (per-game calculation)
                - df['Pts.'] / df['Games'] > 20 (per-game calculation)
                - df['EFG%'] > 50 (string comparison with percentage)

                EXAMPLE CODE FORMAT:
                # For "players with more than 5 assists":
                result = df[df['Ast'] > 5][['Player', 'Pts.', 'Ast']]

                # For "players with more than 50 EFG%":
                result = df[pd.to_numeric(df['EFG%'].str.replace('%', ''), errors='coerce') > 50][['Player', 'Pts.', 'Reb', 'EFG%']]

                # For "which 7 players had the best def rtg but also an off rtg higher than 110":
                filtered = df[df['Off RTG'] > 110]
                result = filtered.nlargest(7, 'Def RTG')[['Player', 'Def RTG', 'Off RTG']]

                # For "how many points did AJ Dybansta score":
                result = df[df['Player'].str.lower() == 'aj dybansta'.lower()][['Player', 'Pts.']]

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
                                    result_str += f"{row['Player']}"
                                    for col in answer.columns:
                                        if col != 'Player':
                                            result_str += f" - {col}: {row[col]}"
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
                    - Always include Player column
                    - Store result in 'result' variable as a DataFrame
                    - NEVER calculate per-game averages (use total stats)
                    - For any "more than X [stat]" use: df['[ColumnName]'] > X
                    - For any "averaged more than X [stat]" use: df['[ColumnName]'] > X (NOT df['[ColumnName]'] / df['Games'] > X)
                    - The word "averaged" in queries means total stats, not per-game averages
                    - Return DataFrame directly, not formatted strings
                    - Use exact column names from the DataFrame
                    - For player name filtering, use case-insensitive matching: df[df['Player'].str.lower() == 'player_name'.lower()]
                    
                    FOR PERCENTAGE COLUMNS (EFG%, FT%, 2Pt%, 3Pt%, etc.):
                    - These contain strings like "50.2%" 
                    - Convert to numeric first: pd.to_numeric(df['EFG%'].str.replace('%', ''), errors='coerce') > 50
                    - For "more than 50 EFG%" use: pd.to_numeric(df['EFG%'].str.replace('%', ''), errors='coerce') > 50
                    
                    Example: 
                    # For "players with more than 5 assists":
                    result = df[df['Ast'] > 5][['Player', 'Pts.', 'Ast']]
                    
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
                                        result_str += f"{row['Player']}"
                                        for col in answer.columns:
                                            if col != 'Player':
                                                result_str += f" - {col}: {row[col]}"
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
                                    # Skip non-statistical columns
                                    if col not in ['Recruiting\nProfile', 'Unnamed: 1']:
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
                                            result += f"{row['Player']} - Points: {row['Pts.']}, {column}: {row[column]}\n"
                                        
                                        result += f"\nTotal: {len(filtered)} players found"
                                        answer = result
                                    else:
                                        answer = f"No players found with more than {threshold} {column}."
                                else:
                                    # Handle percentage columns
                                    if '%' in column:
                                        # Convert percentage strings to numeric
                                        try:
                                            filtered = stats_data.df[pd.to_numeric(stats_data.df[column].str.replace('%', ''), errors='coerce') > threshold]
                                            if len(filtered) > 0:
                                                result = f"Players with more than {threshold}% {column}:\n\n"
                                                for index, row in filtered.iterrows():
                                                    result += f"{row['Player']} - Points: {row['Pts.']}, {column}: {row[column]}\n"
                                                    
                                                result += f"\nTotal: {len(filtered)} players found"
                                                answer = result
                                            else:
                                                answer = f"No players found with more than {threshold}% {column}."
                                        except Exception as e:
                                            answer = f"Sorry, I couldn't process that percentage stat. Try asking about a different stat."
                                    else:
                                        answer = f"Sorry, I couldn't understand that stat. Try asking about points, rebounds, assists, or other common basketball stats."
                            else:
                                answer = f"Sorry, I couldn't find that stat in our data. Try asking about points, rebounds, assists, steals, blocks, or other common basketball stats."
                        else:
                            answer = "I'm not sure which stat you're asking about. Try being more specific, like 'players with more than 10 points' or 'players with more than 5 assists'."

            except Exception as e:
                answer = "Sorry, I'm having trouble understanding your question. Try asking about specific basketball stats like points, rebounds, assists, or shooting percentages."
            
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
                        "Player": st.column_config.TextColumn(
                            "Player",
                            width="medium",
                            help="Player name"
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
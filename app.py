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
    Tries to parse player data and create an HTML table if possible.
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
        
        # If we found structured data, create a table
        if players_data:
            # Create a DataFrame and display as table
            df_result = pd.DataFrame(players_data)
            return df_result.to_html(index=False, classes=['table', 'table-striped', 'table-hover'], escape=False)
        
        # If no structured data found, return formatted text
        return result_text.replace("  ", "\n  ").replace(" - ", "\n  ")
    
    except Exception as e:
        # If formatting fails, return the original with basic cleanup
        return result_text.replace("  ", "\n  ")

def create_nice_output(result_text, query):
    """
    Create a nicely formatted output with the query context.
    """
    # Format the main result
    formatted_result = format_query_result(result_text, query)
    
    # Create the output
    output = f"""
## 🏀 Basketball Stats Results

**Your Question:** {query}

{formatted_result}
"""
    
    return output

# --- Streamlit App ---
st.set_page_config(
    page_title="🏀 Basketball Stats Chat",
    page_icon="🏀",
    layout="wide"
)

# Custom CSS for better table styling
st.markdown("""
<style>
/* Override any Streamlit default styling */
.table {
    border-collapse: collapse;
    width: 100%;
    margin: 10px 0;
    font-family: Arial, sans-serif;
    background-color: white;
}

.table th {
    background-color: #f0f2f6;
    color: #262730 !important;
    font-weight: bold;
    padding: 12px 8px;
    text-align: left;
    border-bottom: 2px solid #e0e0e0;
}

.table td {
    padding: 10px 8px;
    border-bottom: 1px solid #e0e0e0;
    vertical-align: top;
    color: #262730 !important;
    background-color: transparent;
}

.table-striped tbody tr:nth-child(even) {
    background-color: #f8f9fa;
}

.table-striped tbody tr:nth-child(odd) {
    background-color: #ffffff;
}

.table-hover tbody tr:hover {
    background-color: #d1ecf1;
}

.table th:first-child {
    font-weight: bold;
    color: #1f77b4 !important;
}

/* Force text color on all table elements */
.table td, .table th {
    color: #262730 !important;
}

/* Ensure hover state maintains readability */
.table-hover tbody tr:hover td {
    color: #262730 !important;
    background-color: #d1ecf1;
}

/* Override any Streamlit dataframe styling */
.dataframe.table td, .dataframe.table th {
    color: #262730 !important;
}

/* Additional safety for text readability */
table.table td, table.table th {
    color: #262730 !important;
}
</style>
""", unsafe_allow_html=True)

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
        # Use pandas code generation approach for all queries
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
            - ALWAYS format output as: "PlayerName - Stat1: Value1, Stat2: Value2"
            - Example: "John Smith - Points: 25, Rebounds: 10"
            - Use this exact format for ALL players
            - Include line breaks between players
            - Store final result in 'result' variable as a string

            CRITICAL RULES: 
            - Use exact column names from the list above
            - ALWAYS include Player column for identification
            - NEVER calculate per-game averages unless explicitly asked for "per game" or "average per game"
            - For any "more than X [stat]" use: df['[ColumnName]'] > X (NOT df['[ColumnName]'] / df['Games'] > X)
            - For any "averaged more than X [stat]" use: df['[ColumnName]'] > X (NOT df['[ColumnName]'] / df['Games'] > X)
            - The word "averaged" in queries means total stats, not per-game averages
            - Use the exact column names from the DataFrame (case-sensitive)

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
            filtered = df[df['Ast'] > 5][['Player', 'Pts.', 'Ast']]
            result = ""
            for index, row in filtered.iterrows():
                result += f"{{row['Player']}} - Points: {{row['Pts.']}}, Assists: {{row['Ast']}}\\n"

            # For "players with more than 50 EFG%":
            filtered = df[pd.to_numeric(df['EFG%'].str.replace('%', ''), errors='coerce') > 50][['Player', 'Pts.', 'Reb', 'EFG%']]
            result = ""
            for index, row in filtered.iterrows():
                result += f"{{row['Player']}} - Points: {{row['Pts.']}}, Rebounds: {{row['Reb']}}, EFG%: {{row['EFG%']}}\\n"

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
                else:
                    raise Exception("No result variable found")
                    
            except Exception as e:
                # Fallback: try a simpler approach
                simple_prompt = f"""
                For query: {prompt}
                DataFrame columns: {all_columns}
                
                Write simple pandas code to get the data. 
                - Always include Player column
                - Store result in 'result' variable
                - NEVER calculate per-game averages (use total stats)
                - For any "more than X [stat]" use: df['[ColumnName]'] > X
                - For any "averaged more than X [stat]" use: df['[ColumnName]'] > X (NOT df['[ColumnName]'] / df['Games'] > X)
                - The word "averaged" in queries means total stats, not per-game averages
                - Format nicely with line breaks, not to_string()
                - Use exact column names from the DataFrame
                
                FOR PERCENTAGE COLUMNS (EFG%, FT%, 2Pt%, 3Pt%, etc.):
                - These contain strings like "50.2%" 
                - Convert to numeric first: pd.to_numeric(df['EFG%'].str.replace('%', ''), errors='coerce') > 50
                - For "more than 50 EFG%" use: pd.to_numeric(df['EFG%'].str.replace('%', ''), errors='coerce') > 50
                
                Example: 
                # For "players with more than 5 assists":
                filtered = df[df['Ast'] > 5][['Player', 'Pts.', 'Ast']]
                result = ""
                for index, row in filtered.iterrows():
                    result += f"{{row['Player']}} - Points: {{row['Pts.']}}, Assists: {{row['Ast']}}\\n"
                
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
        formatted_answer = create_nice_output(answer, prompt)
        
        # Display the result
        if "<table" in formatted_answer:
            # If it's a table, render as HTML
            st.write(formatted_answer, unsafe_allow_html=True)
        else:
            # If it's regular text, use markdown
            st.markdown(formatted_answer)
        
        st.session_state.messages.append({"role": "assistant", "content": answer}) 
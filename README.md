# PlaiMaker - Basketball Stats Chat

*Basketball statistics and video sit in separate tools. Coaches, analysts, fans, and players move between spreadsheets, video editors, and web dashboards to connect numbers with film. This fragmentation wastes time and leads to missed insights.*

*Our product combines Hoopsalytics' stat-by-video pipeline with an AI native, interactive layer. A user asks a question, receives an answer with supporting charts, and opens the exact clips in the click of a button.*

*The unified workspace cuts analysis time from hours to minutes. All stakeholders see the same data, clips, and annotations, which improves decision-making and engagement.*

## What This Is

This is a proof-of-concept for the AI chat interface. Right now it handles the "ask a question, get an answer" part using basketball statistics. You can ask things like:

- "players with more than 5 assists"
- "list players that averaged more than 50 EFG% and show their points and rebounds average"
- "show me top 10 scorers"

Our AI understands basketball terminology and converts your questions into data queries automatically.

## Getting Started

### Prerequisites
- Python 3.8+
- OpenAI API key

### Setup

1. Clone this repo
2. Install dependencies: `pip install -r requirements.txt`
3. Add your OpenAI API key to `.streamlit/secrets.toml`:
   ```toml
   [openai]
   api_key = "your-openai-api-key-here"
   ```
4. Run: `streamlit run app.py`

## How It Works

The app uses GPT-4o to understand your basketball questions and generate pandas code to query the data. It has multiple fallback layers to handle edge cases and ensure you always get an answer.

Key features:
- Natural language queries ("players with more than 50 EFG%")
- Automatic percentage column handling (EFG%, FT%, etc.)
- Beautiful table output with proper styling
- Smart fallbacks when the AI gets confused

## Current Limitations

This is an early prototype, so:
- Only handles statistics queries (no video integration yet)
- Uses a sample dataset of player stats
- Limited to filtering and ranking operations
- No user accounts or data persistence

## Tech Stack

- **Frontend**: Streamlit
- **AI**: OpenAI GPT-4o
- **Data**: Pandas + CSV
- **Styling**: Custom CSS

## Development

This is a prototype built to test the AI chat interface concept. The code is functional but not production-ready.

## License

This project is provided for demonstration purposes only.  
© 2025 Sameer Chaturvedi. All rights reserved.  
No part of this codebase may be copied, reproduced, or used for commercial or educational purposes without explicit permission.

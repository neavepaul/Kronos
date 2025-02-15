import requests
import chess
import time
from bs4 import BeautifulSoup

# Function to scrape Syzygy Tablebase website
def query_syzygy_tables(fen):
    url = f"https://syzygy-tables.info/?fen={fen.replace(' ', '_')}"
    start = time.time()
    response = requests.get(url)
    end = time.time()
    print(f"🕒 API Request Time: {end - start:.2f} seconds")

    if response.status_code != 200:
        print(f"⚠️ Syzygy Website Error: {response.status_code}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    # Extract DTZ & DTM
    dtz_span = soup.select_one("h2 span.badge:nth-of-type(2)")
    dtm_span = soup.select_one("h2 span.badge:nth-of-type(1)")

    dtz = int(dtz_span.text.split()[-1]) if dtz_span else None
    dtm = int(dtm_span.text.split()[-1]) if dtm_span else None

    # Determine game status
    status_header = soup.select_one("#status").text.strip().lower()
    
    if "white is winning" in status_header:
        game_status = "🔥 White is Winning!"
    elif "black is winning" in status_header:
        game_status = "🔥 Black is Winning!"
    elif "white is losing" in status_header:
        game_status = "🚨 White is Losing!"
    elif "black is losing" in status_header:
        game_status = "🚨 Black is Losing!"
    else:
        game_status = "⚖️ Drawn position."

    # Determine who has to move
    turn_to_move = "White" if "_w_" in url else "Black"

    # Extract Best Move
    best_move_element = soup.select_one("#winning .li")
    
    if not best_move_element and "losing" in game_status.lower():
        best_move_element = soup.select_one("#losing .li")

    best_move_uci = best_move_element["data-uci"] if best_move_element else None

    # Extract Win Probability Stats
    win_stats = soup.select(".list-group.stats .li")

    try:
        white_wins = int(win_stats[0].text.split(":")[1].strip().split(" ")[0].replace(",", ""))
    except (IndexError, ValueError):
        white_wins = 0

    try:
        draws = int(win_stats[1].text.split(":")[1].strip().split(" ")[0].replace(",", ""))
    except (IndexError, ValueError):
        draws = 0

    try:
        black_wins = int(win_stats[2].text.split(":")[1].strip().split(" ")[0].replace(",", ""))
    except (IndexError, ValueError):
        black_wins = 0

    total = white_wins + draws + black_wins
    white_win_percent = (white_wins / total) * 100 if total > 0 else 0
    draw_percent = (draws / total) * 100 if total > 0 else 0
    black_win_percent = (black_wins / total) * 100 if total > 0 else 0

    return {
        "dtz": dtz,
        "dtm": dtm,
        "game_status": game_status,
        "turn_to_move": turn_to_move,
        "best_move": best_move_uci,
        "white_win_percent": white_win_percent,
        "draw_percent": draw_percent,
        "black_win_percent": black_win_percent
    }

# Test Position
fen = "8/8/8/1Q6/8/8/k7/2K5 w - - 0 1"
board = chess.Board(fen)

# Query Syzygy Website
tablebase_data = query_syzygy_tables(fen)

if tablebase_data:
    # Get Data
    white_win_percent = tablebase_data["white_win_percent"]
    draw_percent = tablebase_data["draw_percent"]
    black_win_percent = tablebase_data["black_win_percent"]

    turn_to_move = tablebase_data["turn_to_move"]
    game_status = tablebase_data["game_status"]
    best_move_uci = tablebase_data.get("best_move")
    dtz = tablebase_data.get("dtz")
    dtm = tablebase_data.get("dtm")

    # Convert best move from UCI to SAN notation
    best_move_san = board.san(chess.Move.from_uci(best_move_uci)) if best_move_uci else "None"

    # ASCII Stacked Bar Function
    def stacked_bar(white_pct, draw_pct, black_pct, length=40):
        white_length = int((white_pct / 100) * length)
        draw_length = int((draw_pct / 100) * length)
        black_length = length - (white_length + draw_length)  # Remaining space

        bar = "█" * white_length + "▒" * draw_length + "░" * black_length
        return f"|{bar}| {white_pct:.1f}% / {draw_pct:.1f}% / {black_pct:.1f}%"

    # Display Stats in Terminal
    print("\n" + "=" * 50)
    print(f"♟️ {game_status}")
    print(f"🔄 Turn to Move: {turn_to_move}")
    print(f"📉 DTZ: {dtz} | 📈 DTM: {dtm}")
    print(f"🔥 Best Move: {best_move_san}")

    print("\n🎯 **Win Predictor Bar**")
    print(stacked_bar(white_win_percent, draw_percent, black_win_percent))
    print("=" * 50 + "\n")

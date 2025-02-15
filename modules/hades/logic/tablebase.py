import chess
import chess.syzygy
from pathlib import Path

# Load Syzygy
SYZYGY_PATH = Path("syzygy")
tablebase = chess.syzygy.Tablebase()
tablebase.add_directory(str(SYZYGY_PATH))

# 🔍 Test Position
fen = "8/8/8/8/8/3k4/nQ6/1K6 b - - 0 1"  # White to move
board = chess.Board(fen)

try:
    # Probe WDL
    wdl = tablebase.probe_wdl(board)

    if wdl > 0:
        print(f"🔥 White is Winning!")  
    elif wdl < 0:
        print(f"🚨 Player is Losing!")  
    else:
        print(f"⚖️ Drawn position. No progress possible.")

    dtz = tablebase.probe_dtz(board)
    print(f"Current DTZ: {dtz}")

    # Move Selection (Find best DTZ & restrict Black's movement)
    best_move = None
    best_dtz = None
    best_restrictive_move = None
    min_dtz = float("inf")

    for move in list(board.legal_moves):  # Convert generator to list for stability
        board.push(move)
        try:
            move_dtz = tablebase.probe_dtz(board)

            # Find the lowest DTZ move (fastest mate)
            if move_dtz < min_dtz:
                min_dtz = move_dtz
                best_move = move
                best_restrictive_move = move  # Default to best DTZ move

            # If another move has the same DTZ, check which one restricts Black more
            elif move_dtz == min_dtz:
                # Heuristic: Choose moves that push Black’s king toward checkmate
                opponent_moves_before = len(list(board.legal_moves))  # Before moving
                board.push(move)
                opponent_moves_after = len(list(board.legal_moves))  # After moving
                board.pop()  # Ensures balanced pop

                # If this move reduces Black's legal moves, favor it
                if opponent_moves_after < opponent_moves_before:
                    best_restrictive_move = move

        except chess.syzygy.MissingTableError:
            pass  # Ignore missing tablebase positions
        finally:
            board.pop()  # Ensures pop is always executed correctly

    # Pick the move that both minimizes DTZ & restricts Black
    final_move = best_restrictive_move if best_restrictive_move else best_move

    if final_move:
        print(f"🔥 Best Move: {final_move} (DTZ: {min_dtz})")
    else:
        print("⚠️ No best move found.")

except chess.syzygy.MissingTableError:
    print("⚠️ No Syzygy table found for this position.")

# Close Tablebase
tablebase.close()

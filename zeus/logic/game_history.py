class GameHistory:
    def __init__(self):
        """
        Initialize an empty game history.
        """
        self.history = []

    def add_move(self, fen, move):
        """
        Add a move to the game history.
        :param fen: FEN string of the position before the move.
        :param move: Move made in UCI format.
        """
        self.history.append({"fen": fen, "move": move})

    def get_history(self):
        """
        Retrieve the complete game history.
        :return: List of dicts containing FEN strings and moves.
        """
        return self.history

    def last_position(self):
        """
        Retrieve the last position (FEN) and move in the history.
        :return: Last position as a dict, or None if the history is empty.
        """
        return self.history[-1] if self.history else None

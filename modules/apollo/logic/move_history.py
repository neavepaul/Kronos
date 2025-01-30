class MoveHistory:
    def __init__(self):
        """
        Initialize move history tracking.
        """
        self.history = []

    def add_move(self, move):
        """
        Add a move to the history.
        :param move: A move in UCI format.
        """
        self.history.append(move)

    def get_history(self):
        """
        Get the full move history.
        :return: List of moves in UCI format.
        """
        return self.history

    def last_n_moves(self, n):
        """
        Get the last n moves from the history.
        :param n: Number of moves to retrieve.
        :return: List of the last n moves.
        """
        return self.history[-n:]

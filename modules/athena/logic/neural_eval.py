import torch
import torch.nn as nn

class NeuralEvaluator(nn.Module):
    def __init__(self):
        super(NeuralEvaluator, self).__init__()
        self.fc1 = nn.Linear(64 * 12, 128)  # 64 squares * 12 piece types
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def evaluate_position_nn(board_tensor):
    """
    Evaluate a position using the neural network.
    :param board_tensor: Tensor representation of the board.
    :return: Score as a single float.
    """
    model = NeuralEvaluator()
    model.load_state_dict(torch.load("path_to_model.pth"))
    model.eval()
    with torch.no_grad():
        return model(board_tensor).item()

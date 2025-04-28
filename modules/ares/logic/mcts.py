import numpy as np
import chess
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging

# Configure logging
logger = logging.getLogger('ares.mcts')

@dataclass
class MCTSNode:
    state: chess.Board
    prior: float
    turn: bool  # True for white, False for black
    visit_count: int = 0
    value_sum: float = 0
    children: Dict[chess.Move, 'MCTSNode'] = None
    
    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
    
    def expanded(self) -> bool:
        return self.children is not None
    
    def select_child(self, c_puct: float = 1.0) -> Tuple[chess.Move, 'MCTSNode']:
        """Select the best child node using PUCT algorithm."""
        best_score = float('-inf')
        best_action = None
        best_child = None
        
        # Sum of all visit counts for normalization
        sum_visit_count = sum(child.visit_count for child in self.children.values())
        
        for action, child in self.children.items():
            # UCB score = Q + U
            # Q = current value estimate
            # U = C_puct * P * sqrt(sum_visits) / (1 + visits)
            q_value = -child.value()  # Negamax convention
            u_value = (c_puct * child.prior * np.sqrt(sum_visit_count) / 
                      (1 + child.visit_count))
            score = q_value + u_value
            
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
                
        logger.debug(f"Selected child: move={best_action}, score={best_score:.3f}, visits={best_child.visit_count}")
        return best_action, best_child

class MCTS:
    def __init__(self, network, num_simulations: int = 25, c_puct: float = 1.0):
        logger.info(f"Initializing MCTS with {num_simulations} simulations, c_puct={c_puct}")
        self.network = network  # Neural network (Athena)
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.root = None
        
    def run(self, board: chess.Board) -> Dict[chess.Move, float]:
        """Run MCTS simulations and return action probabilities."""
        logger.info(f"Starting MCTS search from position: {board.fen()}")
        self.root = MCTSNode(
            state=board.copy(),
            prior=0,
            turn=board.turn
        )
        
        for i in range(self.num_simulations):
            if i % 100 == 0:
                logger.info(f"Simulation {i}/{self.num_simulations}")
                
            node = self.root
            path = []
            
            # Selection
            while node.expanded():
                action, node = node.select_child(self.c_puct)
                path.append(node)
                
            # Expansion
            value = self._expand(node)
            
            # Backpropagation
            self._backpropagate(path, value)
            
        # Calculate action probabilities
        visit_counts = np.array([child.visit_count for child in self.root.children.values()])
        actions = list(self.root.children.keys())
        probs = visit_counts / np.sum(visit_counts)
        
        # Log the top 3 moves
        move_probs = list(zip(actions, probs))
        move_probs.sort(key=lambda x: x[1], reverse=True)
        for move, prob in move_probs[:3]:
            logger.info(f"Top move candidate: {move} with probability {prob:.3f}")
            
        return {action: prob for action, prob in zip(actions, probs)}
    
    def _expand(self, node: MCTSNode) -> float:
        """Expand node and return value estimate."""
        board = node.state
        
        # Get policy and value from neural network
        policy, value = self.network.predict(board)
        logger.debug(f"Network evaluation: value={value:.3f}")
        
        # Initialize children nodes
        node.children = {}
        for move in board.legal_moves:
            new_board = board.copy()
            new_board.push(move)
            
            child = MCTSNode(
                state=new_board,
                prior=policy[move],  # Get move probability from policy
                turn=not node.turn
            )
            node.children[move] = child
            
        return value
    
    def _backpropagate(self, path: List[MCTSNode], value: float):
        """Backpropagate value through the tree."""
        logger.debug(f"Backpropagating value: {value:.3f} through {len(path)} nodes")
        for node in reversed(path):
            node.value_sum += value
            node.visit_count += 1
            value = -value  # Negamax value
# ♟️ Kronos: Distributed Neural Chess Engine

**Kronos** is a modular, distributed chess engine built for experimentation in neural search, reinforcement learning, and resource-efficient AI.  
Designed to run across a Raspberry Pi cluster, Kronos splits chess intelligence into specialized "god modules," each mastering a phase of the game.

> Inspired by mythology — coordinated by Zeus, powered by Apollo, Athena, Ares, and Hades.

---

## 🏛️ System Architecture

| Module        | Role                                  | Description                                                                                            |
| :------------ | :------------------------------------ | :----------------------------------------------------------------------------------------------------- |
| ⚡ **Zeus**   | Orchestrator                          | Manages game state, coordinates module outputs, and selects final moves.                               |
| 📘 **Apollo** | Opening Book Module                   | Provides opening moves using a merged Polyglot book and move history analysis.                         |
| 🧠 **Athena** | Neural Evaluation Module              | Predicts best moves using AegisNet, a deep Residual CNN trained via supervised learning and self-play. |
| 🌲 **Ares**   | Monte Carlo Tree Search (MCTS) Module | Executes deep search when Athena signals criticality, using AegisNet evaluations at leaf nodes.        |
| 🧊 **Hades**  | Endgame Tablebase Module              | Ensures perfect play with Syzygy 3-4-5 WDL/DTZ probing for positions with ≤7 pieces.                   |

Modules communicate asynchronously via **ZeroMQ** for high scalability and low latency.

---

## 🧠 Athena: Scalable Neural Chess Engine

Athena serves as Kronos’s brain, following a **streamlined AlphaZero-style design**, but built around her own strategic weapon — **AegisNet**.

### 🚀 Core Components

-   **AegisNet**

    -   Deep Residual CNN with 19 blocks
    -   Dual heads:
        -   🎯 Policy Head: predicts move probabilities
        -   📈 Value Head: predicts game outcome (win/loss/draw)
    -   Input: `chess.Board` object (internally encoded into tensor with optional attack/defense maps)

-   **Ares (MCTS Search)**

    -   Monte Carlo Tree Search guided by AegisNet policy predictions
    -   PUCT-based move selection
    -   Focused search only when Athena detects critical positions

-   **Self-Play Trainer**

    -   Athena plays against herself to generate new training data
    -   Records policy distributions and final outcomes for continual learning

-   **Stockfish Trainer**

    -   Early-phase training supervised by Stockfish
    -   Bootstraps Athena's initial policy and value understanding

-   **Hybrid Trainer**

    -   Curriculum-based phased training:
        -   Start: 100% Stockfish
        -   Transition: 70% Stockfish / 30% Self-Play
        -   Advanced: 40% Stockfish / 60% Self-Play

-   **Evaluator (ELO Estimator)**
    -   Benchmarks Athena by playing against Stockfish (skill levels 0–20)
    -   Estimates ELO based on match outcomes

---

## 📦 Features

-   End-to-end self-play training loop
-   Modular architecture for easy experimentation
-   Real ELO benchmarking
-   Curriculum-driven training progression
-   Criticality-based MCTS rollouts
-   ONNX export and quantization ready
-   Pygame-based live game GUI

---

## 🎯 Current Roadmap

-   [x] Opening book integration (Apollo)
-   [x] Syzygy tablebase probing with cache (Hades)
-   [x] Zeus full game history and coordination
-   [x] Athena Stockfish bootstrapping phase
-   [x] MCTS integration between Athena and Ares
-   [x] Pygame GUI for playing against Athena
-   [ ] Full self-play reinforcement learning phase. (Aegis vs Stockfish)
-   [ ] Adaptive MCTS simulation budget (dynamic criticality scaling)
-   [ ] ONNX quantized model deployment for faster Pi inference
-   [ ] Move history PGN export from self-play games
-   [ ] Parallelize self-play and evaluation for faster training

---

## 👤 About

Kronos is built on the belief that intelligence doesn't require massive silicon — only better ideas.  
Designed to be lightweight, scalable, and efficient, Kronos brings strong, strategic chess play even to small systems and low-power devices.  
It explores how far careful engineering and smart self-learning can go without depending on heavyweight infrastructure.  
Chess remains the ultimate testbed for intelligence — and Kronos is a step toward making it accessible anywhere, on any machine.

Crafted with precision and intent by [@neavepaul](https://github.com/neavepaul).

---

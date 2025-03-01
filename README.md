# Kronos: Distributed Chess Engine

Welcome to **Kronos**, a modular and distributed chess engine designed to push the boundaries of resource-efficient chess computation. Built on a cluster of Raspberry Pis, Kronos utilizes parallelism, precomputed data, and innovative algorithms to provide strong chess-playing capabilities. This project is not just a chess engine but a platform to explore creative and resource-aware approaches to artificial intelligence.

## **Table of Contents**

-   [Introduction](#introduction)
-   [Architecture Overview](#architecture-overview)
-   [Modules and Roles](#modules-and-roles)
<!-- -   [Current Features](#current-features)
-   [Future Roadmap](#future-roadmap)
-   [Contributing](#contributing)
-   [License](#license) -->
-   [TODO](#todo)

---

## **Introduction**

Kronos, named after the Titan who consumed the gods, symbolizes its ability to encompass and integrate the specialized "god modules" of the chess engine. Designed to run on a cluster of Raspberry Pi 4Bs, Kronos leverages modularity and distributed computation to balance performance and resource constraints.

The engine combines traditional chess algorithms (e.g., Minimax, Alpha-Beta pruning) with modern innovations like lightweight neural networks and precomputed tablebases. Each Raspberry Pi in the cluster is assigned a specific role to handle different phases of the game, from openings to endgames.

---

## **Architecture Overview**

Kronos's design is inspired by microservices architecture, with each Raspberry Pi playing a specific role:

-   **Zeus (Orchestrator):** Manages communication between modules and combines results.
-   **Apollo (Opening Module):** Handles opening book queries and move history analysis.
-   **Athena (Evaluation Module):** Evaluates board positions using heuristics and lightweight neural networks.
-   **Ares (Search Module):** Performs search algorithms like Minimax and Alpha-Beta pruning.
-   **Hades (Tablebase Module):** Handles Syzygy tablebase probing for perfect endgame play.

Communication between modules is facilitated by ZeroMQ, enabling parallel processing and seamless coordination.

---

## **Modules and Roles**

1. **Zeus (Orchestrator):**

    - **Role:** Central controller of Kronos.
    - **Tasks:** Distributes tasks, combines results, and manages move generation. Maintains the complete game history for all modules to access.

2. **Apollo (Opening Module):**

    - **Role:** Provides opening move recommendations from a precompiled tablebase.
    - **Tasks:** Tracks move history during the opening phase and analyzes opponent strategies.

3. **Athena (Evaluation Module):**

    - **Role:** Evaluates board positions.
    - **Tasks:** Uses heuristics and lightweight neural networks to assess position strength, particularly during the midgame.

4. **Ares (Search Module):**

    - **Role:** Calculates the best moves using search algorithms.
    - **Tasks:** Implements Minimax and Alpha-Beta pruning, with adaptive depth for critical positions.

5. **Hades (Tablebase Module):**

    - **Role:** Ensures perfect play in the endgame phase.
    - **Tasks:** Queries Syzygy tablebases to determine optimal moves.

---

## **TODO**

-   [x] Set up Zeus (Orchestrator) and game history tracking
-   [x] Combine opening books for Apollo
-   [x] Finalize Apollo's integration for opening book queries
-   [x] Implement Athena (Evaluation Module)
-   [ ] Implement Ares (Search Module)
-   [ ] Implement Hades (Tablebase Module)
-   [ ] Download 345
-   [ ] Download select 6 and 7
-   [ ] cleanup some 345 based on 6 and 7
-   [ ] Train and deploy lightweight neural network for Athena
-   [ ] Actor critic for RL
-   [ ] Quantisation FP16 and ONNX
-   [ ] Set up caching for frequent tablebase queries in Hades
-   [ ] Stockfish 17 ARMv8 Dot Product at Zeus
-   [ ] Integrate ZeroMQ communication
-   [ ] Develop unit tests for all modules

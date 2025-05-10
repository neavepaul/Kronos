# 🧠 Athena / Prometheus Development Logbook

### 🕰️ Initial Architecture: Transformer + PFFTTU

-   [x] Designed an initial transformer-based model using:
    -   **PFFTTU encoding**: Piece, from-square, from-square-type, to-square, to-square-type, upgrade/promotion.
-   [x] Trained on **PGN data from grandmaster games**, including:
    -   Kasparov, Carlsen, Morphy, Fischer, Tal, etc.
-   [x] Encountered key limitations:
    -   Hard to enforce legality without additional logic.
    -   Data-to-reality mismatch (real games vs supervised data).
    -   Transformer struggled with generalizing structured move legality.
-   ❌ Decided to pivot toward AlphaZero-style architecture for more grounded legal move generation and learning.

---

### ✅ Architecture & Core Network

-   [x] Designed and implemented **AegisNet** (4096-policy AlphaZero-style dual-head network).
-   [x] Identified limitations in 4096 policy space (ambiguity, inefficiency).
-   [x] Designed **PrometheusNet** with:
    -   Factorized `from × to` policy output (64 × 64).
    -   Separate **promotion head** (`None`, Q, R, B, N).
    -   Stable `value` head with `tanh` output scaling.
-   [x] Integrated `MCTS` stub into AegisNet for future planning inference.

---

### ⚙️ Data Generation & Training Strategy

-   [x] Built **StockfishDualTrainer**:
    -   White = Lv5, Black = Lv0 (configurable).
    -   Evaluation from Stockfish with `tanh`-scaled CP score for `value`.
    -   Top-k move sampling via `multipv` → soft target label distribution.
    -   Label smoothing and strong weighting to top move.
-   [x] Enhanced training data to include:
    -   Soft policy target (`64x64`).
    -   Promotion label (one-hot of 5).
    -   Evaluation (`tanh(CP/400)`).

---

### 🏗 Training Framework

-   [x] Built `Trainer` class with:
    -   Curriculum support (phased hybrid training).
    -   Logging for policy/value/promotion losses.
    -   Save/load models & weights cleanly.
-   [x] Implemented fallback compile flow for PrometheusNet vs AegisNet.
-   [x] Added dummy input call to avoid “model not built” error before saving weights.

---

### 🧪 Training Observations

-   [x] Initial training losses:
    -   Policy ≈ 0.12–0.13 ✅
    -   Value ≈ 0.59–0.66 ✅
    -   Promotion ≈ 7.0–7.4 ❗️(as expected due to rarity)
-   [x] Validated stable learning curve across 50 iterations with `num_games=20–40`.

---

### 🧠 Insights & Optimization

-   [x] Identified **promotion loss imbalance** and mitigated with:
    -   Loss weight adjustment (`promotion: 0.2`).
-   [x] Skipped neutral positions when `|value| < 0.05` (optional toggle).
-   [x] Boosted learning signal with:
    -   Weighted top-move policy injection.
    -   Temperature-scaled softmax (`temp=150`).

---

### 🧩 Next Steps

-   [ ] Integrate PrometheusNet with **MCTS** rollout via `predict()` method.
-   [ ] Add game-level ELO evaluation via test matches.
-   [ ] Convert PrometheusNet to **ONNX** for Raspberry Pi deployment.
-   [ ] Add live GUI + visualization to play against Prometheus.

---

**Current Verdict:**  
Athena’s policy is sharp. Prometheus is trained and stable. Deployment and evaluation next.

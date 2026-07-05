# 2-Finger Gripper Reward Engineering Lab 🤖

A hands-on, offline-first reinforcement learning laboratory focused on designing, testing, and shaping reward functions for 2-finger robotic grippers (simulated in continuous 3D space via **MuJoCo** and **Gymnasium Robotics**).

---

## 📚 Where to Find the Lecture Slides

All 10 lectures from **David Silver's Reinforcement Learning Course** have been downloaded directly into this repository for offline study:
- **Location:** [documentation/david_silver_slides/](file:///Users/macpro/Documents/GitHub/silver-reward-lab/documentation/david_silver_slides)
- **Files Available:** `L1-intro_RL.pdf` through `L10-games.pdf`

---

## 🗺️ Detailed Roadmap: What Slides to Read & What Code to Try

To master creating reward functions for 2-finger gripper robots, follow this structured curriculum mapping David Silver's foundational theory directly to continuous robotic manipulation code.

### Module 1: The Reward Hypothesis & MDP Foundations
Understand the mathematical definition of rewards, returns, and Markov Decision Processes.

* **📖 What Slides to Read in [david_silver_slides/](file:///Users/macpro/Documents/GitHub/silver-reward-lab/documentation/david_silver_slides):**
  * **[L1-intro_RL.pdf](file:///Users/macpro/Documents/GitHub/silver-reward-lab/documentation/david_silver_slides/L1-intro_RL.pdf) (Slides 6–12):** *The Reward Hypothesis* — "All goals can be described by the maximization of expected cumulative reward."
  * **[L1-intro_RL.pdf](file:///Users/macpro/Documents/GitHub/silver-reward-lab/documentation/david_silver_slides/L1-intro_RL.pdf) (Slides 15–20):** *Sequential Decision Making, State, and Observations*.
  * **[L2-MDP.pdf](file:///Users/macpro/Documents/GitHub/silver-reward-lab/documentation/david_silver_slides/L2-MDP.pdf) (Slides 4–15):** *Markov Reward Processes (MRP)* — Notice how immediate rewards $R_t$ and discount factor $\gamma$ define the cumulative return $G_t = \sum_{k=0}^\infty \gamma^k R_{t+k+1}$.
  * **[L2-MDP.pdf](file:///Users/macpro/Documents/GitHub/silver-reward-lab/documentation/david_silver_slides/L2-MDP.pdf) (Slides 25–35):** *Markov Decision Processes (MDP)* and Bellman Expectation Equations.

* **💻 What to Code & Experiment With:**
  * Open **[main.py](file:///Users/macpro/Documents/GitHub/silver-reward-lab/main.py#L6-L38)** and study `DavidSilverGridWorld`. This is a ground-zero episodic MDP implementing a **time-penalty reward signal** ($R = -1.0$ per step until goal is reached).
  * **Experiment 1A:** In `step()`, change the reward from `-1.0` (time penalty) to `0.0` everywhere except `+10.0` at the goal state (sparse goal reward). Run `uv run python main.py`. Why does an agent exploring randomly struggle when there is no step penalty guiding efficiency?
  * **Experiment 1B:** Add a discount factor $\gamma = 0.9$ calculation inside `main()` to compute the discounted return $G_0$ across the episode.

---

### Module 2: The Curse of Sparse Rewards in Continuous Manipulation
Learn why tabular RL methods fail in 3D robotics and why binary rewards give zero learning gradient.

* **📖 What Slides to Read in [david_silver_slides/](file:///Users/macpro/Documents/GitHub/silver-reward-lab/documentation/david_silver_slides):**
  * **[L4-model-free-prediction.pdf](file:///Users/macpro/Documents/GitHub/silver-reward-lab/documentation/david_silver_slides/L4-model-free-prediction.pdf) (Slides 5–18):** *Monte Carlo vs. Temporal Difference (TD) Learning*. Understand how TD updates value estimates based on immediate reward $R_{t+1}$ plus discounted future value $\gamma V(S_{t+1})$.
  * **Why Continuous Control is Different:** In grid worlds (Lecture 2 & 4), an agent can stumble into every state by chance. In a 3D MuJoCo simulation of a 2-finger gripper, the probability of randomly closing both fingers precisely around a 4cm block is virtually zero!

* **💻 What to Code & Experiment With:**
  * Open **[gripper_reward_lab.py](file:///Users/macpro/Documents/GitHub/silver-reward-lab/gripper_reward_lab.py#L25-L30)** and study `GripperRewardShaper.sparse_reward()`. Notice how it returns `0.0` only when the object is within 5cm of the target, and `-1.0` otherwise.
  * **Experiment 2:** Run `uv run python gripper_reward_lab.py` and inspect the printed telemetry table. Notice how across all 50 steps of the approach trajectory, `Sparse R` remains completely flat at `-1.00`. In policy gradient algorithms, a flat reward curve means **zero gradient ($\nabla_\theta J(\theta) = 0$)** — the robot learns nothing!

---

### Module 3: Dense Reaching & Distance Metrics
Design continuous reward gradients that guide end-effector fingertips toward target objects.

* **📖 What Slides to Read in [david_silver_slides/](file:///Users/macpro/Documents/GitHub/silver-reward-lab/documentation/david_silver_slides):**
  * **[L6-value-function-approximation.pdf](file:///Users/macpro/Documents/GitHub/silver-reward-lab/documentation/david_silver_slides/L6-value-function-approximation.pdf) (Slides 5–15):** *Value Function Approximation*. Learn why we must approximate value functions $V_\theta(s)$ over continuous state feature vectors (3D coordinates, joint angles, velocities).

* **💻 What to Code & Experiment With:**
  * In **[gripper_reward_lab.py](file:///Users/macpro/Documents/GitHub/silver-reward-lab/gripper_reward_lab.py#L32-L39)**, study `GripperRewardShaper.dense_distance_reward()`.
  * **Experiment 3A (Distance Norms):** Currently, reaching uses Euclidean distance ($L_2$ norm: $-\|\mathbf{p}_{\text{grip}} - \mathbf{p}_{\text{obj}}\|_2$). Modify this function to test:
    * **Manhattan Distance ($L_1$ norm):** `-np.sum(np.abs(grip_pos - object_pos))`
    * **Squared Euclidean Distance:** `-np.sum(np.square(grip_pos - object_pos))`
  * **Experiment 3B:** Re-run `uv run python gripper_reward_lab.py` and open `reward_comparison.png`. Notice how squared Euclidean distance penalizes far distances heavily, but provides extremely weak gradients when the gripper gets close to the object!

---

### Module 4: Multi-Stage Contact, Lifting & Action Regularization
Build state-of-the-art shaped reward functions for complex multi-phase manipulation tasks.

* **📖 What Slides to Read in [david_silver_slides/](file:///Users/macpro/Documents/GitHub/silver-reward-lab/documentation/david_silver_slides):**
  * **[L7-policy-gradient-methods.pdf](file:///Users/macpro/Documents/GitHub/silver-reward-lab/documentation/david_silver_slides/L7-policy-gradient-methods.pdf) (Slides 4–16):** *Policy Gradients & The Policy Gradient Theorem*. Understand how policy parameters $\theta$ are updated directly in the direction of higher reward: $\nabla_\theta J(\theta) = \mathbb{E}[\nabla_\theta \log \pi_\theta(s, a) Q^\pi(s, a)]$.
  * **[L7-policy-gradient-methods.pdf](file:///Users/macpro/Documents/GitHub/silver-reward-lab/documentation/david_silver_slides/L7-policy-gradient-methods.pdf) (Slides 18–26):** *Actor-Critic Architecture*. This is the theoretical architecture behind modern robotics algorithms like **PPO (Proximal Policy Optimization)** and **SAC (Soft Actor-Critic)**.

* **💻 What to Code & Experiment With:**
  * In **[gripper_reward_lab.py](file:///Users/macpro/Documents/GitHub/silver-reward-lab/gripper_reward_lab.py#L41-L85)**, study `GripperRewardShaper.multi_stage_shaped_reward()`. A robust 2-finger gripper reward function balances four critical forces:
    1. **Reaching Term ($r_{\text{reach}}$):** Pulls fingertips toward object coordinates.
    2. **Grasping Contact Term ($r_{\text{grasp}}$):** Checks if fingertips are close to the object (`dist < 0.05`) **and** gripper opening width matches object thickness (`finger_width < 0.04`).
    3. **Lifting Term ($r_{\text{lift}}$):** Activated *only after grasping*, rewarding vertical displacement toward the target height.
    4. **Control Regularization ($r_{\text{ctrl}}$):** Penalizes motor torque/acceleration ($-\|\mathbf{u}\|^2$). Without this penalty, policy gradient algorithms exploit the physics engine, causing violent jittering or flinging!
  * **Experiment 4 (Weight Tuning Lab):**
    * In `multi_stage_shaped_reward()`, change the default weights dictionary: try setting `"ctrl": 0.5` (heavy motor penalty) vs. `"ctrl": 0.0` (zero motor penalty).
    * Run `uv run python gripper_reward_lab.py` and observe how excessive control penalties can dominate the reaching reward, causing the robot to freeze and do nothing (the notorious "lazy robot" local minimum)!

---

## 🚀 How to Run the Scripts (100% Offline Ready)

To execute the simulations and generate visual reward charts anytime without internet:

```bash
# Run the 2-Finger Gripper MuJoCo simulation & generate reward_comparison.png
uv run python gripper_reward_lab.py

# Run the foundational 1D grid world demo
uv run python main.py
```

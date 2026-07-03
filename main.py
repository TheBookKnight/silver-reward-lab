"""
main.py

Foundational 1D Grid World demonstrating David Silver's Lecture 2 Reward Hypothesis.

================================================================================
HOW TO RUN THIS SCRIPT (100% OFFLINE READY):
--------------------------------------------------------------------------------
From your terminal inside the project directory, run:
    $ uv run python main.py

WHAT THIS SCRIPT DOES WHEN RUN:
1. Initializes a 1D grid world environment (DavidSilverGridWorld).
2. Simulates an agent moving step-by-step toward a goal state.
3. Prints the state transitions and step rewards (-1.0 per step time penalty).
================================================================================
"""

import numpy as np
import gymnasium as gym


class DavidSilverGridWorld(gym.Env):
    """
    A foundational Ground-Zero environment reflecting David Silver's L1/L2 formulation.
    The agent must learn to exit a 1D grid by choosing actions.
    """

    def __init__(self):
        super().__init__()
        self.state = 0           # Start at index 0
        self.goal_state = 4      # Goal is index 4

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = 0
        return self.state, {}

    def step(self, action):
        # Action space: 0 = Stay, 1 = Move Right
        if action == 1:
            self.state = min(self.state + 1, self.goal_state)

        # --- DAVID SILVER REWARD FUNCTION LOGIC (Lecture 2) ---
        # Reward Hypothesis: All goals can be described by maximizing cumulative reward.
        if self.state == self.goal_state:
            reward = 0.0      # Episodic terminal goal reached
            terminated = True
        else:
            # Time penalty reward signal (encourages efficiency)
            reward = -1.0
            terminated = False

        return self.state, reward, terminated, False, {}


def main():
    print("Initializing Offline David Silver Lab...")
    env = DavidSilverGridWorld()
    state, _ = env.reset()

    # Simulate a sample episode completely offline
    done = False
    total_reward = 0
    step_count = 0

    while not done:
        # Greedily move right (action=1) to illustrate transitions
        action = 1
        state, reward, done, _, _ = env.step(action)
        total_reward += reward
        step_count += 1
        print(
            f"Step {step_count} -> Moved to State: {state} | Step Reward: {reward}")

    print(f"Episode complete! Total Reward accumulated: {total_reward}")


if __name__ == "__main__":
    main()

"""
gripper_reward_lab.py

An offline MVP Reward Engineering Lab for 2-Finger Gripper Robots.
Uses MuJoCo and Gymnasium Robotics (FetchPickAndPlace-v4 / 2-Finger Parallel Gripper).

================================================================================
HOW TO RUN THIS SCRIPT (100% OFFLINE READY):
--------------------------------------------------------------------------------
From your terminal inside the project directory, run:
    $ uv run python gripper_reward_lab.py

WHAT THIS SCRIPT DOES WHEN RUN:
1. Initializes a local 3D MuJoCo simulation of a 2-finger parallel gripper.
2. Simulates a 50-step trajectory approaching and grasping an object.
3. Prints step-by-step telemetry comparing Sparse vs. Shaped rewards.
4. Generates and saves a visual plot comparing reward curves to:
       -> reward_comparison.png

HOW TO EXPERIMENT (FOR YOU & YOUR TEAMMATE):
- Modify the reward weights dictionary in `GripperRewardShaper.multi_stage_shaped_reward`.
- Re-run `uv run python gripper_reward_lab.py` and inspect `reward_comparison.png`
  to observe how shaping impacts gradient smoothness across the trajectory.
================================================================================
"""

import numpy as np
import gymnasium as gym
import gymnasium_robotics
import matplotlib.pyplot as plt


class GripperRewardShaper:
    """
    A collection of reward functions designed for a 2-finger gripper robot.
    In continuous control, reward engineering is critical: sparse rewards provide zero
    gradient for learning until an accidental success occurs.
    """

    @staticmethod
    def sparse_reward(achieved_goal, desired_goal, info=None, distance_threshold=0.05):
        """
        David Silver L2 Ground-Zero formulation: Binary Sparse Reward.
        Returns 0.0 if the object is within the distance threshold of the target goal, else -1.0.
        """
        dist = np.linalg.norm(achieved_goal - desired_goal)
        return 0.0 if dist < distance_threshold else -1.0

    @staticmethod
    def dense_distance_reward(grip_pos, object_pos):
        """
        Phase 1 Dense Reward: Pure reaching distance.
        Provides a continuous gradient encouraging the 2-finger gripper to approach the object.
        """
        dist_grip_obj = np.linalg.norm(grip_pos - object_pos)
        return -dist_grip_obj

    @staticmethod
    def multi_stage_shaped_reward(grip_pos, object_pos, target_pos, gripper_state, action,
                                  weights={"reach": 1.0, "lift": 2.0, "ctrl": 0.05, "bonus": 10.0}):
        """
        State-of-the-art Multi-Stage Shaped Reward for 2-Finger Gripper Manipulation:
        1. Reaching: Distance between gripper fingertips and object.
        2. Grasping/Lifting: Once close to the object, reward moving object upward/to target.
        3. Regularization: Penalize excessive motor control action magnitudes (prevents jitter).
        4. Success Bonus: Large reward upon task completion.
        """
        dist_grip_obj = np.linalg.norm(grip_pos - object_pos)
        dist_obj_target = np.linalg.norm(object_pos - target_pos)

        # 1. Reaching reward
        r_reach = -dist_grip_obj

        # 2. Lifting & Placement reward (activated when gripper is close to object)
        is_close = dist_grip_obj < 0.05
        r_lift = -dist_obj_target if is_close else 0.0

        # 3. Grasp contact indicator (when fingers close around object width ~0.02)
        finger_width = np.sum(gripper_state)
        r_grasp = 0.2 if (is_close and finger_width < 0.04) else 0.0

        # 4. Action energy penalty (L2 norm of action)
        r_ctrl = -np.sum(np.square(action))

        # 5. Success bonus
        is_success = dist_obj_target < 0.05
        r_bonus = weights["bonus"] if is_success else 0.0

        total_reward = (
            weights["reach"] * r_reach
            + weights["lift"] * r_lift
            + r_grasp
            + weights["ctrl"] * r_ctrl
            + r_bonus
        )
        return total_reward, {
            "r_reach": r_reach,
            "r_lift": r_lift,
            "r_grasp": r_grasp,
            "r_ctrl": r_ctrl,
            "total": total_reward,
        }


def run_lab_simulation(num_steps=50):
    """
    Simulates a 2-finger gripper trajectory and records telemetry for comparison.
    Runs 100% offline using MuJoCo local physics computation.
    """
    print("Initializing 2-Finger Gripper MuJoCo Environment (FetchPickAndPlace-v4)...")
    env = gym.make("FetchPickAndPlace-v4")
    obs, _ = env.reset(seed=42)

    history = {
        "step": [],
        "sparse": [],
        "dense": [],
        "shaped": [],
        "dist_grip_obj": [],
        "dist_obj_target": [],
    }

    print("\n--- Starting Offline Gripper Simulation Trajectory ---")
    print(f"{'Step':<6} | {'Grip-Obj Dist':<14} | {'Obj-Target Dist':<16} | {'Sparse R':<10} | {'Shaped R':<10}")
    print("-" * 65)

    for step in range(num_steps):
        # Generate an action: move gripper towards object (Heuristic controller for demo)
        grip_pos = obs["observation"][:3]
        object_pos = obs["observation"][3:6]
        target_pos = obs["desired_goal"]
        gripper_state = obs["observation"][9:11]

        delta_pos = object_pos - grip_pos
        # Scale movement towards object with slight gripper closing
        action = np.zeros(4)
        action[:3] = np.clip(delta_pos * 5.0, -1.0, 1.0)
        action[3] = -0.05  # Slightly close fingers

        # Compute rewards before step
        r_sparse = GripperRewardShaper.sparse_reward(object_pos, target_pos)
        r_dense = GripperRewardShaper.dense_distance_reward(grip_pos, object_pos)
        r_shaped, components = GripperRewardShaper.multi_stage_shaped_reward(
            grip_pos, object_pos, target_pos, gripper_state, action
        )

        # Record metrics
        history["step"].append(step)
        history["sparse"].append(r_sparse)
        history["dense"].append(r_dense)
        history["shaped"].append(r_shaped)
        history["dist_grip_obj"].append(np.linalg.norm(grip_pos - object_pos))
        history["dist_obj_target"].append(np.linalg.norm(object_pos - target_pos))

        if step % 10 == 0 or step == num_steps - 1:
            print(f"{step:<6} | {history['dist_grip_obj'][-1]:<14.4f} | {history['dist_obj_target'][-1]:<16.4f} | {r_sparse:<10.2f} | {r_shaped:<10.4f}")

        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

    env.close()

    # Generate visual comparison plot
    plot_path = "reward_comparison.png"
    plt.figure(figsize=(10, 6))
    plt.plot(history["step"], history["sparse"], label="Sparse Reward (Binary)", linestyle="--", alpha=0.7)
    plt.plot(history["step"], history["dense"], label="Dense Reaching Reward", linewidth=2)
    plt.plot(history["step"], history["shaped"], label="Multi-Stage Shaped Reward", linewidth=2.5)
    plt.title("2-Finger Gripper Reward Signal Comparison Across Approach Trajectory")
    plt.xlabel("Simulation Step")
    plt.ylabel("Reward Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"\nSimulation Complete! Saved reward comparison chart to '{plot_path}'.")
    return history


if __name__ == "__main__":
    run_lab_simulation()

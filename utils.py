import gymnasium as gym
import numpy as np
import pygame
import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches


def visualize_policy(policy_fn, env_id="CliffWalking-v1", title="RL Agent", fps=10, max_episode_steps=1000):
    """Visualizes an agent playing in a Gymnasium environment using a custom action policy function.

    Args:
        policy_fn (function): A function that takes `state` as input and
          returns (int): Action to be taken.

        env_id (str): The Gymnasium environment ID.
        title (str): Custom title to display on the window frame and terminal logs.
        fps (int): Frames per second to throttle the visualization.
    """
    pygame.init()

    env = gym.make(env_id, render_mode="rgb_array", max_episode_steps=max_episode_steps)
    clock = pygame.time.Clock()
    state, info = env.reset()

    # Dynamic window resizing based on initial frame dimensions
    initial_frame = env.render()
    screen = pygame.display.set_mode(
        (initial_frame.shape[1], initial_frame.shape[0])
    )

    action_binds = {0: "Up", 1: "Right", 2: "Down", 3: "Left"}

    terminated = False
    truncated = False
    total_reward = 0
    step_count = 0
    
    # Track physical wall-clock time
    start_time = time.time()

    print(f"\n==========================================")
    print(f" Starting Evaluation: {title}")
    print(f"==========================================")

    while not (terminated or truncated):
        # Update the window frame to show active step tracking
        pygame.display.set_caption(f"{title} | Step: {step_count} | Env: {env_id}")

        # 1. Evaluate the custom policy function; Only perform the best action selection.
        action = policy_fn(state)
        
        # 2. Step the environment
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1

        print(
            f"Step={step_count:3d} | State={state:2d} | Reward={reward:4d} | "
            f"Total Reward={total_reward:4d} | Action={action_binds.get(action, action)}"
        )

        # 3. Render map transitions to Pygame surface
        frame = env.render()
        if frame is not None:
            surface = pygame.image.frombuffer(
                frame.tobytes(), frame.shape[1::-1], "RGB"
            )
            screen.blit(surface, (0, 0))
            pygame.display.flip()

        clock.tick(fps)

        if pygame.event.peek(pygame.QUIT):
            print("Visualization interrupted by user.")
            break

    # Calculate final duration
    elapsed_time = time.time() - start_time

    env.close()
    pygame.quit()
    print(f"------------------------------------------")
    print(f"Simulation ended.")
    print(f"  - Total Episode Steps: {step_count}")
    print(f"  - Total Episode Time:     {elapsed_time:.2f} seconds")
    print(f"  - Final Total Reward:  {total_reward}")



def plot_learning_diagnostics(
    episode_avg_mc_error, episode_rewards, episode_lengths, episode_durations
):
    """Generates a 2x2 dashboard of training metrics to evaluate RL agent performance and convergence.

    Args:
        episode_avg_mc_error (list/array): Average MC error per episode.
        episode_rewards (list/array): Total reward accumulated per episode.
        episode_lengths (list/array): Total environment steps taken per episode.
        episode_durations (list/array): Wall-clock time (seconds) taken per
          episode.
    """
    episodes = np.arange(1, len(episode_rewards) + 1)

    # Set up a clean, high-resolution 2x2 grid plot
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("RL Agent Learning Diagnostics & Convergence", fontsize=16, weight="bold")

    # --- 1. MC Error (Convergence) ---
    axs[0, 0].plot(episodes, episode_avg_mc_error, color="tab:purple", alpha=0.6, label="Raw")
    # Add a moving average to smooth out the noise inherent to MC sampling
    if len(episode_avg_mc_error) >= 10:
        smoothed_mc = np.convolve(episode_avg_mc_error, np.ones(10)/10, mode='valid')
        axs[0, 0].plot(episodes[9:], smoothed_mc, color="purple", linewidth=2, label="10-Ep MA")
    axs[0, 0].set_title("Average MC Error Function Value")
    axs[0, 0].set_ylabel("Mean |G_t - Q(s,a)|")
    axs[0, 0].grid(True, linestyle="--", alpha=0.6)
    axs[0, 0].legend()

    # --- 2. Episode Rewards (Performance) ---
    axs[0, 1].plot(episodes, episode_rewards, color="tab:green", alpha=0.6, label="Raw")
    if len(episode_rewards) >= 10:
        smoothed_rewards = np.convolve(episode_rewards, np.ones(10)/10, mode='valid')
        axs[0, 1].plot(episodes[9:], smoothed_rewards, color="darkgreen", linewidth=2, label="10-Ep MA")
    axs[0, 1].set_title("Total Cumulative Reward per Episode")
    axs[0, 1].set_ylabel("Total Reward")
    axs[0, 1].grid(True, linestyle="--", alpha=0.6)
    axs[0, 1].legend()

    # --- 3. Episode Steps (Path Efficiency) ---
    axs[1, 0].plot(episodes, episode_lengths, color="tab:blue", alpha=0.6, label="Raw")
    if len(episode_lengths) >= 10:
        smoothed_lengths = np.convolve(episode_lengths, np.ones(10)/10, mode='valid')
        axs[1, 0].plot(episodes[9:], smoothed_lengths, color="darkblue", linewidth=2, label="10-Ep MA")
    axs[1, 0].set_title("Episode Length (Steps to Goal/Termination)")
    axs[1, 0].set_xlabel("Episodes")
    axs[1, 0].set_ylabel("Steps Taken")
    axs[1, 0].grid(True, linestyle="--", alpha=0.6)
    axs[1, 0].legend()

    # --- 4. Episode Durations (Computational Overhead) ---
    axs[1, 1].plot(episodes, episode_durations, color="tab:orange", alpha=0.6, label="Raw")
    if len(episode_durations) >= 10:
        smoothed_durations = np.convolve(episode_durations, np.ones(10)/10, mode='valid')
        axs[1, 1].plot(episodes[9:], smoothed_durations, color="darkorange", linewidth=2, label="10-Ep MA")
    axs[1, 1].set_title("Wall-Clock Episode Duration")
    axs[1, 1].set_xlabel("Episodes")
    axs[1, 1].set_ylabel("Time (Seconds)")
    axs[1, 1].grid(True, linestyle="--", alpha=0.6)
    axs[1, 1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_graphs(errors: np.array, episode_rewards: np.array, episode_lengths: np.array, episode_durations: np.array, error_title = "Avg. Update Error", reward_title = "Total Episode Rewards", length_title = "Episode Lengths (Steps)", duration_title = "Episode Durations (Seconds)", figsize=(14, 10), title="RL Agent Learning Diagnostics & Convergence", error_ylabel="Mean |G_t - Q(s,a)|", reward_ylabel="Total Reward", length_ylabel="Steps Taken", duration_ylabel="Time (Seconds)", convergence_threshold=0.001):
    """Plots the learning diagnostics for an RL agent.

    Args:
        errors (np.array): Array of average MC errors per episode.
        episode_rewards (np.array): Array of total rewards per episode.
        episode_lengths (np.array): Array of total steps taken per episode.
        episode_durations (np.array): Array of wall-clock durations per episode.
    """


    episodes = np.arange(1, len(episode_rewards) + 1)    

    # Set up a clean, high-resolution 2x2 grid plot
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16, weight="bold")

    # --- 1. Error (Convergence) ---
    axs[0, 0].plot(episodes, errors, color="tab:purple", alpha=0.6, label="Raw")
    if len(errors) >= 10:
        smoothed_errors = np.convolve(errors, np.ones(10)/10, mode='valid')
        axs[0, 0].plot(episodes[9:], smoothed_errors, color="purple", linewidth=2, label="10-Ep MA")
    axs[0, 0].set_title(error_title)
    axs[0, 0].set_ylabel(error_ylabel)
    axs[0, 0].grid(True, linestyle="--", alpha=0.6)
    axs[0, 0].legend()

    # --- 2. Episode Rewards (Performance) ---
    axs[0, 1].plot(episodes, episode_rewards, color="tab:green", alpha=0.6, label="Raw")
    if len(episode_rewards) >= 10:
        smoothed_rewards = np.convolve(episode_rewards, np.ones(10)/10, mode='valid')
        axs[0, 1].plot(episodes[9:], smoothed_rewards, color="darkgreen", linewidth=2, label="10-Ep MA")
    axs[0, 1].set_title(reward_title)
    axs[0, 1].set_ylabel(reward_ylabel)
    axs[0, 1].grid(True, linestyle="--", alpha=0.6)
    axs[0, 1].legend()

    # --- 3. Episode Steps (Path Efficiency) ---
    axs[1, 0].plot(episodes, episode_lengths, color="tab:blue", alpha=0.6, label="Raw")
    if len(episode_lengths) >= 10:
        smoothed_lengths = np.convolve(episode_lengths, np.ones(10)/10, mode='valid')
        axs[1, 0].plot(episodes[9:], smoothed_lengths, color="darkblue", linewidth=2, label="10-Ep MA")
    axs[1, 0].set_title(length_title)
    axs[1, 0].set_xlabel("Episodes")
    axs[1, 0].set_ylabel(length_ylabel)
    axs[1, 0].grid(True, linestyle="--", alpha=0.6)
    axs[1, 0].legend()

    # --- 4. Episode Durations (Computational Overhead) ---
    axs[1, 1].plot(episodes, episode_durations, color="tab:orange", alpha=0.6, label="Raw")
    if len(episode_durations) >= 10:
        smoothed_durations = np.convolve(episode_durations, np.ones(10)/10, mode='valid')
        axs[1, 1].plot(episodes[9:], smoothed_durations, color="darkorange", linewidth=2, label="10-Ep MA")
    axs[1, 1].set_title(duration_title)
    axs[1, 1].set_xlabel("Episodes")
    axs[1, 1].set_ylabel(duration_ylabel)
    axs[1, 1].grid(True, linestyle="--", alpha=0.6)
    axs[1, 1].legend()



    # Show Summary Statistics in the Console

    convergence_episode = next((i+1 for i, e in enumerate(errors) if e < convergence_threshold), None)

    print(f"\n{'='*50}")
    print(f"Agent Learning Summary:")
    print(f"  - Final {error_title}: {errors[-1]:.4f}")
    print(f"  - Total Episodes: {len(episode_rewards)}")
    print(f"  - Best Episode Reward: {np.max(episode_rewards):,}")
    print(f"  - Worst Episode Reward: {np.min(episode_rewards):,}")
    print(f"  - Average Episode Length: {np.mean(episode_lengths):.2f}")
    print(f"  - Average Episode Duration: {np.mean(episode_durations):.2f}")
    print(f"  - Convergence Status: {f'Converged in {convergence_episode}' if convergence_episode else 'Not Converged'} (Threshold: {convergence_threshold})")
    print(f"{'='*50}")



import matplotlib.pyplot as plt
import numpy as np


def plot_smooth_graphs(
    errors: np.array,
    episode_rewards: np.array,
    episode_lengths: np.array,
    episode_durations: np.array,
    error_title="Avg. Update Error",
    reward_title="Total Episode Rewards",
    length_title="Episode Lengths (Steps)",
    duration_title="Episode Durations (Seconds)",
    figsize=(14, 10),
    title="RL Agent Learning Diagnostics & Convergence",
    error_ylabel="Mean |G_t - Q(s,a)|",
    reward_ylabel="Total Reward",
    length_ylabel="Steps Taken",
    duration_ylabel="Time (Seconds)",
    convergence_threshold=0.001,
):
    """Plots the learning diagnostics for an RL agent with smoothed curves and outlier truncation,

    overlaying BOTH true historical minimums and maximums inside text labels.
    """

    episodes = np.arange(1, len(episode_rewards) + 1)

    fig, axs = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16, weight="bold")

    def get_ma(data, window=10):
        if len(data) >= window:
            return np.convolve(data, np.ones(window) / window, mode="valid")
        return None

    # --- 1. Error (Convergence) ---
    axs[0, 0].plot(episodes, errors, color="tab:purple", alpha=0.3, label="Raw")
    ma_errors = get_ma(errors)
    if ma_errors is not None:
        axs[0, 0].plot(
            episodes[9:], ma_errors, color="purple", linewidth=2, label="10-Ep MA"
        )
    y_max_error = np.percentile(errors, 95)
    axs[0, 0].set_ylim(bottom=0, top=y_max_error)
    axs[0, 0].set_title(error_title)
    axs[0, 0].set_ylabel(error_ylabel)
    axs[0, 0].grid(True, linestyle="--", alpha=0.5)
    axs[0, 0].legend()

    # Extremes Text Box
    axs[0, 0].text(
        0.05,
        0.82,
        f"True Max: {np.max(errors):.4f}\nTrue Min: {np.min(errors):.4f}",
        transform=axs[0, 0].transAxes,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="purple", boxstyle="round,pad=0.4"),
    )

    # --- 2. Episode Rewards (Performance) ---
    axs[0, 1].plot(
        episodes, episode_rewards, color="tab:green", alpha=0.3, label="Raw"
    )
    ma_rewards = get_ma(episode_rewards)
    if ma_rewards is not None:
        axs[0, 1].plot(
            episodes[9:], ma_rewards, color="darkgreen", linewidth=2, label="10-Ep MA"
        )
    y_min_reward = np.percentile(episode_rewards, 5)
    axs[0, 1].set_ylim(bottom=y_min_reward, top=0)
    axs[0, 1].set_title(reward_title)
    axs[0, 1].set_ylabel(reward_ylabel)
    axs[0, 1].grid(True, linestyle="--", alpha=0.5)
    axs[0, 1].legend()

    # Extremes Text Box (Placed at bottom-left so it doesn't cover the main curve baseline near 0)
    axs[0, 1].text(
        0.05,
        0.08,
        f"True Max: {np.max(episode_rewards):,}\nTrue Min: {np.min(episode_rewards):,}",
        transform=axs[0, 1].transAxes,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="darkgreen", boxstyle="round,pad=0.4"),
    )

    # --- 3. Episode Steps (Path Efficiency) ---
    axs[1, 0].plot(
        episodes, episode_lengths, color="tab:blue", alpha=0.3, label="Raw"
    )
    ma_lengths = get_ma(episode_lengths)
    if ma_lengths is not None:
        axs[1, 0].plot(
            episodes[9:], ma_lengths, color="darkblue", linewidth=2, label="10-Ep MA"
        )
    y_max_length = np.percentile(episode_lengths, 95)
    axs[1, 0].set_ylim(bottom=0, top=y_max_length)
    axs[1, 0].set_title(length_title)
    axs[1, 0].set_xlabel("Episodes")
    axs[1, 0].set_ylabel(length_ylabel)
    axs[1, 0].grid(True, linestyle="--", alpha=0.5)
    axs[1, 0].legend()

    # Extremes Text Box (Placed right side since steps flatten out toward the bottom right)
    axs[1, 0].text(
        0.55,
        0.82,
        f"True Max: {np.max(episode_lengths):,}\nTrue Min: {np.min(episode_lengths):,}",
        transform=axs[1, 0].transAxes,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="darkblue", boxstyle="round,pad=0.4"),
    )

    # --- 4. Episode Durations (Computational Overhead) ---
    axs[1, 1].plot(
        episodes, episode_durations, color="tab:orange", alpha=0.3, label="Raw"
    )
    ma_durations = get_ma(episode_durations)
    if ma_durations is not None:
        axs[1, 1].plot(
            episodes[9:], ma_durations, color="darkorange", linewidth=2, label="10-Ep MA"
        )
    y_max_duration = np.percentile(episode_durations, 95)
    axs[1, 1].set_ylim(bottom=0, top=y_max_duration)
    axs[1, 1].set_title(duration_title)
    axs[1, 1].set_xlabel("Episodes")
    axs[1, 1].set_ylabel(duration_ylabel)
    axs[1, 1].grid(True, linestyle="--", alpha=0.5)
    axs[1, 1].legend()

    # Extremes Text Box
    axs[1, 1].text(
        0.55,
        0.82,
        f"True Max: {np.max(episode_durations):.2f}s\nTrue Min: {np.min(episode_durations):.4f}s",
        transform=axs[1, 1].transAxes,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="darkorange", boxstyle="round,pad=0.4"),
    )

    plt.tight_layout()

    # Show Summary Statistics in the Console
    convergence_episode = next(
        (i + 1 for i, e in enumerate(errors) if e < convergence_threshold), None
    )
    print(f"\n{'='*50}")
    print(f"Agent Learning Summary:")
    print(f"  - Final {error_title}: {errors[-1]:.4f}")
    print(f"  - Total Episodes: {len(episode_rewards)}")
    print(f"  - Best Episode Reward: {np.max(episode_rewards):,}")
    print(f"  - Worst Episode Reward: {np.min(episode_rewards):,}")
    print(f"  - Average Episode Length: {np.mean(episode_lengths):.2f}")
    print(f"  - Average Episode Duration: {np.mean(episode_durations):.2f}")
    print(
        f"  - Convergence Status: {f'Converged in {convergence_episode}' if convergence_episode else 'Not Converged'} (Threshold: {convergence_threshold})"
    )
    print(f"{'='*50}")

    plt.show()



def plot_smoothed_diagnostics(
    episode_avg_mc_error, episode_rewards, episode_lengths, episode_durations
):
    """Generates a 2x2 dashboard of training metrics, automatically cropping extreme

    outliers to expose micro-trends while preserving and displaying historical
    extremes.
    """
    episodes = np.arange(1, len(episode_rewards) + 1)

    # Set up a high-resolution 2x2 grid
    fig, axs = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle(
        "RL Agent Learning Diagnostics (Outliers Truncated for Visual Clarity)",
        fontsize=16,
        weight="bold",
    )

    # Helper for moving average
    def get_ma(data, window=10):
        if len(data) >= window:
            return np.convolve(data, np.ones(window) / window, mode="valid")
        return None

    # -------------------------------------------------------------------------
    # 1. MC ERROR PLOT (Convergence)
    # -------------------------------------------------------------------------
    axs[0, 0].plot(
        episodes, episode_avg_mc_error, color="tab:purple", alpha=0.3, label="Raw"
    )
    ma_mc = get_ma(episode_avg_mc_error)
    if ma_mc is not None:
        axs[0, 0].plot(
            episodes[9:], ma_mc, color="purple", linewidth=2, label="10-Ep MA"
        )

    # Crop out top 5% highest error spikes
    y_max_mc = np.percentile(episode_avg_mc_error, 95)
    axs[0, 0].set_ylim(bottom=0, top=y_max_mc)
    axs[0, 0].set_title("Average MC Error Function Value")
    axs[0, 0].set_ylabel("Mean |G_t - Q(s,a)|")
    axs[0, 0].grid(True, linestyle="--", alpha=0.5)
    axs[0, 0].legend()

    # Outlier Label
    axs[0, 0].text(
        0.05,
        0.90,
        f"True Max Error: {np.max(episode_avg_mc_error):.2f}",
        transform=axs[0, 0].transAxes,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="purple"),
    )

    # -------------------------------------------------------------------------
    # 2. TOTAL REWARDS PLOT (Performance)
    # -------------------------------------------------------------------------
    axs[0, 1].plot(
        episodes, episode_rewards, color="tab:green", alpha=0.3, label="Raw"
    )
    ma_rew = get_ma(episode_rewards)
    if ma_rew is not None:
        axs[0, 1].plot(
            episodes[9:], ma_rew, color="darkgreen", linewidth=2, label="10-Ep MA"
        )

    # Crop out bottom 5% lowest catastrophic disaster rewards
    y_min_rew = np.percentile(episode_rewards, 5)
    axs[0, 1].set_ylim(bottom=y_min_rew, top=0)
    axs[0, 1].set_title("Total Cumulative Reward per Episode")
    axs[0, 1].set_ylabel("Total Reward")
    axs[0, 1].grid(True, linestyle="--", alpha=0.5)
    axs[0, 1].legend()

    # Outlier Label
    axs[0, 1].text(
        0.05,
        0.05,
        f"True Min Reward: {np.min(episode_rewards):,}",
        transform=axs[0, 1].transAxes,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="darkgreen"),
    )

    # -------------------------------------------------------------------------
    # 3. EPISODE LENGTHS PLOT (Trajectory Efficiency)
    # -------------------------------------------------------------------------
    axs[1, 0].plot(
        episodes, episode_lengths, color="tab:blue", alpha=0.3, label="Raw"
    )
    ma_len = get_ma(episode_lengths)
    if ma_len is not None:
        axs[1, 0].plot(
            episodes[9:], ma_len, color="darkblue", linewidth=2, label="10-Ep MA"
        )

    # Crop out the longest 5% of exploratory loops
    y_max_len = np.percentile(episode_lengths, 95)
    axs[1, 0].set_ylim(bottom=0, top=y_max_len)
    axs[1, 0].set_title("Episode Length (Steps to Goal)")
    axs[1, 0].set_xlabel("Episodes")
    axs[1, 0].set_ylabel("Steps Taken")
    axs[1, 0].grid(True, linestyle="--", alpha=0.5)
    axs[1, 0].legend()

    # Outlier Label
    axs[1, 0].text(
        0.55,
        0.90,
        f"True Max Steps: {np.max(episode_lengths):,}",
        transform=axs[1, 0].transAxes,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="darkblue"),
    )

    # -------------------------------------------------------------------------
    # 4. EPISODE DURATIONS PLOT (Computation Overhead)
    # -------------------------------------------------------------------------
    axs[1, 1].plot(
        episodes, episode_durations, color="tab:orange", alpha=0.3, label="Raw"
    )
    ma_dur = get_ma(episode_durations)
    if ma_dur is not None:
        axs[1, 1].plot(
            episodes[9:], ma_dur, color="darkorange", linewidth=2, label="10-Ep MA"
        )

    # Crop out the longest 5% compute execution delays
    y_max_dur = np.percentile(episode_durations, 95)
    axs[1, 1].set_ylim(bottom=0, top=y_max_dur)
    axs[1, 1].set_title("Wall-Clock Episode Duration")
    axs[1, 1].set_xlabel("Episodes")
    axs[1, 1].set_ylabel("Time (Seconds)")
    axs[1, 1].grid(True, linestyle="--", alpha=0.5)
    axs[1, 1].legend()

    # Outlier Label
    axs[1, 1].text(
        0.55,
        0.90,
        f"True Max Duration: {np.max(episode_durations):.2f}s",
        transform=axs[1, 1].transAxes,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="darkorange"),
    )

    # Final presentation adjustment
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_q_table(q_table):
    rows, cols = 4, 12
    fig, ax = plt.subplots(figsize=(14, 5))

    # Normalize Q-values for background color mapping
    vmin, vmax = -20, 0
    cmap = plt.cm.viridis

    for s in range(len(q_table)):
        r = s // cols
        c = s % cols
        
        # Map row index to matplotlib's upward y-axis
        y = rows - 1 - r
        x = c
        
        # 1. Define the absolute outer corners of the grid box
        bl = [x, y]          # Bottom-Left
        br = [x + 1, y]      # Bottom-Right
        tl = [x, y + 1]      # Top-Left
        tr = [x + 1, y + 1]  # Top-Right
        center = [x + 0.5, y + 0.5]
        
        # 2. Slice the box into 4 clean triangles using a cross pattern
        corners = {
            0: [tl, tr, center],  # Up (Top Triangle)
            1: [tr, br, center],  # Right (Right Triangle)
            2: [bl, br, center],  # Down (Bottom Triangle)
            3: [tl, bl, center]   # Left (Left Triangle)
        }
        
        for a in range(4):
            q_val = q_table[s, a]
            
            # Highlight Cliff zone
            if r == 3 and 0 < c < 11:
                color = 'black'
                text_color = 'white'
            else:
                clipped_val = max(vmin, min(vmax, q_val))
                color = cmap((clipped_val - vmin) / (vmax - vmin))
                text_color = 'white' if (clipped_val - vmin) / (vmax - vmin) < 0.5 else 'black'
            
            # Draw the triangle patch
            poly = patches.Polygon(corners[a], closed=True, facecolor=color, edgecolor='white', linewidth=0.5)
            ax.add_patch(poly)
            
            # 3. Position text exactly in the middle/meat of each slice
            offset = 0.22
            if a == 0: text_pos = [center[0], center[1] + offset]    # Up
            elif a == 1: text_pos = [center[0] + offset, center[1]]  # Right
            elif a == 2: text_pos = [center[0], center[1] - offset]  # Down
            elif a == 3: text_pos = [center[0] - offset, center[1]]  # Left
            
            if r == 3 and 0 < c < 11:
                if a == 0:
                    ax.text(center[0], center[1], "CLIFF", color='white', ha='center', va='center', fontsize=8, fontweight='bold')
            else:
                ax.text(text_pos[0], text_pos[1], f"{q_val:.1f}", color=text_color, ha='center', va='center', fontsize=7)

    # Label START and GOAL positions at the bottom of their respective cells
    ax.text(0.5, 0.1, "START", color='white', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.text(11.5, 0.1, "GOAL", color='white', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Grid line adjustments
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    ax.set_xticklabels([f"C{c}" for c in range(cols)])
    ax.set_yticklabels([f"R{r}" for r in reversed(range(rows))])
    ax.grid(color='white', linestyle='-', linewidth=1)
    ax.set_aspect('equal')
    
    plt.title("Cliff Walking Q-Table (Cross-Section Triangles)", fontsize=14, pad=15)
    plt.show()
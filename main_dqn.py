import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as cx
import time
from shapely.geometry import LineString
from road_repair_env import RoadRepairEnv
from dqn_agent import DQNAgent
import random
import matplotlib.animation as animation
import os
import numpy as np
import torch

def get_valid_actions(env, state):
    """Get valid actions based on budget constraints"""
    valid_actions = []
    for i in range(env.num_roads):
        if i not in env.repaired and env.remaining_budget >= env.df.iloc[i]['repair_cost']:
            valid_actions.append(i)
    return valid_actions

def run_dqn_visualization(model_path=None, max_steps=30, budget=100000):
    """Run DQN agent with visualization"""
    
    # Load preprocessed GeoJSON road data
    df = gpd.read_file("coimbatore_road_preprocessed.geojson")
    df = df.to_crs(epsg=3857)
    
    # Initialize environment
    env = RoadRepairEnv(df, max_steps=max_steps, budget=budget)
    
    # Get state and action dimensions
    state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_size = env.action_space.n
    
    # Initialize DQN agent
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        lr=0.001,
        gamma=0.99,
        epsilon=0.0,  # No exploration during visualization
        epsilon_min=0.01,
        epsilon_decay=0.995,
        memory_size=10000,
        batch_size=64,
        target_update=100
    )
    
    # Load trained model if provided
    if model_path and os.path.exists(model_path):
        agent.load_model(model_path)
        print(f"Loaded trained model from {model_path}")
    else:
        print("No trained model found, using random policy")
    
    # Reset environment
    obs = env.reset()
    done = False
    total_reward = 0
    print("Starting DQN Road Repair Episode")
    
    # Setup plot
    fig, ax = plt.subplots(figsize=(14, 14))
    plt.ion()
    
    # Plot all roads in light gray
    df.plot(ax=ax, color='lightgray', linewidth=1, alpha=0.6)
    cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, zoom=13)
    ax.set_title("DQN Road Repair Simulation", fontsize=20, pad=20)
    plt.draw()
    plt.pause(1.0)
    
    # Track repaired segment indices
    repaired_segments = set()
    step = 0
    # Track snapshots of priority for city-wide heatmap animation
    priority_snapshots = []
    repaired_progress = []
    
    # Run DQN agent
    while step < max_steps and not done:
        state_flat = obs.flatten()
        valid_actions = get_valid_actions(env, obs)
        
        if not valid_actions:
            print("No valid actions remaining, episode ended")
            break
        
        # Get action from DQN agent
        action = agent.act(state_flat, valid_actions)
        
        # Get segment info for visualization
        segment = df.iloc[action]
        road_name = segment['name']
        segment_cost = segment['repair_cost']
        
        # Check if we can afford this repair
        if env.remaining_budget < segment_cost:
            print(f"Cannot afford repair {action}, skipping...")
            continue
        
        # Zoom in to show repair
        ax.clear()
        df.plot(ax=ax, color='lightgray', linewidth=0.5, alpha=0.3)
        gpd.GeoSeries([segment.geometry]).plot(ax=ax, color='red', linewidth=3, zorder=5)
        buffer = segment.geometry.buffer(300)
        ax.set_xlim(buffer.bounds[0], buffer.bounds[2])
        ax.set_ylim(buffer.bounds[1], buffer.bounds[3])
        cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, zoom=16)
        ax.set_title(f"DQN Repairing: {road_name}\nStep {step+1}/{max_steps}", 
                    fontsize=16, color='red')
        plt.tight_layout()
        plt.draw()
        plt.pause(0.6)
        
        # Step environment
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        env.render()
        repaired_segments.add(action)
        # Store snapshot for heatmap visualization
        priority_snapshots.append(env.df['priority_score'].copy())
        repaired_progress.append(set(repaired_segments))
        
        # Show green after repair
        ax.clear()
        df.plot(ax=ax, color='lightgray', linewidth=0.5, alpha=0.3)
        gpd.GeoSeries([segment.geometry]).plot(ax=ax, color='green', linewidth=3, zorder=5)
        ax.set_xlim(buffer.bounds[0], buffer.bounds[2])
        ax.set_ylim(buffer.bounds[1], buffer.bounds[3])
        cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, zoom=16)
        ax.set_title(f"DQN Repaired: {road_name}\nReward: {reward:.2f}", 
                    fontsize=16, color='green')
        plt.tight_layout()
        plt.draw()
        plt.pause(0.8)
        
        step += 1
        
        if env.remaining_budget <= 0:
            print("Budget exhausted, episode ended")
            break
    
    # Final city overview
    ax.clear()
    df.plot(ax=ax, color='lightgray', linewidth=0.5, alpha=0.3)
    if repaired_segments:
        df.loc[list(repaired_segments)].plot(ax=ax, color='green', linewidth=2, zorder=5)
    ax.set_xlim(df.total_bounds[0], df.total_bounds[2])
    ax.set_ylim(df.total_bounds[1], df.total_bounds[3])
    cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, zoom=13)
    ax.set_title(f"DQN All Repaired Segments\nTotal Reward: {total_reward:.2f}", 
                fontsize=16, color='green')
    plt.tight_layout()
    plt.draw()
    plt.pause(2.0)
    
    print(f"\nDQN Episode finished! Total reward collected: {total_reward:.2f}")
    print(f"Remaining budget: {env.remaining_budget:.2f}")
    print(f"Segments repaired: {len(repaired_segments)}")
    
    # Save as video
    save_animation(fig, df, repaired_segments, "dqn_repair_animation.mp4")
    # Save city-wide heatmap video of priority over time
    if priority_snapshots:
        save_heatmap_animation(df, priority_snapshots, repaired_progress, "dqn_city_heatmap.mp4")
    
    plt.ioff()
    plt.show()
    
    return total_reward, len(repaired_segments)

def save_animation(fig, df, repaired_segments, filename):
    """Save repair animation as video"""
    from matplotlib.animation import FFMpegWriter
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(current_dir, filename)
    
    writer = FFMpegWriter(fps=2, metadata=dict(artist='DQN Agent'), bitrate=1800)
    
    with writer.saving(fig, video_path, dpi=100):
        for idx in list(repaired_segments):
            segment = df.loc[idx]
            ax = fig.axes[0]
            ax.clear()
            df.plot(ax=ax, color='lightgray', linewidth=0.5, alpha=0.3)
            gpd.GeoSeries([segment.geometry]).plot(ax=ax, color='green', linewidth=3, zorder=5)
            buffer = segment.geometry.buffer(300)
            ax.set_xlim(buffer.bounds[0], buffer.bounds[2])
            ax.set_ylim(buffer.bounds[1], buffer.bounds[3])
            cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, zoom=16)
            ax.set_title(f"DQN Repaired: {df.loc[idx, 'name']}", fontsize=12)
            plt.tight_layout()
            writer.grab_frame()
    
    print(f"Animation saved as {video_path}")

def save_heatmap_animation(df, priority_snapshots, repaired_progress, filename):
    """Save city-wide heatmap animation of priority over time."""
    from matplotlib.animation import FFMpegWriter
    import matplotlib.cm as cm
    import matplotlib.colors as colors

    fig, ax = plt.subplots(figsize=(12, 12))
    current_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(current_dir, filename)

    # Normalize across all snapshots for consistent color scale
    all_vals = np.concatenate([snap.values.astype(float) for snap in priority_snapshots])
    vmin = float(np.nanmin(all_vals)) if np.isfinite(all_vals).any() else 0.0
    vmax = float(np.nanmax(all_vals)) if np.isfinite(all_vals).any() else 1.0
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap('inferno')

    writer = FFMpegWriter(fps=2, metadata=dict(artist='DQN Agent'), bitrate=1800)
    with writer.saving(fig, video_path, dpi=100):
        for t, (snap, repaired_set) in enumerate(zip(priority_snapshots, repaired_progress), start=1):
            ax.clear()
            # Map snapshot values to color per segment
            color_series = snap.reindex(df.index).astype(float)
            # Default gray for missing
            colors_list = [cmap(norm(val)) if np.isfinite(val) else (0.8, 0.8, 0.8, 1.0) for val in color_series]
            df.plot(ax=ax, color=colors_list, linewidth=1.0, alpha=0.9)
            # Overlay repaired segments so far in green
            if repaired_set:
                df.loc[list(repaired_set)].plot(ax=ax, color='limegreen', linewidth=2.5, zorder=5)
            cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, zoom=13)
            ax.set_title(f"City Priority Heatmap - Step {t}")
            ax.set_xlim(df.total_bounds[0], df.total_bounds[2])
            ax.set_ylim(df.total_bounds[1], df.total_bounds[3])
            plt.tight_layout()
            writer.grab_frame()
    plt.close(fig)
    print(f"Heatmap animation saved as {video_path}")

def compare_strategies():
    """Compare DQN vs Random vs Greedy strategies"""
    print("Comparing different repair strategies...")
    
    # Load data
    df = gpd.read_file("coimbatore_road_preprocessed.geojson")
    df = df.to_crs(epsg=3857)
    
    strategies = {
        'Random': run_random_strategy(df),
        'Greedy': run_greedy_strategy(df),
        'DQN': run_dqn_strategy(df)
    }
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    strategy_names = list(strategies.keys())
    rewards = [strategies[name]['reward'] for name in strategy_names]
    segments = [strategies[name]['segments'] for name in strategy_names]
    
    x = np.arange(len(strategy_names))
    width = 0.35
    
    ax2 = ax.twinx()
    bars1 = ax.bar(x - width/2, rewards, width, label='Total Reward', alpha=0.8)
    bars2 = ax2.bar(x + width/2, segments, width, label='Segments Repaired', alpha=0.8, color='orange')
    
    ax.set_xlabel('Strategy')
    ax.set_ylabel('Total Reward', color='blue')
    ax2.set_ylabel('Segments Repaired', color='orange')
    ax.set_title('Strategy Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(strategy_names)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}', ha='center', va='bottom')
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('strategy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return strategies

def run_random_strategy(df):
    """Run random repair strategy"""
    env = RoadRepairEnv(df, max_steps=30, budget=100000)
    obs = env.reset()
    total_reward = 0
    repaired = set()
    
    for step in range(30):
        valid_actions = get_valid_actions(env, obs)
        if not valid_actions:
            break
        action = random.choice(valid_actions)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        repaired.add(action)
        if done:
            break
    
    return {'reward': total_reward, 'segments': len(repaired)}

def run_greedy_strategy(df):
    """Run greedy repair strategy (highest priority first)"""
    env = RoadRepairEnv(df, max_steps=30, budget=100000)
    obs = env.reset()
    total_reward = 0
    repaired = set()
    
    # Sort by priority score
    available_segments = df[~df.index.isin(repaired)].copy()
    available_segments = available_segments.sort_values('priority_score', ascending=False)
    
    for step in range(30):
        valid_actions = get_valid_actions(env, obs)
        if not valid_actions:
            break
        
        # Choose highest priority available action
        for idx in available_segments.index:
            if idx in valid_actions:
                action = idx
                break
        else:
            break
        
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        repaired.add(action)
        if done:
            break
    
    return {'reward': total_reward, 'segments': len(repaired)}

def run_dqn_strategy(df):
    """Run DQN strategy"""
    env = RoadRepairEnv(df, max_steps=30, budget=100000)
    state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_size = env.action_space.n
    
    agent = DQNAgent(state_size=state_size, action_size=action_size, epsilon=0.0)
    
    # Try to load trained model
    model_paths = ["dqn_model_final.pth", "dqn_model_episode_900.pth", "dqn_model_episode_800.pth"]
    model_loaded = False
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            agent.load_model(model_path)
            model_loaded = True
            break
    
    if not model_loaded:
        print("No trained DQN model found, using random policy")
    
    obs = env.reset()
    total_reward = 0
    repaired = set()
    
    for step in range(30):
        state_flat = obs.flatten()
        valid_actions = get_valid_actions(env, obs)
        if not valid_actions:
            break
        
        action = agent.act(state_flat, valid_actions)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        repaired.add(action)
        if done:
            break
    
    return {'reward': total_reward, 'segments': len(repaired)}

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    print("DQN Road Repair Automation")
    print("=" * 50)
    
    # Check if trained model exists
    model_paths = ["dqn_model_final.pth", "dqn_model_episode_900.pth", "dqn_model_episode_800.pth"]
    model_path = None
    
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path:
        print(f"Found trained model: {model_path}")
        print("Running DQN visualization...")
        reward, segments = run_dqn_visualization(model_path)
    else:
        print("No trained model found. Please run train_dqn.py first.")
        print("Running comparison of strategies...")
        strategies = compare_strategies()

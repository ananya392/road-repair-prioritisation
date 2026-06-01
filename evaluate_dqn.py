import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from road_repair_env import RoadRepairEnv
from dqn_agent import DQNAgent
import torch
import random
import os
from collections import defaultdict

def get_valid_actions(env, state):
    """Get valid actions based on budget constraints"""
    valid_actions = []
    for i in range(env.num_roads):
        if i not in env.repaired and env.remaining_budget >= env.df.iloc[i]['repair_cost']:
            valid_actions.append(i)
    return valid_actions

def evaluate_agent_comprehensive(agent, env, num_episodes=50, strategy_name="DQN"):
    """Comprehensive evaluation of the agent"""
    print(f"Evaluating {strategy_name} agent over {num_episodes} episodes...")
    
    episode_data = []
    
    for episode in range(num_episodes):
        state = env.reset()
        state_flat = state.flatten()
        total_reward = 0
        step = 0
        budget_used = 0
        repair_sequence = []
        
        while step < env.max_steps:
            valid_actions = get_valid_actions(env, state)
            
            if not valid_actions:
                break
            
            # Use greedy policy (no exploration)
            old_epsilon = agent.epsilon
            agent.epsilon = 0.0
            action = agent.act(state_flat, valid_actions)
            agent.epsilon = old_epsilon
            
            # Get segment info before repair
            segment_info = {
                'index': action,
                'name': env.df.iloc[action]['name'],
                'damage_level': env.df.iloc[action]['damage_level'],
                'traffic': env.df.iloc[action]['traffic'],
                'repair_cost': env.df.iloc[action]['repair_cost'],
                'priority_score': env.df.iloc[action]['priority_score'],
                'social_weight': env.df.iloc[action]['social_weight']
            }
            
            next_state, reward, done, _ = env.step(action)
            next_state_flat = next_state.flatten()
            
            repair_sequence.append(segment_info)
            state_flat = next_state_flat
            total_reward += reward
            budget_used += segment_info['repair_cost']
            step += 1
            
            if done:
                break
        
        episode_data.append({
            'episode': episode,
            'total_reward': total_reward,
            'steps': step,
            'budget_used': budget_used,
            'budget_remaining': env.remaining_budget,
            'segments_repaired': len(repair_sequence),
            'repair_sequence': repair_sequence
        })
    
    return episode_data

def analyze_repair_patterns(episode_data, strategy_name):
    """Analyze repair patterns and preferences"""
    print(f"\nAnalyzing repair patterns for {strategy_name}...")
    
    # Collect all repair decisions
    all_repairs = []
    for episode in episode_data:
        for repair in episode['repair_sequence']:
            all_repairs.append(repair)
    
    if not all_repairs:
        print("No repairs made in any episode")
        return
    
    # Convert to DataFrame for analysis
    repair_df = pd.DataFrame(all_repairs)
    
    # Analyze by different criteria
    print(f"\n{strategy_name} Repair Analysis:")
    print(f"Total repairs across all episodes: {len(all_repairs)}")
    print(f"Average repairs per episode: {len(all_repairs) / len(episode_data):.2f}")
    
    # Damage level analysis
    print(f"\nDamage Level Distribution:")
    print(f"Mean damage level: {repair_df['damage_level'].mean():.3f}")
    print(f"Std damage level: {repair_df['damage_level'].std():.3f}")
    
    # Traffic analysis
    print(f"\nTraffic Distribution:")
    print(f"Mean traffic: {repair_df['traffic'].mean():.0f}")
    print(f"Std traffic: {repair_df['traffic'].std():.0f}")
    
    # Cost analysis
    print(f"\nCost Distribution:")
    print(f"Mean repair cost: {repair_df['repair_cost'].mean():.2f}")
    print(f"Std repair cost: {repair_df['repair_cost'].std():.2f}")
    
    # Priority score analysis
    print(f"\nPriority Score Distribution:")
    print(f"Mean priority score: {repair_df['priority_score'].mean():.2f}")
    print(f"Std priority score: {repair_df['priority_score'].std():.2f}")
    
    return repair_df

def plot_evaluation_results(episode_data, strategy_name):
    """Plot comprehensive evaluation results"""
    rewards = [ep['total_reward'] for ep in episode_data]
    steps = [ep['steps'] for ep in episode_data]
    budget_used = [ep['budget_used'] for ep in episode_data]
    segments = [ep['segments_repaired'] for ep in episode_data]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{strategy_name} Evaluation Results', fontsize=16)
    
    # Episode rewards
    axes[0, 0].hist(rewards, bins=20, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Episode Rewards Distribution')
    axes[0, 0].set_xlabel('Total Reward')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(np.mean(rewards), color='red', linestyle='--', label=f'Mean: {np.mean(rewards):.2f}')
    axes[0, 0].legend()
    
    # Episode steps
    axes[0, 1].hist(steps, bins=20, alpha=0.7, edgecolor='black', color='orange')
    axes[0, 1].set_title('Episode Length Distribution')
    axes[0, 1].set_xlabel('Steps')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].axvline(np.mean(steps), color='red', linestyle='--', label=f'Mean: {np.mean(steps):.2f}')
    axes[0, 1].legend()
    
    # Budget usage
    axes[0, 2].hist(budget_used, bins=20, alpha=0.7, edgecolor='black', color='green')
    axes[0, 2].set_title('Budget Usage Distribution')
    axes[0, 2].set_xlabel('Budget Used')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].axvline(np.mean(budget_used), color='red', linestyle='--', label=f'Mean: {np.mean(budget_used):.2f}')
    axes[0, 2].legend()
    
    # Segments repaired
    axes[1, 0].hist(segments, bins=20, alpha=0.7, edgecolor='black', color='purple')
    axes[1, 0].set_title('Segments Repaired Distribution')
    axes[1, 0].set_xlabel('Segments Repaired')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].axvline(np.mean(segments), color='red', linestyle='--', label=f'Mean: {np.mean(segments):.2f}')
    axes[1, 0].legend()
    
    # Reward vs Steps scatter
    axes[1, 1].scatter(steps, rewards, alpha=0.6)
    axes[1, 1].set_title('Reward vs Steps')
    axes[1, 1].set_xlabel('Steps')
    axes[1, 1].set_ylabel('Total Reward')
    
    # Budget efficiency
    efficiency = [r/b if b > 0 else 0 for r, b in zip(rewards, budget_used)]
    axes[1, 2].hist(efficiency, bins=20, alpha=0.7, edgecolor='black', color='brown')
    axes[1, 2].set_title('Budget Efficiency (Reward/Budget)')
    axes[1, 2].set_xlabel('Efficiency')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].axvline(np.mean(efficiency), color='red', linestyle='--', label=f'Mean: {np.mean(efficiency):.3f}')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig(f'{strategy_name.lower()}_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()

def compare_strategies_detailed():
    """Compare DQN vs other strategies in detail"""
    print("Running detailed strategy comparison...")
    
    # Load data
    df = gpd.read_file("coimbatore_road_preprocessed.geojson")
    df = df.to_crs(epsg=3857)
    
    strategies = {}
    
    # Random Strategy
    print("\nEvaluating Random Strategy...")
    env = RoadRepairEnv(df, max_steps=30, budget=100000)
    state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_size = env.action_space.n
    random_agent = DQNAgent(state_size=state_size, action_size=action_size, epsilon=1.0)
    strategies['Random'] = evaluate_agent_comprehensive(random_agent, env, num_episodes=20, strategy_name="Random")
    
    # Greedy Strategy
    print("\nEvaluating Greedy Strategy...")
    greedy_agent = DQNAgent(state_size=state_size, action_size=action_size, epsilon=0.0)
    # For greedy, we'll implement a custom action selection
    strategies['Greedy'] = evaluate_greedy_strategy(df, num_episodes=20)
    
    # DQN Strategy
    print("\nEvaluating DQN Strategy...")
    dqn_agent = DQNAgent(state_size=state_size, action_size=action_size, epsilon=0.0)
    
    # Try to load trained model
    model_paths = ["dqn_model_final.pth", "dqn_model_episode_900.pth", "dqn_model_episode_800.pth"]
    model_loaded = False
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            dqn_agent.load_model(model_path)
            model_loaded = True
            print(f"Loaded DQN model from {model_path}")
            break
    
    if model_loaded:
        strategies['DQN'] = evaluate_agent_comprehensive(dqn_agent, env, num_episodes=20, strategy_name="DQN")
    else:
        print("No trained DQN model found, skipping DQN evaluation")
    
    # Analyze and plot results
    for strategy_name, episode_data in strategies.items():
        print(f"\n{'='*50}")
        print(f"STRATEGY: {strategy_name}")
        print(f"{'='*50}")
        
        rewards = [ep['total_reward'] for ep in episode_data]
        steps = [ep['steps'] for ep in episode_data]
        segments = [ep['segments_repaired'] for ep in episode_data]
        
        print(f"Average Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        print(f"Average Steps: {np.mean(steps):.2f} ± {np.std(steps):.2f}")
        print(f"Average Segments: {np.mean(segments):.2f} ± {np.std(segments):.2f}")
        
        # Plot individual strategy results
        plot_evaluation_results(episode_data, strategy_name)
        
        # Analyze repair patterns
        repair_df = analyze_repair_patterns(episode_data, strategy_name)
    
    # Plot comparison
    plot_strategy_comparison(strategies)
    
    return strategies

def evaluate_greedy_strategy(df, num_episodes=20):
    """Evaluate greedy strategy (highest priority first)"""
    episode_data = []
    
    for episode in range(num_episodes):
        env = RoadRepairEnv(df, max_steps=30, budget=100000)
        state = env.reset()
        total_reward = 0
        step = 0
        budget_used = 0
        repair_sequence = []
        
        # Sort segments by priority score
        available_segments = df.copy()
        available_segments = available_segments.sort_values('priority_score', ascending=False)
        
        for step in range(30):
            valid_actions = get_valid_actions(env, state)
            if not valid_actions:
                break
            
            # Choose highest priority available action
            action = None
            for idx in available_segments.index:
                if idx in valid_actions:
                    action = idx
                    break
            
            if action is None:
                break
            
            # Get segment info
            segment_info = {
                'index': action,
                'name': env.df.iloc[action]['name'],
                'damage_level': env.df.iloc[action]['damage_level'],
                'traffic': env.df.iloc[action]['traffic'],
                'repair_cost': env.df.iloc[action]['repair_cost'],
                'priority_score': env.df.iloc[action]['priority_score'],
                'social_weight': env.df.iloc[action]['social_weight']
            }
            
            next_state, reward, done, _ = env.step(action)
            repair_sequence.append(segment_info)
            total_reward += reward
            budget_used += segment_info['repair_cost']
            step += 1
            
            if done:
                break
        
        episode_data.append({
            'episode': episode,
            'total_reward': total_reward,
            'steps': step,
            'budget_used': budget_used,
            'budget_remaining': env.remaining_budget,
            'segments_repaired': len(repair_sequence),
            'repair_sequence': repair_sequence
        })
    
    return episode_data

def plot_strategy_comparison(strategies):
    """Plot comparison between different strategies"""
    strategy_names = list(strategies.keys())
    
    # Extract metrics
    rewards = {name: [ep['total_reward'] for ep in data] for name, data in strategies.items()}
    steps = {name: [ep['steps'] for ep in data] for name, data in strategies.items()}
    segments = {name: [ep['segments_repaired'] for ep in data] for name, data in strategies.items()}
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Strategy Comparison', fontsize=16)
    
    # Box plots for rewards
    axes[0, 0].boxplot([rewards[name] for name in strategy_names], labels=strategy_names)
    axes[0, 0].set_title('Reward Distribution')
    axes[0, 0].set_ylabel('Total Reward')
    
    # Box plots for steps
    axes[0, 1].boxplot([steps[name] for name in strategy_names], labels=strategy_names)
    axes[0, 1].set_title('Episode Length Distribution')
    axes[0, 1].set_ylabel('Steps')
    
    # Box plots for segments
    axes[1, 0].boxplot([segments[name] for name in strategy_names], labels=strategy_names)
    axes[1, 0].set_title('Segments Repaired Distribution')
    axes[1, 0].set_ylabel('Segments Repaired')
    
    # Bar plot for average metrics
    avg_rewards = [np.mean(rewards[name]) for name in strategy_names]
    avg_steps = [np.mean(steps[name]) for name in strategy_names]
    avg_segments = [np.mean(segments[name]) for name in strategy_names]
    
    x = np.arange(len(strategy_names))
    width = 0.25
    
    axes[1, 1].bar(x - width, avg_rewards, width, label='Avg Reward', alpha=0.8)
    axes[1, 1].bar(x, avg_steps, width, label='Avg Steps', alpha=0.8)
    axes[1, 1].bar(x + width, avg_segments, width, label='Avg Segments', alpha=0.8)
    
    axes[1, 1].set_title('Average Metrics Comparison')
    axes[1, 1].set_xlabel('Strategy')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(strategy_names)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('strategy_comparison_detailed.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    print("DQN Road Repair Evaluation")
    print("=" * 50)
    
    # Run comprehensive evaluation
    strategies = compare_strategies_detailed()
    
    print("\nEvaluation completed! Check the generated plots for detailed analysis.")

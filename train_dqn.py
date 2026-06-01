import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import os
from road_repair_env import RoadRepairEnv
from deterioration_model import MarkovDeteriorationModel
from dqn_agent import DQNAgent
import torch
import random
from collections import deque

def get_valid_actions(env, state):
    """Get valid actions based on budget constraints"""
    valid_actions = []
    for i in range(env.num_roads):
        if i not in env.repaired and env.remaining_budget >= env.df.iloc[i]['repair_cost']:
            valid_actions.append(i)
    return valid_actions

def train_dqn(episodes=1000, max_steps=30, budget=100000, save_interval=100):
    """Train DQN agent on road repair environment"""
    
    # Load preprocessed road data
    print("Loading road data...")
    df = gpd.read_file("coimbatore_road_preprocessed.geojson")
    df = df.to_crs(epsg=3857)
    
    # Initialize environment with a default Markov deterioration model (optional override)
    # Slightly more deterioration than default for training signal
    num_states = 7
    tm = np.eye(num_states) * 0.80
    for i in range(num_states):
        if i < num_states - 1:
            tm[i, i + 1] += 0.20
        else:
            tm[i, i] += 0.20
    tm = tm / tm.sum(axis=1, keepdims=True)
    markov = MarkovDeteriorationModel(num_states=num_states, transition_matrix=tm)
    env = RoadRepairEnv(df, max_steps=max_steps, budget=budget, markov_model=markov)
    
    # Get state and action dimensions
    state_size = env.observation_space.shape[0] * env.observation_space.shape[1]  # Flatten state
    action_size = env.action_space.n
    
    # Initialize DQN agent
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        lr=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        memory_size=10000,
        batch_size=64,
        target_update=100
    )
    
    # Training tracking
    episode_rewards = []
    episode_lengths = []
    avg_rewards = deque(maxlen=100)
    
    print(f"Starting DQN training for {episodes} episodes...")
    print(f"State size: {state_size}, Action size: {action_size}")
    
    for episode in range(episodes):
        # Reset environment
        state = env.reset()
        state_flat = state.flatten()
        total_reward = 0
        step = 0
        
        while step < max_steps:
            # Get valid actions
            valid_actions = get_valid_actions(env, state)
            
            if not valid_actions:
                # No valid actions, episode ends
                break
            
            # Choose action
            action = agent.act(state_flat, valid_actions)
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            next_state_flat = next_state.flatten()
            
            # Store experience
            agent.remember(state_flat, action, reward, next_state_flat, done)
            
            # Train the agent
            agent.replay()
            
            # Update state and tracking
            state_flat = next_state_flat
            total_reward += reward
            step += 1
            
            if done:
                break
        
        # Track episode results
        episode_rewards.append(total_reward)
        episode_lengths.append(step)
        avg_rewards.append(total_reward)
        agent.rewards.append(total_reward)
        
        # Print progress
        if episode % 50 == 0:
            avg_reward = np.mean(avg_rewards)
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, "
                  f"Epsilon: {agent.epsilon:.3f}, Steps: {step}")
        
        # Save model periodically
        if episode % save_interval == 0 and episode > 0:
            model_path = f"dqn_model_episode_{episode}.pth"
            agent.save_model(model_path)
            print(f"Model saved at episode {episode}")
    
    # Save final model
    final_model_path = "dqn_model_final.pth"
    agent.save_model(final_model_path)
    print(f"Final model saved as {final_model_path}")
    
    # Plot training progress
    plot_training_progress(episode_rewards, episode_lengths, agent.losses)
    
    return agent, episode_rewards, episode_lengths

def plot_training_progress(episode_rewards, episode_lengths, losses):
    """Plot training progress"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Episode rewards
    axes[0, 0].plot(episode_rewards)
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].grid(True)
    
    # Moving average of rewards
    window = 100
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        axes[0, 1].plot(moving_avg)
        axes[0, 1].set_title(f'Moving Average Rewards (window={window})')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Average Reward')
        axes[0, 1].grid(True)
    
    # Episode lengths
    axes[1, 0].plot(episode_lengths)
    axes[1, 0].set_title('Episode Lengths')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Steps')
    axes[1, 0].grid(True)
    
    # Training losses
    if losses:
        axes[1, 1].plot(losses)
        axes[1, 1].set_title('Training Loss')
        axes[1, 1].set_xlabel('Training Steps')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_agent(agent, env, num_episodes=10, render=False):
    """Evaluate trained agent"""
    print(f"Evaluating agent over {num_episodes} episodes...")
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        state = env.reset()
        state_flat = state.flatten()
        total_reward = 0
        step = 0
        
        if render:
            print(f"\nEpisode {episode + 1}:")
        
        while step < env.max_steps:
            valid_actions = get_valid_actions(env, state)
            
            if not valid_actions:
                break
            
            # Use greedy policy (no exploration)
            old_epsilon = agent.epsilon
            agent.epsilon = 0.0
            action = agent.act(state_flat, valid_actions)
            agent.epsilon = old_epsilon
            
            next_state, reward, done, _ = env.step(action)
            next_state_flat = next_state.flatten()
            
            state_flat = next_state_flat
            total_reward += reward
            step += 1
            
            if render:
                print(f"  Step {step}: Action {action}, Reward {reward:.2f}, "
                      f"Budget: {env.remaining_budget:.2f}")
            
            if done:
                break
        
        episode_rewards.append(total_reward)
        episode_lengths.append(step)
        
        if render:
            print(f"  Episode {episode + 1} finished: Reward {total_reward:.2f}, Steps {step}")
    
    avg_reward = np.mean(episode_rewards)
    avg_length = np.mean(episode_lengths)
    
    print(f"\nEvaluation Results:")
    print(f"Average Reward: {avg_reward:.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Length: {avg_length:.2f} ± {np.std(episode_lengths):.2f}")
    
    return episode_rewards, episode_lengths

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Train the agent
    agent, rewards, lengths = train_dqn(
        episodes=5,
        max_steps=30,
        budget=100000,
        save_interval=1
    )
    
    # Evaluate the trained agent
    df = gpd.read_file("coimbatore_road_preprocessed.geojson")
    df = df.to_crs(epsg=3857)
    env = RoadRepairEnv(df, max_steps=30, budget=100000)
    
    eval_rewards, eval_lengths = evaluate_agent(agent, env, num_episodes=10, render=True)

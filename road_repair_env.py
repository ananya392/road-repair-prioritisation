import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional
from deterioration_model import MarkovDeteriorationModel, damage_to_state

class RoadRepairEnv(gym.Env):
    def __init__(self, road_df, max_steps=50, budget=100000, markov_model: Optional[MarkovDeteriorationModel] = None):
        super(RoadRepairEnv, self).__init__()

        self.original_df = road_df.copy()
        self.df = road_df.copy()
        self.max_steps = max_steps
        self.budget = budget
        self.remaining_budget = budget
        self.current_step = 0
        self.repaired = set()
        self.repair_log = []

        self.num_roads = len(self.df)

        # Markov deterioration model setup (7 damage states across [0,1])
        self.num_damage_states = 7
        self.state_bins = np.linspace(0.0, 1.0 + 1e-6, self.num_damage_states + 1).tolist()
        if markov_model is None:
            # Default: mostly stay, some chance to worsen by one state
            tm = np.eye(self.num_damage_states) * 0.85
            for i in range(self.num_damage_states):
                if i < self.num_damage_states - 1:
                    tm[i, i + 1] += 0.15
                else:
                    tm[i, i] += 0.15
            tm = tm / tm.sum(axis=1, keepdims=True)
            self.markov_model = MarkovDeteriorationModel(num_states=self.num_damage_states, transition_matrix=tm)
        else:
            self.markov_model = markov_model
        # Discrete damage states per segment (aligned with df index)
        self.segment_state = None

        self.action_space = spaces.Discrete(self.num_roads)
        self.observation_space = spaces.Box(
            low=0,
            high=np.inf,
            shape=(self.num_roads, 3),
            dtype=np.float32
        )

    def reset(self):
        self.df = self.original_df.copy()
        self.remaining_budget = self.budget
        self.current_step = 0
        self.repaired = set()
        self.repair_log = []
        # Initialize discrete states from current damage
        states, _, _ = damage_to_state(self.df['damage_level'].values, num_states=self.num_damage_states)
        # Store as pandas Series indexed by df index for stable alignment
        self.segment_state = pd.Series(states, index=self.df.index)
        return self._get_observation()

    def step(self, action):
        done = False
        reward = 0

        if self.remaining_budget <= 0 or self.current_step >= self.max_steps:
            done = True
            return self._get_observation(), reward, done, {}

        row = self.df.loc[action]  # ✅ index-based access

        if action in self.repaired:
            reward = -1  # Penalty for repeating
        elif self.remaining_budget >= row['repair_cost']:
            # ✅ Log before repair
            self.repair_log.append({
                'index': action,
                'name': row['name'],
                'damage_level': row['damage_level'],
                'traffic': row['traffic'],
                'repair_cost': row['repair_cost'],
                'priority_score': row['priority_score']
            })

            # ✅ Perform repair
            self.df.at[action, 'damage_level'] = 0.0
            if self.segment_state is not None and action in self.segment_state.index:
                self.segment_state.at[action] = 0  # reset to best state
            self.remaining_budget -= row['repair_cost']
            reward = row['priority_score']
            self.repaired.add(action)
        else:
            reward = -5  # Penalty for insufficient budget

        # Apply deterioration to all unrepaired segments after action
        self._apply_markov_deterioration()

        # Recompute dependent metrics after deterioration
        self._recompute_metrics()

        self.current_step += 1
        done = self.current_step >= self.max_steps or self.remaining_budget <= 0

        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        return self.df[['traffic', 'damage_level', 'repair_cost']].values.astype(np.float32)

    def render(self):
        print(f"\nStep {self.current_step}/{self.max_steps}")
        print(f"Remaining Budget: ₹{self.remaining_budget:.2f}")
        print(f"Segments repaired so far: {len(self.repaired)}")

        if self.repair_log:
            print("\n  Repaired Segments:")
            for entry in self.repair_log:
                print(f"  - Segment {entry['index']} | Name: {entry['name']} | "
                      f"Damage: {entry['damage_level']:.2f} | Traffic: {entry['traffic']} | "
                      f"Cost: ₹{entry['repair_cost']:.2f} | Priority: {entry['priority_score']:.2f}")

    # -----------------------
    # Internal helpers
    # -----------------------
    def _apply_markov_deterioration(self):
        if self.segment_state is None:
            return
        # Deteriorate only segments not repaired yet
        for idx in self.df.index:
            if idx in self.repaired:
                continue
            current_state = int(self.segment_state.at[idx])
            next_state = self.markov_model.next_state(current_state)
            self.segment_state.at[idx] = next_state
            # Map state to damage as bin midpoint
            low = self.state_bins[next_state]
            high = self.state_bins[next_state + 1]
            self.df.at[idx, 'damage_level'] = float((low + high) / 2.0)

    def _recompute_metrics(self):
        # Recompute PCI
        self.df['PCI'] = (1.0 - self.df['damage_level'].clip(0.0, 1.0)) * 100.0
        # Recompute priority score (traffic * damage * social_weight)
        if 'social_weight' in self.df.columns:
            self.df['priority_score'] = self.df['traffic'] * self.df['damage_level'] * self.df['social_weight']
        else:
            self.df['priority_score'] = self.df['traffic'] * self.df['damage_level']
        # Recompute repair cost consistent with preprocess formula
        base_rate = 5
        traffic_factor = self.df['traffic'] / self.df['traffic'].max()
        damage_factor = self.df['damage_level']
        if 'length' in self.df.columns:
            length = self.df['length'].fillna(0)
        else:
            # Fallback: unit length if not available
            length = 1.0
        self.df['repair_cost'] = base_rate * length * (1.0 + damage_factor + traffic_factor)

import numpy as np
import pandas as pd
from typing import List, Optional, Sequence, Tuple, Dict


class MarkovDeteriorationModel:
    """First-order Markov chain deterioration model for road segments.

    - States represent discrete condition classes (e.g., derived from PCI or damage).
    - Transition probabilities are estimated from panel/time-series observations.
    - Supports simulation with optional repair actions that reset/improve state.
    """

    def __init__(self, num_states: int, transition_matrix: Optional[np.ndarray] = None, state_names: Optional[List[str]] = None):
        if num_states <= 1:
            raise ValueError("num_states must be >= 2")
        self.num_states: int = num_states
        self.transition_matrix: np.ndarray = (
            transition_matrix if transition_matrix is not None else self._uniform_matrix(num_states)
        )
        self.state_names: List[str] = state_names if state_names is not None else [f"S{i}" for i in range(num_states)]
        self._validate()

    def _uniform_matrix(self, k: int) -> np.ndarray:
        return np.ones((k, k), dtype=float) / float(k)

    def _validate(self) -> None:
        if self.transition_matrix.shape != (self.num_states, self.num_states):
            raise ValueError("transition_matrix shape must be (num_states, num_states)")
        row_sums = self.transition_matrix.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-6):
            raise ValueError("Each row of transition_matrix must sum to 1")

    # -----------------------
    # Fitting
    # -----------------------
    @staticmethod
    def fit_from_sequences(
        sequences: List[Sequence[int]], num_states: int, laplace_alpha: float = 1.0
    ) -> "MarkovDeteriorationModel":
        """Estimate transition matrix from a list of integer state sequences.

        Args:
            sequences: list of sequences of state integers in [0, num_states-1]
            num_states: number of states
            laplace_alpha: pseudocount for Laplace smoothing to avoid zero rows
        """
        counts = np.full((num_states, num_states), laplace_alpha, dtype=float)
        for seq in sequences:
            for s, s_next in zip(seq[:-1], seq[1:]):
                if 0 <= s < num_states and 0 <= s_next < num_states:
                    counts[s, s_next] += 1.0
        # Normalize rows
        row_sums = counts.sum(axis=1, keepdims=True)
        trans = counts / np.maximum(row_sums, 1e-12)
        return MarkovDeteriorationModel(num_states=num_states, transition_matrix=trans)

    @staticmethod
    def fit_from_panel(
        df: pd.DataFrame,
        id_col: str,
        time_col: str,
        state_col: str,
        num_states: int,
        laplace_alpha: float = 1.0,
    ) -> "MarkovDeteriorationModel":
        """Estimate transitions from a panel DataFrame.

        The DataFrame should have rows per (segment_id, time) with an integer state label.
        It can be derived from PCI/damage via helper functions below.
        """
        # Ensure correct ordering per id
        sequences: List[List[int]] = []
        for _, group in df[[id_col, time_col, state_col]].dropna().sort_values([id_col, time_col]).groupby(id_col):
            seq = group[state_col].astype(int).tolist()
            if len(seq) >= 2:
                sequences.append(seq)
        return MarkovDeteriorationModel.fit_from_sequences(sequences, num_states, laplace_alpha)

    # -----------------------
    # Simulation
    # -----------------------
    def next_state(self, current_state: int, rng: Optional[np.random.Generator] = None) -> int:
        if rng is None:
            rng = np.random.default_rng()
        probs = self.transition_matrix[current_state]
        return int(rng.choice(self.num_states, p=probs))

    def simulate_horizon(
        self,
        initial_state: int,
        horizon: int,
        repair_policy: Optional["RepairPolicy"] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> List[int]:
        """Simulate state trajectory over a horizon with optional repairs."""
        if rng is None:
            rng = np.random.default_rng()
        states = [int(initial_state)]
        current = int(initial_state)
        for t in range(horizon):
            # Optionally apply repair before deterioration step
            if repair_policy is not None:
                current = repair_policy.apply(current, rng)
            current = self.next_state(current, rng)
            states.append(current)
        return states

    # -----------------------
    # Utilities
    # -----------------------
    def expected_years_to_absorbing(self, absorbing_states: Sequence[int], start_state: int, max_years: int = 100) -> float:
        """Monte Carlo approximation of expected time to reach any absorbing state."""
        rng = np.random.default_rng(123)
        trials = 2000
        total = 0.0
        for _ in range(trials):
            s = start_state
            for year in range(1, max_years + 1):
                s = self.next_state(s, rng)
                if s in absorbing_states:
                    total += year
                    break
            else:
                total += max_years
        return total / trials


class RepairPolicy:
    """Simple repair policy mapping a state to an improved state with probability.

    Examples:
        - Deterministic reset to a target state (e.g., post-repair PCI state).
        - Probabilistic improvement by k states with some probability of failure.
    """

    def __init__(self, improve_to_state: Optional[int] = None, improve_by: int = 0, success_prob: float = 1.0):
        if improve_to_state is None and improve_by == 0:
            raise ValueError("Specify improve_to_state or improve_by")
        self.improve_to_state = improve_to_state
        self.improve_by = improve_by
        self.success_prob = float(success_prob)

    def apply(self, current_state: int, rng: Optional[np.random.Generator] = None) -> int:
        if rng is None:
            rng = np.random.default_rng()
        if rng.random() > self.success_prob:
            return current_state
        if self.improve_to_state is not None:
            return int(self.improve_to_state)
        improved = max(0, current_state - int(self.improve_by))
        return improved


# -----------------------
# Discretization helpers
# -----------------------
def discretize_by_bins(values: np.ndarray, bin_edges: Sequence[float]) -> np.ndarray:
    """Discretize continuous values to integer states using bin edges.

    Returns integer states in [0, len(bin_edges)-2]. Uses np.digitize semantics with right=False.
    """
    indices = np.digitize(values, bins=np.array(bin_edges)[1:-1], right=False)
    return indices.astype(int)


def pci_to_state(pci_values: Sequence[float]) -> Tuple[np.ndarray, List[str], List[float]]:
    """Map PCI [0,100] to 7 states (X1 best .. X7 worst) using common buckets.

    Buckets mirror those used in your preprocessing script.
    """
    # Same bins as preprocess_road_data.py: [0,10,25,40,55,70,85,100]
    bins = [0.0, 10.0, 25.0, 40.0, 55.0, 70.0, 85.0, 100.01]
    states = discretize_by_bins(np.asarray(pci_values, dtype=float), bins)
    # Map to X1..X7 (reverse order), but return state names aligned with index 0..6
    state_names = ["X1", "X2", "X3", "X4", "X5", "X6", "X7"]
    return states, state_names, bins


def damage_to_state(damage_values: Sequence[float], num_states: int = 7) -> Tuple[np.ndarray, List[str], List[float]]:
    """Map damage in [0,1] to num_states quantile bins (worst at high damage)."""
    damage = np.asarray(damage_values, dtype=float)
    if np.any((damage < 0.0) | (damage > 1.0)):
        raise ValueError("damage values must be within [0,1]")
    # Create approximately equal-width bins by value (not frequency) for stability
    bins = np.linspace(0.0, 1.0 + 1e-6, num_states + 1).tolist()
    states = discretize_by_bins(damage, bins)
    state_names = [f"D{i+1}" for i in range(num_states)]
    return states, state_names, bins


# -----------------------
# Example usage (for reference)
# -----------------------
def example_fit_from_preprocessed_csv(csv_path: str, id_col: str = "u", time_col: str = "year", use_pci: bool = True) -> MarkovDeteriorationModel:
    """Example: fit a deterioration model from a CSV exported by preprocessing.

    Expects repeated observations per segment over time with either `PCI` or `damage_level`.
    If your dataset is cross-sectional (single snapshot), you can still simulate forward
    by assuming the fitted matrix from external data or defaults.
    """
    df = pd.read_csv(csv_path)
    if use_pci and "PCI" in df.columns:
        states, names, _ = pci_to_state(df["PCI"].values)
    elif "damage_level" in df.columns:
        states, names, _ = damage_to_state(df["damage_level"].values)
    else:
        raise ValueError("CSV must contain either 'PCI' or 'damage_level'")

    # If there is no time column, synthesize a trivial panel per segment (cannot truly fit transitions)
    if time_col not in df.columns:
        # Fall back to a heuristic stationary matrix: mild drift toward worse states
        k = len(np.unique(states))
        base = np.eye(k) * 0.85
        for i in range(k):
            if i < k - 1:
                base[i, i + 1] += 0.15
            else:
                base[i, i] += 0.15
        base = base / base.sum(axis=1, keepdims=True)
        return MarkovDeteriorationModel(num_states=k, transition_matrix=base, state_names=names)

    # Build panel with integer states
    df_panel = df[[id_col, time_col]].copy()
    df_panel["state"] = states

    model = MarkovDeteriorationModel.fit_from_panel(
        df=df_panel, id_col=id_col, time_col=time_col, state_col="state", num_states=len(names)
    )
    model.state_names = names
    return model


def simulate_segment_paths(
    model: MarkovDeteriorationModel,
    initial_states: Sequence[int],
    years: int,
    repair_after_years: Optional[int] = None,
    repair_reset_state: Optional[int] = None,
) -> np.ndarray:
    """Simulate multiple segments forward, optionally repairing at a fixed interval.

    Returns array of shape (n_segments, years+1) with state trajectories.
    """
    rng = np.random.default_rng(123)
    n = len(initial_states)
    trajectories = np.zeros((n, years + 1), dtype=int)
    for i, s0 in enumerate(initial_states):
        policy = None
        if repair_after_years is not None and repair_reset_state is not None:
            policy = _PeriodicRepairPolicy(repair_after_years, repair_reset_state)
        trajectories[i] = np.array(model.simulate_horizon(s0, years, repair_policy=policy, rng=rng), dtype=int)
    return trajectories


class _PeriodicRepairPolicy(RepairPolicy):
    """Repair every k years by resetting to a target state before deterioration."""

    def __init__(self, period_years: int, reset_state: int):
        super().__init__(improve_to_state=reset_state, success_prob=1.0)
        self.period_years = int(period_years)
        self._t = 0

    def apply(self, current_state: int, rng: Optional[np.random.Generator] = None) -> int:
        self._t += 1
        if self._t % self.period_years == 0:
            return super().apply(current_state, rng)
        return current_state



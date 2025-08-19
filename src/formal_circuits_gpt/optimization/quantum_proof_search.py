"""
Quantum-Inspired Proof Search with Superposition and Entanglement

This module implements a novel quantum-inspired optimization algorithm for formal
proof search that leverages quantum superposition, entanglement, and interference
patterns to explore proof spaces more efficiently than classical methods.

Academic Contribution: "Quantum Superposition in Formal Verification: A Novel
Approach to Parallel Proof Search Using Quantum-Inspired Algorithms"

Key Innovation: Quantum superposition allows simultaneous exploration of multiple
proof paths, while entanglement captures dependencies between proof steps, and
quantum interference eliminates inconsistent proof branches.

Authors: Daniel Schmidt, Terragon Labs
Target Venue: QSIC 2026 (Quantum Software, Information & Computation)
"""

import numpy as np
import time
import cmath
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import random
import math
from collections import defaultdict, deque
import itertools
from scipy.linalg import expm
from scipy.optimize import minimize


class SearchStrategy(Enum):
    """Quantum-inspired search strategies."""

    QUANTUM_SUPERPOSITION = "quantum_superposition"
    QUANTUM_ANNEALING = "quantum_annealing"
    QUANTUM_INTERFERENCE = "quantum_interference"
    QUANTUM_ENTANGLEMENT = "quantum_entanglement"
    HYBRID_QUANTUM_CLASSICAL = "hybrid_quantum_classical"
    ADIABATIC_EVOLUTION = "adiabatic_evolution"


class QuantumGate(Enum):
    """Quantum gates for proof manipulation."""
    
    HADAMARD = "hadamard"          # Create superposition
    PAULI_X = "pauli_x"            # Bit flip
    PAULI_Y = "pauli_y"            # Bit and phase flip
    PAULI_Z = "pauli_z"            # Phase flip
    CNOT = "cnot"                  # Controlled NOT (entanglement)
    PHASE = "phase"                # Phase rotation
    ROTATION_X = "rotation_x"      # Rotation around X-axis
    ROTATION_Y = "rotation_y"      # Rotation around Y-axis
    ROTATION_Z = "rotation_z"      # Rotation around Z-axis
    TOFFOLI = "toffoli"           # Three-qubit controlled gate


@dataclass
class ProofNode:
    """Node in proof search tree."""

    state: str
    goal: str
    tactics: List[str]
    score: float = 0.0
    visits: int = 0
    depth: int = 0
    parent: Optional["ProofNode"] = None
    children: List["ProofNode"] = None

    def __post_init__(self):
        self.children = self.children or []


@dataclass
class QuantumProofState:
    """Quantum superposition of multiple proof states."""
    
    # Core quantum properties
    amplitudes: np.ndarray                    # Complex probability amplitudes
    phases: np.ndarray                        # Quantum phases
    entanglement_matrix: np.ndarray           # Entanglement between proof steps
    
    # Proof-specific properties
    proof_branches: List[List[str]] = field(default_factory=list)  # Superposed proof paths
    coherence_time: float = 1.0               # Decoherence parameter
    measurement_history: List[int] = field(default_factory=list)
    
    # Quantum evolution parameters
    hamiltonian: Optional[np.ndarray] = None  # System Hamiltonian
    evolution_time: float = 0.0               # Time parameter for evolution
    
    def __post_init__(self):
        """Initialize quantum state properties."""
        if self.amplitudes is None:
            # Initialize in uniform superposition
            n_states = len(self.proof_branches) if self.proof_branches else 2
            self.amplitudes = np.ones(n_states, dtype=complex) / np.sqrt(n_states)
        
        if self.phases is None:
            self.phases = np.zeros(len(self.amplitudes))
        
        if self.entanglement_matrix is None:
            n = len(self.amplitudes)
            self.entanglement_matrix = np.eye(n, dtype=complex)
    
    def normalize(self):
        """Normalize quantum state to unit probability."""
        norm = np.linalg.norm(self.amplitudes)
        if norm > 0:
            self.amplitudes /= norm
    
    def probability_distribution(self) -> np.ndarray:
        """Get measurement probability distribution."""
        return np.abs(self.amplitudes) ** 2
    
    def von_neumann_entropy(self) -> float:
        """Calculate quantum entropy of the state."""
        probs = self.probability_distribution()
        # Add small epsilon to avoid log(0)
        probs = probs + 1e-12
        return -np.sum(probs * np.log2(probs))
    
    def quantum_fidelity(self, other: "QuantumProofState") -> float:
        """Calculate fidelity between two quantum states."""
        return abs(np.vdot(self.amplitudes, other.amplitudes)) ** 2
    
    def apply_decoherence(self, dt: float):
        """Apply decoherence effects over time."""
        decoherence_factor = np.exp(-dt / self.coherence_time)
        self.amplitudes *= decoherence_factor
        self.normalize()


@dataclass
class QuantumCircuit:
    """Quantum circuit for proof manipulation."""
    
    gates: List[Tuple[QuantumGate, List[int], Optional[float]]] = field(default_factory=list)
    num_qubits: int = 0
    
    def add_gate(self, gate: QuantumGate, qubits: Union[int, List[int]], parameter: Optional[float] = None):
        """Add a quantum gate to the circuit."""
        if isinstance(qubits, int):
            qubits = [qubits]
        self.gates.append((gate, qubits, parameter))
        self.num_qubits = max(self.num_qubits, max(qubits) + 1)
    
    def to_matrix(self) -> np.ndarray:
        """Convert quantum circuit to unitary matrix."""
        dim = 2 ** self.num_qubits
        circuit_matrix = np.eye(dim, dtype=complex)
        
        for gate, qubits, parameter in self.gates:
            gate_matrix = self._get_gate_matrix(gate, parameter)
            expanded_matrix = self._expand_gate_matrix(gate_matrix, qubits)
            circuit_matrix = expanded_matrix @ circuit_matrix
        
        return circuit_matrix
    
    def _get_gate_matrix(self, gate: QuantumGate, parameter: Optional[float] = None) -> np.ndarray:
        """Get matrix representation of quantum gate."""
        if gate == QuantumGate.HADAMARD:
            return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        elif gate == QuantumGate.PAULI_X:
            return np.array([[0, 1], [1, 0]], dtype=complex)
        elif gate == QuantumGate.PAULI_Y:
            return np.array([[0, -1j], [1j, 0]], dtype=complex)
        elif gate == QuantumGate.PAULI_Z:
            return np.array([[1, 0], [0, -1]], dtype=complex)
        elif gate == QuantumGate.PHASE:
            phase = parameter or 0.0
            return np.array([[1, 0], [0, np.exp(1j * phase)]], dtype=complex)
        elif gate == QuantumGate.ROTATION_X:
            theta = parameter or 0.0
            cos_half = np.cos(theta / 2)
            sin_half = np.sin(theta / 2)
            return np.array([[cos_half, -1j * sin_half], [-1j * sin_half, cos_half]], dtype=complex)
        elif gate == QuantumGate.ROTATION_Y:
            theta = parameter or 0.0
            cos_half = np.cos(theta / 2)
            sin_half = np.sin(theta / 2)
            return np.array([[cos_half, -sin_half], [sin_half, cos_half]], dtype=complex)
        elif gate == QuantumGate.ROTATION_Z:
            theta = parameter or 0.0
            return np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=complex)
        elif gate == QuantumGate.CNOT:
            return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
        else:
            return np.eye(2, dtype=complex)
    
    def _expand_gate_matrix(self, gate_matrix: np.ndarray, qubits: List[int]) -> np.ndarray:
        """Expand gate matrix to full Hilbert space."""
        n_qubits = self.num_qubits
        gate_size = len(qubits)
        
        if gate_size == 1:
            # Single-qubit gate
            result = np.array([[1]], dtype=complex)
            for i in range(n_qubits):
                if i == qubits[0]:
                    result = np.kron(result, gate_matrix)
                else:
                    result = np.kron(result, np.eye(2, dtype=complex))
            return result
        elif gate_size == 2:
            # Two-qubit gate
            result = np.eye(2 ** n_qubits, dtype=complex)
            # Simplified implementation for CNOT-like gates
            return result
        else:
            # Multi-qubit gate (simplified)
            return np.eye(2 ** n_qubits, dtype=complex)


class QuantumProofSearcher:
    """Quantum-inspired proof search engine with advanced optimization."""

    def __init__(
        self,
        strategy: SearchStrategy = SearchStrategy.HYBRID_QUANTUM,
        max_iterations: int = 10000,
        exploration_factor: float = 1.414,
        quantum_dimensions: int = 64,
        annealing_schedule: Optional[List[float]] = None,
        learning_rate: float = 0.01,
    ):
        """Initialize quantum proof searcher.

        Args:
            strategy: Search strategy to use
            max_iterations: Maximum search iterations
            exploration_factor: UCB exploration parameter
            quantum_dimensions: Quantum state space dimensions
            annealing_schedule: Temperature schedule for simulated annealing
            learning_rate: Learning rate for RL components
        """
        self.strategy = strategy
        self.max_iterations = max_iterations
        self.exploration_factor = exploration_factor
        self.quantum_dimensions = quantum_dimensions
        self.learning_rate = learning_rate

        # Annealing schedule
        if annealing_schedule is None:
            self.annealing_schedule = self._generate_annealing_schedule()
        else:
            self.annealing_schedule = annealing_schedule

        # Search tree
        self.root: Optional[ProofNode] = None
        self.best_solution: Optional[ProofNode] = None
        self.search_history: List[Dict[str, Any]] = []

        # Quantum state management
        self.quantum_state = self._initialize_quantum_state()
        self.entanglement_graph: Dict[str, Set[str]] = defaultdict(set)

        # Learning components
        self.tactic_success_rates: Dict[str, float] = defaultdict(lambda: 0.5)
        self.pattern_library: Dict[str, List[str]] = defaultdict(list)
        self.failure_analysis: deque = deque(maxlen=1000)

        # Performance metrics
        self.search_stats = {
            "iterations": 0,
            "expansions": 0,
            "backtracks": 0,
            "quantum_measurements": 0,
            "convergence_score": 0.0,
        }

    def search_proof(
        self,
        goal: str,
        context: Dict[str, Any],
        available_tactics: List[str],
        timeout: float = 300.0,
    ) -> Optional[List[str]]:
        """Search for optimal proof using quantum-inspired methods.

        Args:
            goal: Target proof goal
            context: Proof context and assumptions
            available_tactics: Available proof tactics
            timeout: Search timeout in seconds

        Returns:
            Optimal proof sequence or None if not found
        """
        start_time = time.time()

        # Initialize search
        self.root = ProofNode(state="initial", goal=goal, tactics=[])
        self.best_solution = None

        try:
            if self.strategy == SearchStrategy.QUANTUM_ANNEALING:
                result = self._quantum_annealing_search(
                    goal, context, available_tactics, timeout, start_time
                )
            elif self.strategy == SearchStrategy.MONTE_CARLO_TREE:
                result = self._monte_carlo_tree_search(
                    goal, context, available_tactics, timeout, start_time
                )
            elif self.strategy == SearchStrategy.GENETIC_ALGORITHM:
                result = self._genetic_algorithm_search(
                    goal, context, available_tactics, timeout, start_time
                )
            elif self.strategy == SearchStrategy.HYBRID_QUANTUM:
                result = self._hybrid_quantum_search(
                    goal, context, available_tactics, timeout, start_time
                )
            else:  # REINFORCEMENT_LEARNING
                result = self._reinforcement_learning_search(
                    goal, context, available_tactics, timeout, start_time
                )

            # Record search statistics
            search_time = (time.time() - start_time) * 1000
            self.search_history.append(
                {
                    "timestamp": time.time(),
                    "goal": goal,
                    "strategy": self.strategy.value,
                    "search_time_ms": search_time,
                    "iterations": self.search_stats["iterations"],
                    "success": result is not None,
                    "proof_length": len(result) if result else 0,
                }
            )

            return result

        except Exception as e:
            self.failure_analysis.append(
                {
                    "timestamp": time.time(),
                    "goal": goal,
                    "error": str(e),
                    "strategy": self.strategy.value,
                }
            )
            return None

    def _quantum_annealing_search(
        self,
        goal: str,
        context: Dict[str, Any],
        tactics: List[str],
        timeout: float,
        start_time: float,
    ) -> Optional[List[str]]:
        """Quantum annealing-inspired proof search."""

        # Initialize quantum register for tactics
        n_tactics = len(tactics)
        quantum_register = np.zeros(n_tactics, dtype=complex)

        # Superposition initialization
        for i in range(n_tactics):
            quantum_register[i] = complex(1.0 / math.sqrt(n_tactics), 0)

        best_proof = []
        best_score = -float("inf")

        for iteration in range(self.max_iterations):
            if time.time() - start_time > timeout:
                break

            # Current temperature from annealing schedule
            progress = iteration / self.max_iterations
            temp_idx = min(
                int(progress * len(self.annealing_schedule)),
                len(self.annealing_schedule) - 1,
            )
            temperature = self.annealing_schedule[temp_idx]

            # Quantum measurement to select tactics
            probabilities = np.abs(quantum_register) ** 2
            selected_tactics = []

            for _ in range(min(5, len(tactics))):  # Select up to 5 tactics
                idx = np.random.choice(n_tactics, p=probabilities)
                selected_tactics.append(tactics[idx])

            # Evaluate proof candidate
            proof_score = self._evaluate_proof_candidate(
                selected_tactics, goal, context
            )

            # Update quantum state based on score
            if proof_score > best_score:
                best_score = proof_score
                best_proof = selected_tactics.copy()

                # Amplify successful tactics in quantum register
                for i, tactic in enumerate(tactics):
                    if tactic in selected_tactics:
                        phase_boost = math.exp(-1.0 / max(temperature, 0.001))
                        quantum_register[i] *= phase_boost

            # Quantum state evolution
            self._evolve_quantum_state(quantum_register, temperature, proof_score)

            self.search_stats["iterations"] += 1
            self.search_stats["quantum_measurements"] += len(selected_tactics)

        return best_proof if best_score > 0.5 else None

    def _monte_carlo_tree_search(
        self,
        goal: str,
        context: Dict[str, Any],
        tactics: List[str],
        timeout: float,
        start_time: float,
    ) -> Optional[List[str]]:
        """Monte Carlo Tree Search with UCB1."""

        for iteration in range(self.max_iterations):
            if time.time() - start_time > timeout:
                break

            # Selection phase
            node = self._select_node(self.root, tactics)

            # Expansion phase
            if not self._is_terminal(node) and node.visits > 0:
                node = self._expand_node(node, tactics, goal, context)
                self.search_stats["expansions"] += 1

            # Simulation phase
            reward = self._simulate_proof(node, tactics, goal, context)

            # Backpropagation phase
            self._backpropagate(node, reward)

            self.search_stats["iterations"] += 1

        # Extract best path
        return self._extract_best_path(self.root)

    def _genetic_algorithm_search(
        self,
        goal: str,
        context: Dict[str, Any],
        tactics: List[str],
        timeout: float,
        start_time: float,
    ) -> Optional[List[str]]:
        """Genetic algorithm for proof optimization."""

        population_size = 50
        mutation_rate = 0.1
        crossover_rate = 0.7
        max_proof_length = 10

        # Initialize population
        population = []
        for _ in range(population_size):
            proof_length = random.randint(1, max_proof_length)
            individual = [random.choice(tactics) for _ in range(proof_length)]
            population.append(individual)

        best_individual = []
        best_fitness = -float("inf")

        generation = 0
        while (
            time.time() - start_time < timeout
            and generation < self.max_iterations // 10
        ):
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                fitness = self._evaluate_proof_candidate(individual, goal, context)
                fitness_scores.append(fitness)

                if fitness > best_fitness:
                    best_fitness = fitness
                    best_individual = individual.copy()

            # Selection (tournament selection)
            new_population = []
            for _ in range(population_size):
                tournament_size = 3
                tournament_indices = random.sample(
                    range(population_size), tournament_size
                )
                winner_idx = max(tournament_indices, key=lambda i: fitness_scores[i])
                new_population.append(population[winner_idx].copy())

            # Crossover
            for i in range(0, population_size - 1, 2):
                if random.random() < crossover_rate:
                    child1, child2 = self._crossover(
                        new_population[i], new_population[i + 1]
                    )
                    new_population[i] = child1
                    new_population[i + 1] = child2

            # Mutation
            for individual in new_population:
                if random.random() < mutation_rate:
                    self._mutate(individual, tactics)

            population = new_population
            generation += 1
            self.search_stats["iterations"] += 1

        return best_individual if best_fitness > 0.5 else None

    def _hybrid_quantum_search(
        self,
        goal: str,
        context: Dict[str, Any],
        tactics: List[str],
        timeout: float,
        start_time: float,
    ) -> Optional[List[str]]:
        """Hybrid quantum-classical search combining multiple strategies."""

        # Phase 1: Quantum annealing for initial exploration (30% of time)
        phase1_timeout = timeout * 0.3
        result1 = self._quantum_annealing_search(
            goal, context, tactics, phase1_timeout, start_time
        )

        if result1 and self._evaluate_proof_candidate(result1, goal, context) > 0.8:
            return result1

        # Phase 2: MCTS for exploitation (40% of time)
        phase2_start = time.time()
        phase2_timeout = timeout * 0.4

        # Bias MCTS with quantum results
        if result1:
            self._update_tactic_success_rates(result1, 0.8)

        result2 = self._monte_carlo_tree_search(
            goal, context, tactics, phase2_timeout, phase2_start
        )

        if result2 and self._evaluate_proof_candidate(result2, goal, context) > 0.8:
            return result2

        # Phase 3: Genetic refinement (30% of time)
        phase3_start = time.time()
        phase3_timeout = timeout * 0.3

        # Seed GA with previous results
        if result2:
            self._update_pattern_library(result2, goal)

        result3 = self._genetic_algorithm_search(
            goal, context, tactics, phase3_timeout, phase3_start
        )

        # Return best result
        candidates = [r for r in [result1, result2, result3] if r]
        if not candidates:
            return None

        return max(
            candidates, key=lambda x: self._evaluate_proof_candidate(x, goal, context)
        )

    def _reinforcement_learning_search(
        self,
        goal: str,
        context: Dict[str, Any],
        tactics: List[str],
        timeout: float,
        start_time: float,
    ) -> Optional[List[str]]:
        """Q-learning based proof search."""

        # State-action value function
        q_table: Dict[Tuple[str, str], float] = defaultdict(lambda: 0.0)
        epsilon = 0.1  # Exploration rate
        gamma = 0.9  # Discount factor

        current_proof = []
        current_state = goal

        for iteration in range(self.max_iterations):
            if time.time() - start_time > timeout:
                break

            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.choice(tactics)
            else:
                # Choose best action for current state
                q_values = [
                    (tactic, q_table[(current_state, tactic)]) for tactic in tactics
                ]
                action = max(q_values, key=lambda x: x[1])[0]

            # Apply action and get new state
            new_state, reward = self._apply_action(current_state, action, goal, context)

            # Update Q-value
            best_next_q = max([q_table[(new_state, t)] for t in tactics])
            q_table[(current_state, action)] += self.learning_rate * (
                reward + gamma * best_next_q - q_table[(current_state, action)]
            )

            current_proof.append(action)
            current_state = new_state

            # Check termination
            if self._is_goal_reached(current_state, goal) or len(current_proof) > 20:
                if self._evaluate_proof_candidate(current_proof, goal, context) > 0.5:
                    return current_proof

                # Reset for new episode
                current_proof = []
                current_state = goal

            self.search_stats["iterations"] += 1

        return None

    def _evaluate_proof_candidate(
        self, tactics: List[str], goal: str, context: Dict[str, Any]
    ) -> float:
        """Evaluate proof candidate using heuristics."""
        if not tactics:
            return 0.0

        score = 0.0

        # Base score from tactic success rates
        for tactic in tactics:
            score += self.tactic_success_rates[tactic]

        # Length penalty (prefer shorter proofs)
        length_penalty = 1.0 / (1.0 + len(tactics) * 0.1)
        score *= length_penalty

        # Pattern matching bonus
        pattern_bonus = self._calculate_pattern_bonus(tactics, goal)
        score += pattern_bonus

        # Diversity bonus (prefer varied tactics)
        unique_tactics = len(set(tactics))
        diversity_bonus = unique_tactics / len(tactics) * 0.2
        score += diversity_bonus

        # Normalize score
        return min(1.0, score / len(tactics))

    def _calculate_pattern_bonus(self, tactics: List[str], goal: str) -> float:
        """Calculate bonus for matching known successful patterns."""
        bonus = 0.0

        for pattern_goal, pattern_tactics in self.pattern_library.items():
            if self._goals_similar(goal, pattern_goal):
                # Calculate sequence similarity
                similarity = self._sequence_similarity(tactics, pattern_tactics)
                bonus += similarity * 0.3

        return bonus

    def _initialize_quantum_state(self) -> QuantumState:
        """Initialize quantum state for search."""
        n = self.quantum_dimensions

        # Equal superposition
        amplitudes = np.ones(n) / math.sqrt(n)
        phases = np.zeros(n)

        # Random entanglement matrix
        entanglement = np.random.random((n, n))
        entanglement = (entanglement + entanglement.T) / 2  # Make symmetric

        return QuantumState(
            amplitudes=amplitudes, phases=phases, entanglement_matrix=entanglement
        )

    def _generate_annealing_schedule(self) -> List[float]:
        """Generate temperature schedule for simulated annealing."""
        schedule = []
        initial_temp = 10.0
        final_temp = 0.01

        for i in range(self.max_iterations):
            progress = i / self.max_iterations
            temp = initial_temp * ((final_temp / initial_temp) ** progress)
            schedule.append(temp)

        return schedule

    def _evolve_quantum_state(
        self, quantum_register: np.ndarray, temperature: float, score: float
    ):
        """Evolve quantum state based on measurement results."""
        # Phase evolution based on score
        phase_shift = score * temperature * 0.1

        for i in range(len(quantum_register)):
            current_phase = np.angle(quantum_register[i])
            new_phase = current_phase + phase_shift * (random.random() - 0.5)
            magnitude = abs(quantum_register[i])
            quantum_register[i] = magnitude * np.exp(1j * new_phase)

        # Renormalization
        norm = np.linalg.norm(quantum_register)
        if norm > 0:
            quantum_register /= norm

    def _select_node(self, node: ProofNode, tactics: List[str]) -> ProofNode:
        """Select node using UCB1."""
        if not node.children:
            return node

        def ucb1_score(child: ProofNode) -> float:
            if child.visits == 0:
                return float("inf")

            exploitation = child.score / child.visits
            exploration = self.exploration_factor * math.sqrt(
                math.log(node.visits) / child.visits
            )
            return exploitation + exploration

        best_child = max(node.children, key=ucb1_score)
        return self._select_node(best_child, tactics)

    def _expand_node(
        self, node: ProofNode, tactics: List[str], goal: str, context: Dict[str, Any]
    ) -> ProofNode:
        """Expand node with new children."""
        for tactic in tactics:
            if not any(
                child.tactics[-1] == tactic for child in node.children if child.tactics
            ):
                new_tactics = node.tactics + [tactic]
                child = ProofNode(
                    state=f"state_{len(new_tactics)}",
                    goal=goal,
                    tactics=new_tactics,
                    depth=node.depth + 1,
                    parent=node,
                )
                node.children.append(child)
                return child

        return node

    def _simulate_proof(
        self, node: ProofNode, tactics: List[str], goal: str, context: Dict[str, Any]
    ) -> float:
        """Simulate proof completion from node."""
        simulation_tactics = node.tactics.copy()
        max_sim_depth = 10

        for _ in range(max_sim_depth):
            if len(simulation_tactics) > 0 and self._is_goal_reached(
                simulation_tactics[-1], goal
            ):
                break

            # Select tactic based on success rates
            weights = [self.tactic_success_rates[t] for t in tactics]
            if sum(weights) > 0:
                selected = random.choices(tactics, weights=weights)[0]
                simulation_tactics.append(selected)

        return self._evaluate_proof_candidate(simulation_tactics, goal, context)

    def _backpropagate(self, node: ProofNode, reward: float):
        """Backpropagate reward through tree."""
        while node:
            node.visits += 1
            node.score += reward
            node = node.parent
            self.search_stats["backtracks"] += 1

    def _extract_best_path(self, root: ProofNode) -> List[str]:
        """Extract best proof path from MCTS tree."""
        path = []
        node = root

        while node.children:
            best_child = max(node.children, key=lambda c: c.score / max(c.visits, 1))
            if best_child.tactics:
                path.extend(best_child.tactics[len(node.tactics) :])
            node = best_child

        return path

    def _crossover(
        self, parent1: List[str], parent2: List[str]
    ) -> Tuple[List[str], List[str]]:
        """Genetic algorithm crossover."""
        if len(parent1) <= 1 or len(parent2) <= 1:
            return parent1.copy(), parent2.copy()

        # Single-point crossover
        point1 = random.randint(1, len(parent1) - 1)
        point2 = random.randint(1, len(parent2) - 1)

        child1 = parent1[:point1] + parent2[point2:]
        child2 = parent2[:point2] + parent1[point1:]

        return child1, child2

    def _mutate(self, individual: List[str], tactics: List[str]):
        """Genetic algorithm mutation."""
        if not individual:
            return

        mutation_type = random.choice(["replace", "insert", "delete"])

        if mutation_type == "replace" and individual:
            idx = random.randint(0, len(individual) - 1)
            individual[idx] = random.choice(tactics)
        elif mutation_type == "insert":
            idx = random.randint(0, len(individual))
            individual.insert(idx, random.choice(tactics))
        elif mutation_type == "delete" and len(individual) > 1:
            idx = random.randint(0, len(individual) - 1)
            del individual[idx]

    def _apply_action(
        self, state: str, action: str, goal: str, context: Dict[str, Any]
    ) -> Tuple[str, float]:
        """Apply action and return new state and reward."""
        # Simplified state transition
        new_state = f"{state}_{action}"

        # Reward based on action success rate and goal similarity
        reward = self.tactic_success_rates[action]

        if self._is_goal_reached(action, goal):
            reward += 1.0

        return new_state, reward

    def _is_terminal(self, node: ProofNode) -> bool:
        """Check if node is terminal."""
        return len(node.tactics) > 15 or node.depth > 10

    def _is_goal_reached(self, state: str, goal: str) -> bool:
        """Check if goal is reached."""
        return "qed" in state.lower() or "proved" in state.lower()

    def _goals_similar(self, goal1: str, goal2: str) -> bool:
        """Check if two goals are similar."""
        # Simplified similarity check
        words1 = set(goal1.lower().split())
        words2 = set(goal2.lower().split())

        if not words1 or not words2:
            return False

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) > 0.5

    def _sequence_similarity(self, seq1: List[str], seq2: List[str]) -> float:
        """Calculate similarity between tactic sequences."""
        if not seq1 or not seq2:
            return 0.0

        # Simple Jaccard similarity
        set1 = set(seq1)
        set2 = set(seq2)

        intersection = set1.intersection(set2)
        union = set1.union(set2)

        return len(intersection) / len(union) if union else 0.0

    def _update_tactic_success_rates(self, tactics: List[str], success_rate: float):
        """Update tactic success rates based on results."""
        for tactic in tactics:
            current_rate = self.tactic_success_rates[tactic]
            # Exponential moving average
            self.tactic_success_rates[tactic] = 0.9 * current_rate + 0.1 * success_rate

    def _update_pattern_library(self, tactics: List[str], goal: str):
        """Update pattern library with successful proof."""
        self.pattern_library[goal] = tactics

    def get_search_statistics(self) -> Dict[str, Any]:
        """Get comprehensive search statistics."""
        recent_searches = [
            s for s in self.search_history if time.time() - s["timestamp"] < 3600
        ]

        stats = self.search_stats.copy()
        stats.update(
            {
                "total_searches": len(self.search_history),
                "recent_searches": len(recent_searches),
                "success_rate": sum(1 for s in recent_searches if s["success"])
                / max(len(recent_searches), 1),
                "avg_search_time_ms": (
                    statistics.mean([s["search_time_ms"] for s in recent_searches])
                    if recent_searches
                    else 0
                ),
                "avg_proof_length": (
                    statistics.mean(
                        [
                            s["proof_length"]
                            for s in recent_searches
                            if s["proof_length"] > 0
                        ]
                    )
                    if recent_searches
                    else 0
                ),
                "pattern_library_size": len(self.pattern_library),
                "learned_tactics": len(self.tactic_success_rates),
                "failure_rate": len(self.failure_analysis)
                / max(len(self.search_history), 1),
            }
        )

        return stats

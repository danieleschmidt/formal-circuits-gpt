"""Tests for quantum-inspired proof search optimization."""

import pytest
import time
from unittest.mock import Mock, patch
import numpy as np

from src.formal_circuits_gpt.optimization.quantum_proof_search import (
    QuantumProofSearcher,
    SearchStrategy,
    ProofNode,
    QuantumState
)


class TestQuantumProofSearcher:
    """Test quantum-inspired proof search optimization."""
    
    @pytest.fixture
    def searcher(self):
        """Create quantum proof searcher for testing."""
        return QuantumProofSearcher(
            strategy=SearchStrategy.QUANTUM_ANNEALING,
            max_iterations=100,  # Reduced for faster tests
            quantum_dimensions=16
        )
    
    @pytest.fixture
    def sample_tactics(self):
        """Sample proof tactics for testing."""
        return ["intro", "apply", "destruct", "induction", "simpl", "auto"]
    
    @pytest.fixture
    def sample_context(self):
        """Sample proof context."""
        return {
            "prover": "isabelle",
            "model": "gpt-4-turbo",
            "timeout": 300
        }
    
    def test_searcher_initialization(self):
        """Test quantum searcher initialization."""
        searcher = QuantumProofSearcher()
        
        assert searcher.strategy == SearchStrategy.HYBRID_QUANTUM
        assert searcher.max_iterations == 10000
        assert searcher.quantum_dimensions == 64
        assert searcher.quantum_state is not None
        assert isinstance(searcher.quantum_state.amplitudes, np.ndarray)
    
    def test_quantum_state_initialization(self, searcher):
        """Test quantum state initialization."""
        state = searcher.quantum_state
        
        assert len(state.amplitudes) == searcher.quantum_dimensions
        assert len(state.phases) == searcher.quantum_dimensions
        assert state.entanglement_matrix.shape == (searcher.quantum_dimensions, searcher.quantum_dimensions)
        
        # Check normalization
        norm = np.linalg.norm(state.amplitudes)
        assert abs(norm - 1.0) < 1e-10
    
    def test_annealing_schedule_generation(self, searcher):
        """Test annealing schedule generation."""
        schedule = searcher._generate_annealing_schedule()
        
        assert len(schedule) == searcher.max_iterations
        assert schedule[0] > schedule[-1]  # Temperature decreases
        assert all(temp > 0 for temp in schedule)  # All positive temperatures
    
    def test_quantum_annealing_search(self, searcher, sample_tactics, sample_context):
        """Test quantum annealing search algorithm."""
        goal = "forall x, x + 0 = x"
        
        with patch.object(searcher, '_evaluate_proof_candidate', return_value=0.8):
            result = searcher._quantum_annealing_search(
                goal, sample_context, sample_tactics, 1.0, time.time()
            )
        
        assert result is not None
        assert isinstance(result, list)
        assert all(tactic in sample_tactics for tactic in result)
    
    def test_monte_carlo_tree_search(self, searcher, sample_tactics, sample_context):
        """Test Monte Carlo Tree Search algorithm."""
        goal = "forall x, x + 0 = x"
        
        # Initialize root
        searcher.root = ProofNode(state="initial", goal=goal, tactics=[])
        
        with patch.object(searcher, '_evaluate_proof_candidate', return_value=0.7):
            result = searcher._monte_carlo_tree_search(
                goal, sample_context, sample_tactics, 0.5, time.time()
            )
        
        assert isinstance(result, list)
    
    def test_genetic_algorithm_search(self, searcher, sample_tactics, sample_context):
        """Test genetic algorithm search."""
        goal = "forall x, x + 0 = x"
        
        with patch.object(searcher, '_evaluate_proof_candidate') as mock_eval:
            mock_eval.side_effect = lambda tactics, goal, ctx: len(tactics) * 0.1 + 0.3
            
            result = searcher._genetic_algorithm_search(
                goal, sample_context, sample_tactics, 0.5, time.time()
            )
        
        assert isinstance(result, list)
    
    def test_hybrid_quantum_search(self, searcher, sample_tactics, sample_context):
        """Test hybrid quantum search combining multiple strategies."""
        goal = "forall x, x + 0 = x"
        
        with patch.object(searcher, '_evaluate_proof_candidate', return_value=0.6):
            result = searcher._hybrid_quantum_search(
                goal, sample_context, sample_tactics, 1.5, time.time()
            )
        
        assert isinstance(result, list)
    
    def test_reinforcement_learning_search(self, searcher, sample_tactics, sample_context):
        """Test reinforcement learning search."""
        goal = "forall x, x + 0 = x"
        
        with patch.object(searcher, '_apply_action') as mock_action:
            mock_action.return_value = ("new_state", 0.5)
            
            with patch.object(searcher, '_is_goal_reached', return_value=True):
                result = searcher._reinforcement_learning_search(
                    goal, sample_context, sample_tactics, 0.5, time.time()
                )
        
        assert isinstance(result, list)
    
    def test_proof_evaluation(self, searcher, sample_tactics, sample_context):
        """Test proof candidate evaluation."""
        goal = "forall x, x + 0 = x"
        
        # Test with valid tactics
        score1 = searcher._evaluate_proof_candidate(sample_tactics[:3], goal, sample_context)
        assert 0 <= score1 <= 1
        
        # Test with empty tactics
        score2 = searcher._evaluate_proof_candidate([], goal, sample_context)
        assert score2 == 0.0
        
        # Test with single tactic
        score3 = searcher._evaluate_proof_candidate(["intro"], goal, sample_context)
        assert 0 <= score3 <= 1
    
    def test_pattern_bonus_calculation(self, searcher):
        """Test pattern matching bonus calculation."""
        tactics1 = ["intro", "apply", "auto"]
        tactics2 = ["intro", "destruct", "simpl"]
        goal = "forall x, x = x"
        
        # Add pattern to library
        searcher.pattern_library[goal] = tactics1
        
        # Test bonus calculation
        bonus1 = searcher._calculate_pattern_bonus(tactics1, goal)
        bonus2 = searcher._calculate_pattern_bonus(tactics2, goal)
        
        assert bonus1 >= bonus2  # Exact match should have higher bonus
    
    def test_quantum_state_evolution(self, searcher):
        """Test quantum state evolution."""
        register = np.array([0.5+0j, 0.5+0j, 0.0+0j, 0.0+0j])
        initial_norm = np.linalg.norm(register)
        
        searcher._evolve_quantum_state(register, temperature=1.0, score=0.8)
        
        final_norm = np.linalg.norm(register)
        assert abs(initial_norm - final_norm) < 1e-10  # Norm preserved
    
    def test_ucb1_node_selection(self, searcher, sample_tactics):
        """Test UCB1 node selection."""
        root = ProofNode(state="root", goal="test", tactics=[], visits=10)
        
        # Create children with different scores
        child1 = ProofNode(state="child1", goal="test", tactics=["intro"], visits=5, score=2.0)
        child2 = ProofNode(state="child2", goal="test", tactics=["apply"], visits=3, score=1.0)
        child3 = ProofNode(state="child3", goal="test", tactics=["auto"], visits=0, score=0.0)
        
        root.children = [child1, child2, child3]
        child1.parent = root
        child2.parent = root
        child3.parent = root
        
        selected = searcher._select_node(root, sample_tactics)
        
        # Unvisited node (child3) should be selected due to infinite UCB1 value
        assert selected == child3 or selected.visits == 0
    
    def test_node_expansion(self, searcher, sample_tactics):
        """Test node expansion."""
        node = ProofNode(state="test", goal="forall x, x = x", tactics=["intro"])
        
        expanded = searcher._expand_node(node, sample_tactics, "test_goal", {})
        
        assert expanded.parent == node
        assert len(expanded.tactics) == len(node.tactics) + 1
        assert expanded in node.children
    
    def test_proof_simulation(self, searcher, sample_tactics, sample_context):
        """Test proof simulation."""
        node = ProofNode(state="test", goal="test", tactics=["intro"])
        
        with patch.object(searcher, '_is_goal_reached', return_value=False):
            reward = searcher._simulate_proof(node, sample_tactics, "test", sample_context)
        
        assert 0 <= reward <= 1
    
    def test_backpropagation(self, searcher):
        """Test reward backpropagation."""
        # Create simple tree
        root = ProofNode(state="root", goal="test", tactics=[])
        child = ProofNode(state="child", goal="test", tactics=["intro"], parent=root)
        grandchild = ProofNode(state="grandchild", goal="test", tactics=["intro", "apply"], parent=child)
        
        initial_visits = [root.visits, child.visits, grandchild.visits]
        initial_scores = [root.score, child.score, grandchild.score]
        
        searcher._backpropagate(grandchild, 0.8)
        
        # Check that visits and scores were updated
        assert root.visits > initial_visits[0]
        assert child.visits > initial_visits[1]
        assert grandchild.visits > initial_visits[2]
        
        assert root.score > initial_scores[0]
        assert child.score > initial_scores[1]
        assert grandchild.score > initial_scores[2]
    
    def test_genetic_crossover(self, searcher):
        """Test genetic algorithm crossover."""
        parent1 = ["intro", "apply", "auto"]
        parent2 = ["destruct", "simpl", "trivial"]
        
        child1, child2 = searcher._crossover(parent1, parent2)
        
        # Children should be different from parents
        assert child1 != parent1 or child2 != parent2
        
        # Children should contain elements from both parents
        all_elements = set(parent1 + parent2)
        child1_elements = set(child1)
        child2_elements = set(child2)
        
        assert child1_elements.issubset(all_elements)
        assert child2_elements.issubset(all_elements)
    
    def test_genetic_mutation(self, searcher, sample_tactics):
        """Test genetic algorithm mutation."""
        original = ["intro", "apply", "auto"]
        individual = original.copy()
        
        searcher._mutate(individual, sample_tactics)
        
        # Individual might be changed
        # We can't guarantee change due to randomness, but structure should be valid
        assert isinstance(individual, list)
        assert all(tactic in sample_tactics for tactic in individual)
    
    def test_tactic_success_rate_updates(self, searcher):
        """Test tactic success rate updates."""
        tactics = ["intro", "apply", "auto"]
        initial_rates = {tactic: searcher.tactic_success_rates[tactic] for tactic in tactics}
        
        searcher._update_tactic_success_rates(tactics, 0.9)
        
        # Success rates should be updated
        for tactic in tactics:
            new_rate = searcher.tactic_success_rates[tactic]
            assert new_rate != initial_rates[tactic]
            assert 0 <= new_rate <= 1
    
    def test_pattern_library_updates(self, searcher):
        """Test pattern library updates."""
        tactics = ["intro", "apply", "qed"]
        goal = "simple proof goal"
        
        initial_size = len(searcher.pattern_library)
        searcher._update_pattern_library(tactics, goal)
        
        assert len(searcher.pattern_library) >= initial_size
        assert goal in searcher.pattern_library
        assert searcher.pattern_library[goal] == tactics
    
    def test_search_statistics(self, searcher):
        """Test search statistics collection."""
        # Add some mock history
        searcher.search_history = [
            {
                "timestamp": time.time(),
                "goal": "test1",
                "strategy": "quantum_annealing",
                "search_time_ms": 100,
                "iterations": 50,
                "success": True,
                "proof_length": 3
            },
            {
                "timestamp": time.time() - 3600,  # Old entry
                "goal": "test2", 
                "strategy": "monte_carlo",
                "search_time_ms": 200,
                "iterations": 80,
                "success": False,
                "proof_length": 0
            }
        ]
        
        stats = searcher.get_search_statistics()
        
        assert "total_searches" in stats
        assert "recent_searches" in stats
        assert "success_rate" in stats
        assert "avg_search_time_ms" in stats
        
        # Recent searches should filter out old entries
        assert stats["recent_searches"] <= stats["total_searches"]
    
    def test_integration_search_proof(self, searcher, sample_tactics, sample_context):
        """Test full proof search integration."""
        goal = "forall x : nat, x + 0 = x"
        
        # Mock evaluation to return reasonable scores
        def mock_evaluation(tactics, goal, context):
            if not tactics:
                return 0.0
            if "intro" in tactics and "simpl" in tactics:
                return 0.9  # High score for good combination
            return 0.5  # Moderate score otherwise
        
        with patch.object(searcher, '_evaluate_proof_candidate', side_effect=mock_evaluation):
            result = searcher.search_proof(goal, sample_context, sample_tactics, timeout=2.0)
        
        # Should return a result (might be None if search fails)
        assert result is None or isinstance(result, list)
        
        if result:
            assert all(tactic in sample_tactics for tactic in result)
    
    def test_different_strategies(self, sample_tactics, sample_context):
        """Test all different search strategies."""
        goal = "simple goal"
        
        strategies = [
            SearchStrategy.QUANTUM_ANNEALING,
            SearchStrategy.MONTE_CARLO_TREE,
            SearchStrategy.GENETIC_ALGORITHM,
            SearchStrategy.HYBRID_QUANTUM,
            SearchStrategy.REINFORCEMENT_LEARNING
        ]
        
        for strategy in strategies:
            searcher = QuantumProofSearcher(
                strategy=strategy,
                max_iterations=10  # Very limited for speed
            )
            
            with patch.object(searcher, '_evaluate_proof_candidate', return_value=0.6):
                result = searcher.search_proof(goal, sample_context, sample_tactics, timeout=0.5)
            
            # Each strategy should complete without error
            assert result is None or isinstance(result, list)
    
    def test_error_handling(self, searcher, sample_tactics, sample_context):
        """Test error handling in search."""
        goal = "problematic goal"
        
        # Mock evaluation to raise exception
        with patch.object(searcher, '_evaluate_proof_candidate', side_effect=Exception("Test error")):
            result = searcher.search_proof(goal, sample_context, sample_tactics, timeout=1.0)
        
        # Should handle error gracefully
        assert result is None
        assert len(searcher.failure_analysis) > 0
    
    @pytest.mark.parametrize("quantum_dims", [8, 16, 32, 64])
    def test_different_quantum_dimensions(self, quantum_dims, sample_tactics, sample_context):
        """Test searcher with different quantum dimensions."""
        searcher = QuantumProofSearcher(
            quantum_dimensions=quantum_dims,
            max_iterations=10
        )
        
        assert len(searcher.quantum_state.amplitudes) == quantum_dims
        
        # Should work with any reasonable dimension size
        goal = "test goal"
        with patch.object(searcher, '_evaluate_proof_candidate', return_value=0.5):
            result = searcher.search_proof(goal, sample_context, sample_tactics, timeout=0.5)
        
        assert result is None or isinstance(result, list)
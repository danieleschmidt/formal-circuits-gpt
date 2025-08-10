"""Tests for ML-based proof optimization."""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch

from src.formal_circuits_gpt.optimization.ml_proof_optimizer import (
    MLProofOptimizer,
    OptimizationType,
    ProofFeatures,
    OptimizationResult,
    NeuralNetworkOptimizer,
    ReinforcementLearningOptimizer,
    EvolutionaryOptimizer,
    TransformerOptimizer
)


class TestMLProofOptimizer:
    """Test ML-based proof optimization."""
    
    @pytest.fixture
    def optimizer(self):
        """Create ML proof optimizer for testing."""
        return MLProofOptimizer(
            optimization_type=OptimizationType.ENSEMBLE,
            max_training_samples=100  # Reduced for testing
        )
    
    @pytest.fixture
    def sample_tactics(self):
        """Sample proof tactics for testing."""
        return ["intro", "apply H", "destruct x", "induction n", "simpl", "auto"]
    
    @pytest.fixture
    def sample_context(self):
        """Sample optimization context."""
        return {
            "prover": "isabelle",
            "model": "gpt-4-turbo",
            "timeout": 300
        }
    
    @pytest.fixture
    def sample_features(self):
        """Sample proof features."""
        return ProofFeatures(
            ast_depth=3,
            node_count=15,
            operator_frequencies={"apply": 2, "intro": 1},
            variable_count=4,
            quantifier_depth=2,
            formula_complexity=0.6,
            similar_proof_success_rate=0.7,
            tactic_sequence_length=6,
            prover_type="isabelle",
            model_type="gpt-4-turbo"
        )
    
    def test_optimizer_initialization(self):
        """Test ML optimizer initialization."""
        optimizer = MLProofOptimizer()
        
        assert optimizer.optimization_type == OptimizationType.ENSEMBLE
        assert len(optimizer.ensemble_weights) == 4
        assert sum(optimizer.ensemble_weights.values()) == pytest.approx(1.0)
        assert optimizer.neural_network is not None
        assert optimizer.rl_optimizer is not None
        assert optimizer.evolutionary_optimizer is not None
        assert optimizer.transformer_optimizer is not None
    
    def test_feature_extraction(self, optimizer, sample_tactics, sample_context):
        """Test feature extraction from proof data."""
        goal = "forall x : nat, x + 0 = x"
        
        features = optimizer._extract_features(sample_tactics, goal, sample_context)
        
        assert features.tactic_sequence_length == len(sample_tactics)
        assert features.prover_type == "isabelle"
        assert features.model_type == "gpt-4-turbo"
        assert features.timeout_seconds == 300
        assert features.variable_count > 0
        assert features.formula_complexity >= 0
    
    def test_operator_counting(self, optimizer):
        """Test operator frequency counting."""
        tactics = ["apply lemma1", "intro x", "apply lemma2", "simpl"]
        
        operators = optimizer._count_operators(tactics)
        
        assert operators["apply"] == 2
        assert "intro" in operators
        assert "simpl" in operators
    
    def test_variable_counting(self, optimizer):
        """Test variable counting in formulas."""
        goal1 = "forall x y : nat, x + y = y + x"
        goal2 = "exists z, z > 0"
        
        count1 = optimizer._count_variables(goal1)
        count2 = optimizer._count_variables(goal2)
        
        assert count1 >= 2  # At least x and y
        assert count2 >= 1  # At least z
    
    def test_quantifier_depth_calculation(self, optimizer):
        """Test quantifier depth calculation."""
        formula1 = "forall x, exists y, x < y"
        formula2 = "forall x, forall y, forall z, x + y + z = 0"
        
        depth1 = optimizer._calculate_quantifier_depth(formula1)
        depth2 = optimizer._calculate_quantifier_depth(formula2)
        
        assert depth1 == 2  # One forall, one exists
        assert depth2 == 3  # Three foralls
    
    def test_formula_complexity_calculation(self, optimizer):
        """Test formula complexity scoring."""
        simple_formula = "x = x"
        complex_formula = "forall x y z, (x + y) * z = (x * z) + (y * z) ∧ (x > 0 → y > 0)"
        
        complexity1 = optimizer._calculate_formula_complexity(simple_formula)
        complexity2 = optimizer._calculate_formula_complexity(complex_formula)
        
        assert complexity2 > complexity1  # Complex formula should score higher
        assert complexity1 >= 0
        assert complexity2 >= 0
    
    def test_parentheses_depth_calculation(self, optimizer):
        """Test parentheses nesting depth calculation."""
        formula1 = "x + y"
        formula2 = "(x + (y * z))"
        formula3 = "((x + y) * (z + w))"
        
        depth1 = optimizer._calculate_parentheses_depth(formula1)
        depth2 = optimizer._calculate_parentheses_depth(formula2)
        depth3 = optimizer._calculate_parentheses_depth(formula3)
        
        assert depth1 == 0
        assert depth2 == 2
        assert depth3 == 2
    
    def test_similar_proof_success_rate(self, optimizer):
        """Test similar proof success rate calculation."""
        # Add training samples
        optimizer.training_samples.extend([
            {"goal": "forall x, x + 0 = x", "success": True},
            {"goal": "forall y, y + 0 = y", "success": True},
            {"goal": "forall z, z * 1 = z", "success": False}
        ])
        
        goal = "forall a, a + 0 = a"
        success_rate = optimizer._get_similar_proof_success_rate(goal)
        
        # Should find similarity with first two samples
        assert 0.0 <= success_rate <= 1.0
        assert success_rate > 0.5  # Should be > 0.5 due to similar successful proofs
    
    def test_ensemble_optimization(self, optimizer, sample_tactics, sample_features, sample_context):
        """Test ensemble optimization combining multiple algorithms."""
        
        # Mock individual optimizers
        with patch.object(optimizer.neural_network, 'optimize') as mock_nn, \
             patch.object(optimizer.rl_optimizer, 'optimize') as mock_rl, \
             patch.object(optimizer.evolutionary_optimizer, 'optimize') as mock_evo, \
             patch.object(optimizer.transformer_optimizer, 'optimize') as mock_transformer:
            
            # Set up mock returns
            mock_nn.return_value = OptimizationResult(
                optimized_tactics=["intro", "apply", "auto"],
                confidence_score=0.8,
                optimization_time_ms=100.0,
                model_used="neural_network",
                improvement_estimate=0.3
            )
            
            mock_rl.return_value = OptimizationResult(
                optimized_tactics=["intro", "simpl", "auto"],
                confidence_score=0.7,
                optimization_time_ms=150.0,
                model_used="reinforcement_learning", 
                improvement_estimate=0.25
            )
            
            mock_evo.return_value = OptimizationResult(
                optimized_tactics=["destruct", "intro", "apply"],
                confidence_score=0.6,
                optimization_time_ms=200.0,
                model_used="evolutionary",
                improvement_estimate=0.2
            )
            
            mock_transformer.return_value = OptimizationResult(
                optimized_tactics=["intro", "apply", "qed"],
                confidence_score=0.9,
                optimization_time_ms=120.0,
                model_used="transformer",
                improvement_estimate=0.4
            )
            
            result = optimizer._ensemble_optimize(sample_tactics, sample_features, sample_context)
            
            assert isinstance(result.optimized_tactics, list)
            assert len(result.optimized_tactics) > 0
            assert result.model_used == "ensemble"
            assert 0.0 <= result.confidence_score <= 1.0
            assert result.improvement_estimate > 0
    
    def test_ensemble_result_combination(self, optimizer):
        """Test combination of ensemble results."""
        results = [
            (OptimizationResult(
                optimized_tactics=["intro", "apply", "auto"],
                confidence_score=0.8,
                optimization_time_ms=100.0,
                model_used="nn",
                improvement_estimate=0.3
            ), 0.3),
            (OptimizationResult(
                optimized_tactics=["intro", "simpl", "qed"],
                confidence_score=0.7,
                optimization_time_ms=150.0,
                model_used="rl",
                improvement_estimate=0.25
            ), 0.4)
        ]
        
        combined_tactics = optimizer._combine_ensemble_results(results)
        
        assert isinstance(combined_tactics, list)
        assert len(combined_tactics) > 0
        # "intro" appears in both, so should be preferred
        assert "intro" in combined_tactics
    
    def test_optimization_recording(self, optimizer, sample_tactics, sample_features, sample_context):
        """Test optimization recording for learning."""
        result = OptimizationResult(
            optimized_tactics=["intro", "apply", "qed"],
            confidence_score=0.8,
            optimization_time_ms=100.0,
            model_used="test",
            improvement_estimate=0.3
        )
        
        initial_samples = len(optimizer.training_samples)
        initial_history = len(optimizer.optimization_history)
        
        optimizer._record_optimization(sample_tactics, result, sample_features, sample_context)
        
        assert len(optimizer.training_samples) == initial_samples + 1
        assert len(optimizer.optimization_history) == initial_history + 1
        
        # Check recorded data
        latest_sample = optimizer.training_samples[-1]
        assert latest_sample["original_tactics"] == sample_tactics
        assert latest_sample["optimized_tactics"] == result.optimized_tactics
        assert latest_sample["confidence"] == result.confidence_score
    
    @pytest.mark.asyncio
    async def test_model_updates(self, optimizer):
        """Test automatic model updates."""
        # Add enough samples to trigger update
        for i in range(150):
            sample = {
                "timestamp": time.time(),
                "original_tactics": [f"tactic_{i}"],
                "optimized_tactics": [f"optimized_{i}"],
                "confidence": 0.7,
                "improvement": 0.2,
                "features": {"test": i},
                "context": {"prover": "test"}
            }
            optimizer.training_samples.append(sample)
        
        with patch.object(optimizer, '_update_models') as mock_update:
            await optimizer._maybe_update_models()
            mock_update.assert_called_once()
    
    def test_ensemble_weight_updates(self, optimizer):
        """Test ensemble weight updates based on performance."""
        # Add performance history
        optimizer.optimization_history = [
            {"timestamp": time.time(), "model_used": "neural_network", "confidence": 0.8, "improvement": 0.3},
            {"timestamp": time.time(), "model_used": "neural_network", "confidence": 0.7, "improvement": 0.2},
            {"timestamp": time.time(), "model_used": "reinforcement_learning", "confidence": 0.6, "improvement": 0.1},
            {"timestamp": time.time(), "model_used": "evolutionary", "confidence": 0.5, "improvement": 0.1},
            {"timestamp": time.time(), "model_used": "transformer", "confidence": 0.9, "improvement": 0.4}
        ]
        
        initial_nn_weight = optimizer.ensemble_weights[OptimizationType.NEURAL_NETWORK]
        
        optimizer._update_ensemble_weights()
        
        # Neural network and transformer should get higher weights due to better performance
        final_nn_weight = optimizer.ensemble_weights[OptimizationType.NEURAL_NETWORK]
        transformer_weight = optimizer.ensemble_weights[OptimizationType.TRANSFORMER]
        
        # Weights should still sum to ~1.0
        total_weight = sum(optimizer.ensemble_weights.values())
        assert abs(total_weight - 1.0) < 0.01
    
    def test_optimization_statistics(self, optimizer):
        """Test optimization statistics reporting."""
        # Add some history
        optimizer.optimization_history = [
            {
                "timestamp": time.time(),
                "model_used": "neural_network",
                "confidence": 0.8,
                "improvement": 0.3,
                "optimization_time_ms": 100.0
            },
            {
                "timestamp": time.time(),
                "model_used": "ensemble",
                "confidence": 0.7,
                "improvement": 0.25,
                "optimization_time_ms": 150.0
            }
        ]
        
        optimizer.training_samples.extend([{"dummy": "data"} for _ in range(50)])
        
        stats = optimizer.get_optimization_stats()
        
        assert stats["total_optimizations"] == 2
        assert stats["avg_confidence"] == 0.75  # (0.8 + 0.7) / 2
        assert stats["avg_improvement"] == 0.275  # (0.3 + 0.25) / 2
        assert stats["training_samples"] == 50
        assert "model_usage" in stats
        assert "ensemble_weights" in stats


class TestNeuralNetworkOptimizer:
    """Test neural network-based optimizer."""
    
    @pytest.fixture
    def nn_optimizer(self):
        """Create neural network optimizer for testing."""
        return NeuralNetworkOptimizer()
    
    def test_untrained_optimization(self, nn_optimizer, sample_tactics, sample_features, sample_context):
        """Test optimization with untrained model."""
        result = nn_optimizer.optimize(sample_tactics, sample_features, sample_context)
        
        assert result.optimized_tactics == sample_tactics  # Should return original
        assert result.confidence_score == 0.3
        assert result.model_used == "neural_network"
        assert "needs more training" in result.recommendations[0].lower()
    
    def test_feature_vector_construction(self, nn_optimizer, sample_features):
        """Test feature vector construction."""
        vector = nn_optimizer._construct_feature_vector(sample_features)
        
        assert isinstance(vector, np.ndarray)
        assert len(vector) == 5  # Based on current implementation
        assert all(isinstance(val, (int, float, np.number)) for val in vector)
    
    def test_model_training_simulation(self, nn_optimizer):
        """Test model training simulation."""
        assert not nn_optimizer.is_trained
        
        # Provide enough training samples
        training_samples = [{"dummy": f"sample_{i}"} for i in range(60)]
        nn_optimizer.update_model(training_samples)
        
        assert nn_optimizer.is_trained
    
    def test_trained_optimization(self, nn_optimizer, sample_tactics, sample_features, sample_context):
        """Test optimization with trained model."""
        # Mark as trained
        nn_optimizer.is_trained = True
        
        result = nn_optimizer.optimize(sample_tactics, sample_features, sample_context)
        
        assert isinstance(result.optimized_tactics, list)
        assert result.confidence_score == 0.7
        assert result.model_used == "neural_network"
        assert result.improvement_estimate == 0.3


class TestReinforcementLearningOptimizer:
    """Test reinforcement learning optimizer."""
    
    @pytest.fixture
    def rl_optimizer(self):
        """Create RL optimizer for testing."""
        return ReinforcementLearningOptimizer()
    
    def test_state_to_key_conversion(self, rl_optimizer, sample_features):
        """Test state to key conversion."""
        key = rl_optimizer._state_to_key(sample_features)
        
        assert isinstance(key, str)
        assert "len_" in key
        assert "vars_" in key
        assert "depth_" in key
    
    def test_optimization_exploration_exploitation(self, rl_optimizer, sample_tactics, sample_features, sample_context):
        """Test exploration vs exploitation in RL optimization."""
        # Set epsilon to 0 for pure exploitation
        rl_optimizer.epsilon = 0.0
        
        result1 = rl_optimizer.optimize(sample_tactics, sample_features, sample_context)
        
        # Set epsilon to 1 for pure exploration
        rl_optimizer.epsilon = 1.0
        
        result2 = rl_optimizer.optimize(sample_tactics, sample_features, sample_context)
        
        # Both should return valid results
        assert isinstance(result1.optimized_tactics, list)
        assert isinstance(result2.optimized_tactics, list)
        assert result1.model_used == "reinforcement_learning"
        assert result2.model_used == "reinforcement_learning"
    
    def test_q_table_updates(self, rl_optimizer):
        """Test Q-table updates during policy learning."""
        initial_q_size = len(rl_optimizer.q_table)
        
        # Mock training samples
        training_samples = [
            {
                "features": {"tactic_sequence_length": 3, "variable_count": 2, "quantifier_depth": 1},
                "optimized_tactics": ["intro", "apply"],
                "improvement": 0.5
            }
        ]
        
        rl_optimizer.update_policy(training_samples)
        
        # Q-table should be updated
        assert len(rl_optimizer.q_table) >= initial_q_size


class TestEvolutionaryOptimizer:
    """Test evolutionary algorithm optimizer."""
    
    @pytest.fixture
    def evo_optimizer(self):
        """Create evolutionary optimizer for testing."""
        return EvolutionaryOptimizer()
    
    def test_population_initialization(self, evo_optimizer, sample_tactics):
        """Test population initialization."""
        population = evo_optimizer._initialize_population(sample_tactics)
        
        assert len(population) == evo_optimizer.population_size
        assert all(isinstance(individual, list) for individual in population)
        assert all(len(individual) == len(sample_tactics) for individual in population)
        
        # Each individual should contain all tactics (possibly reordered)
        for individual in population:
            assert set(individual) == set(sample_tactics)
    
    def test_fitness_evaluation(self, evo_optimizer, sample_tactics, sample_features):
        """Test fitness evaluation."""
        individual = sample_tactics[:3]  # Take subset
        
        fitness = evo_optimizer._evaluate_fitness(individual, sample_features)
        
        assert isinstance(fitness, (int, float))
        # Fitness can be negative due to length penalty
    
    def test_tournament_selection(self, evo_optimizer, sample_tactics, sample_features):
        """Test tournament selection."""
        population = evo_optimizer._initialize_population(sample_tactics)
        fitness_scores = [evo_optimizer._evaluate_fitness(ind, sample_features) for ind in population]
        
        selected = evo_optimizer._tournament_selection(population, fitness_scores)
        
        assert len(selected) == evo_optimizer.population_size
        assert all(isinstance(individual, list) for individual in selected)
    
    def test_crossover_operation(self, evo_optimizer):
        """Test crossover operation."""
        parent1 = ["intro", "apply", "auto", "qed"]
        parent2 = ["destruct", "simpl", "trivial", "exact"]
        
        child1, child2 = evo_optimizer._crossover(parent1, parent2)
        
        assert isinstance(child1, list)
        assert isinstance(child2, list)
        
        # Children should contain elements from parents
        all_parent_elements = set(parent1 + parent2)
        assert set(child1).issubset(all_parent_elements)
        assert set(child2).issubset(all_parent_elements)
    
    def test_mutation_operation(self, evo_optimizer, sample_tactics):
        """Test mutation operation."""
        original = sample_tactics[:4].copy()
        individual = original.copy()
        
        evo_optimizer._mutate(individual, sample_tactics)
        
        # Individual should still be valid
        assert isinstance(individual, list)
        assert all(tactic in sample_tactics for tactic in individual)
        
        # Length might change due to insert/delete mutations
        assert len(individual) >= 1  # Should not be empty
    
    def test_evolution_process(self, evo_optimizer, sample_tactics, sample_features, sample_context):
        """Test full evolution process."""
        result = evo_optimizer.optimize(sample_tactics, sample_features, sample_context)
        
        assert isinstance(result.optimized_tactics, list)
        assert len(result.optimized_tactics) >= 1
        assert result.model_used == "evolutionary"
        assert result.confidence_score == 0.5  # Fixed in implementation
        assert result.improvement_estimate == 0.2


class TestTransformerOptimizer:
    """Test transformer-based optimizer."""
    
    @pytest.fixture
    def transformer_optimizer(self):
        """Create transformer optimizer for testing."""
        return TransformerOptimizer()
    
    def test_untrained_optimization(self, transformer_optimizer, sample_tactics, sample_features, sample_context):
        """Test optimization with untrained transformer."""
        result = transformer_optimizer.optimize(sample_tactics, sample_features, sample_context)
        
        assert result.optimized_tactics == sample_tactics
        assert result.confidence_score == 0.4
        assert result.model_used == "transformer"
        assert "pre-training" in result.recommendations[0].lower()
    
    def test_vocabulary_building(self, transformer_optimizer):
        """Test vocabulary building from training data."""
        training_samples = [
            {"original_tactics": ["intro", "apply"], "optimized_tactics": ["apply", "intro"]},
            {"original_tactics": ["destruct", "simpl"], "optimized_tactics": ["simpl", "auto"]}
        ]
        
        transformer_optimizer._build_vocabulary(training_samples)
        
        expected_tactics = {"intro", "apply", "destruct", "simpl", "auto"}
        assert set(transformer_optimizer.vocab.keys()) == expected_tactics
    
    def test_attention_mechanism(self, transformer_optimizer, sample_tactics, sample_features):
        """Test attention mechanism application."""
        # Mark as trained
        transformer_optimizer.is_trained = True
        
        reordered = transformer_optimizer._apply_attention(sample_tactics, sample_features)
        
        assert isinstance(reordered, list)
        assert len(reordered) == len(sample_tactics)
        assert set(reordered) == set(sample_tactics)  # Same tactics, possibly reordered
    
    def test_model_training(self, transformer_optimizer):
        """Test transformer model training."""
        assert not transformer_optimizer.is_trained
        
        # Provide sufficient training data
        training_samples = [{"dummy": f"sample_{i}"} for i in range(150)]
        transformer_optimizer.update_model(training_samples)
        
        assert transformer_optimizer.is_trained
        assert len(transformer_optimizer.vocab) > 0


class TestProofFeatures:
    """Test proof features data structure."""
    
    def test_features_initialization(self):
        """Test proof features initialization."""
        features = ProofFeatures()
        
        assert features.ast_depth == 0
        assert features.node_count == 0
        assert isinstance(features.operator_frequencies, dict)
        assert features.variable_count == 0
        assert features.quantifier_depth == 0
        assert features.formula_complexity == 0.0
    
    def test_features_with_data(self):
        """Test proof features with provided data."""
        features = ProofFeatures(
            ast_depth=5,
            node_count=20,
            operator_frequencies={"apply": 3, "intro": 2},
            variable_count=4,
            quantifier_depth=2,
            formula_complexity=0.8,
            similar_proof_success_rate=0.75,
            tactic_sequence_length=8,
            prover_type="coq",
            model_type="claude-3"
        )
        
        assert features.ast_depth == 5
        assert features.node_count == 20
        assert features.operator_frequencies["apply"] == 3
        assert features.variable_count == 4
        assert features.quantifier_depth == 2
        assert features.formula_complexity == 0.8
        assert features.similar_proof_success_rate == 0.75
        assert features.tactic_sequence_length == 8
        assert features.prover_type == "coq"
        assert features.model_type == "claude-3"


class TestOptimizationResult:
    """Test optimization result data structure."""
    
    def test_result_initialization(self):
        """Test optimization result initialization."""
        result = OptimizationResult(
            optimized_tactics=["intro", "apply", "qed"],
            confidence_score=0.8,
            optimization_time_ms=150.0,
            model_used="test_model",
            improvement_estimate=0.3,
            recommendations=["Good optimization", "Try different approach"]
        )
        
        assert result.optimized_tactics == ["intro", "apply", "qed"]
        assert result.confidence_score == 0.8
        assert result.optimization_time_ms == 150.0
        assert result.model_used == "test_model"
        assert result.improvement_estimate == 0.3
        assert len(result.recommendations) == 2
    
    def test_result_default_recommendations(self):
        """Test optimization result with default recommendations."""
        result = OptimizationResult(
            optimized_tactics=["test"],
            confidence_score=0.5,
            optimization_time_ms=100.0,
            model_used="test",
            improvement_estimate=0.1
        )
        
        assert isinstance(result.recommendations, list)
        assert len(result.recommendations) == 0  # Empty by default


@pytest.mark.integration
class TestMLOptimizationIntegration:
    """Integration tests for ML optimization."""
    
    def test_end_to_end_optimization(self):
        """Test end-to-end optimization pipeline."""
        optimizer = MLProofOptimizer(
            optimization_type=OptimizationType.ENSEMBLE,
            max_training_samples=50
        )
        
        tactics = ["intro x", "destruct x", "apply lemma", "simpl", "auto"]
        goal = "forall x : nat, x + 0 = x"
        context = {"prover": "isabelle", "model": "gpt-4-turbo", "timeout": 300}
        
        # Run optimization
        result = optimizer.optimize_proof(tactics, goal, context)
        
        # Verify result
        assert isinstance(result, OptimizationResult)
        assert isinstance(result.optimized_tactics, list)
        assert len(result.optimized_tactics) > 0
        assert 0.0 <= result.confidence_score <= 1.0
        assert result.optimization_time_ms >= 0
        assert result.model_used in ["neural_network", "reinforcement_learning", 
                                    "evolutionary", "transformer", "ensemble", "fallback"]
    
    def test_multiple_optimizations(self):
        """Test multiple optimizations and learning."""
        optimizer = MLProofOptimizer(max_training_samples=20)
        
        test_cases = [
            (["intro", "simpl", "auto"], "forall x, x = x"),
            (["destruct", "apply", "qed"], "exists x, x > 0"),
            (["induction", "intro", "simpl"], "forall n, n + 0 = n")
        ]
        
        context = {"prover": "coq", "model": "gpt-4-turbo"}
        
        results = []
        for tactics, goal in test_cases:
            result = optimizer.optimize_proof(tactics, goal, context)
            results.append(result)
        
        # Verify all optimizations completed
        assert len(results) == 3
        assert all(isinstance(r, OptimizationResult) for r in results)
        
        # Verify learning occurred
        assert len(optimizer.training_samples) == 3
        assert len(optimizer.optimization_history) == 3
        
        # Check statistics
        stats = optimizer.get_optimization_stats()
        assert stats["total_optimizations"] == 3
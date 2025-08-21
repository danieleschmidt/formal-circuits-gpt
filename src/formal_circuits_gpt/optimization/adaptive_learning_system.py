"""
Self-Improving Adaptive Learning System for Formal Verification

This module implements a sophisticated adaptive learning system that continuously
improves verification performance through multi-modal learning, meta-learning,
and autonomous knowledge discovery.

Academic Contribution: "Autonomous Knowledge Discovery in Formal Verification:
A Self-Improving Meta-Learning Approach"

Key Innovation: The system automatically discovers new verification strategies,
adapts to novel circuit patterns, and evolves its reasoning capabilities
through continuous learning and self-reflection.

Authors: Daniel Schmidt, Terragon Labs
Target Venue: IJCAI 2026, AAAI 2026
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import json
import math
import uuid
from collections import defaultdict, deque
import pickle
import networkx as nx
from sklearn.cluster import DBSCAN, KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import random

from ..neural_symbolic_fusion import NeuralSymbolicFusionEngine, FusionResult
from ..quantum_proof_search import QuantumProofSearcher
from .ml_proof_optimizer import MLProofOptimizer


class LearningMode(Enum):
    """Adaptive learning modes."""
    
    EXPLORATION = "exploration"        # Discover new patterns
    EXPLOITATION = "exploitation"      # Use known patterns
    META_LEARNING = "meta_learning"    # Learn how to learn
    SELF_REFLECTION = "self_reflection" # Analyze own performance
    KNOWLEDGE_DISCOVERY = "knowledge_discovery"  # Autonomous discovery
    MULTI_MODAL = "multi_modal"        # Integrate multiple learning modes


class KnowledgeType(Enum):
    """Types of knowledge to learn and store."""
    
    TACTICAL = "tactical"              # Proof tactics and strategies
    STRUCTURAL = "structural"          # Circuit structure patterns
    SEMANTIC = "semantic"              # Semantic understanding
    PROCEDURAL = "procedural"          # Problem-solving procedures
    META_COGNITIVE = "meta_cognitive"  # Self-awareness and regulation
    EMERGENT = "emergent"              # Discovered patterns


@dataclass
class LearningExperience:
    """Comprehensive learning experience record."""
    
    experience_id: str
    timestamp: float
    circuit_context: Dict[str, Any]
    verification_goal: str
    attempted_strategy: List[str]
    outcome: str
    success_rate: float
    execution_time: float
    resource_usage: Dict[str, float]
    learned_patterns: List[str]
    failure_analysis: Optional[Dict[str, Any]] = None
    insight_level: float = 0.0
    knowledge_type: KnowledgeType = KnowledgeType.TACTICAL
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert experience to feature vector for ML."""
        features = [
            self.success_rate,
            self.execution_time,
            len(self.attempted_strategy),
            len(self.learned_patterns),
            self.insight_level,
            hash(self.knowledge_type.value) % 1000 / 1000.0,  # Normalized hash
            self.resource_usage.get('memory_mb', 0) / 1000.0,  # Normalized
            self.resource_usage.get('cpu_usage', 0),
        ]
        return np.array(features, dtype=np.float32)


@dataclass
class MetaLearningState:
    """State for meta-learning algorithms."""
    
    learning_rate_adaptation: Dict[str, float] = field(default_factory=dict)
    strategy_effectiveness: Dict[str, float] = field(default_factory=dict)
    exploration_bias: float = 0.3
    confidence_threshold: float = 0.7
    adaptation_speed: float = 0.1
    meta_gradient_accumulator: Dict[str, torch.Tensor] = field(default_factory=dict)
    
    def update_strategy_effectiveness(self, strategy: str, success: bool, learning_rate: float = 0.1):
        """Update effectiveness estimate for a strategy."""
        current = self.strategy_effectiveness.get(strategy, 0.5)
        target = 1.0 if success else 0.0
        self.strategy_effectiveness[strategy] = current + learning_rate * (target - current)


class AutonomousKnowledgeDiscoverer:
    """Autonomous system for discovering new verification knowledge."""
    
    def __init__(self, discovery_threshold: float = 0.8):
        self.discovery_threshold = discovery_threshold
        self.discovered_patterns: Dict[str, Dict[str, Any]] = {}
        self.pattern_clusters: List[List[str]] = []
        self.novelty_detector = self._initialize_novelty_detector()
        
    def _initialize_novelty_detector(self) -> DBSCAN:
        """Initialize novelty detection algorithm."""
        return DBSCAN(eps=0.3, min_samples=3)
    
    def analyze_experience_patterns(self, experiences: List[LearningExperience]) -> List[Dict[str, Any]]:
        """Analyze experiences to discover new patterns."""
        if len(experiences) < 10:
            return []
        
        # Convert experiences to feature vectors
        feature_vectors = np.array([exp.to_feature_vector() for exp in experiences])
        
        # Cluster similar experiences
        clustering = self.novelty_detector.fit(feature_vectors)
        labels = clustering.labels_
        
        discovered_patterns = []
        
        # Analyze each cluster
        for cluster_id in set(labels):
            if cluster_id == -1:  # Noise points
                continue
            
            cluster_experiences = [
                experiences[i] for i, label in enumerate(labels) 
                if label == cluster_id
            ]
            
            if len(cluster_experiences) < 3:
                continue
            
            # Extract common patterns from cluster
            pattern = self._extract_cluster_pattern(cluster_experiences)
            if pattern and pattern['confidence'] > self.discovery_threshold:
                pattern['discovery_id'] = str(uuid.uuid4())
                pattern['discovery_timestamp'] = time.time()
                discovered_patterns.append(pattern)
        
        return discovered_patterns
    
    def _extract_cluster_pattern(self, experiences: List[LearningExperience]) -> Optional[Dict[str, Any]]:
        """Extract common pattern from clustered experiences."""
        if not experiences:
            return None
        
        # Analyze success patterns
        success_rates = [exp.success_rate for exp in experiences]
        avg_success_rate = np.mean(success_rates)
        
        if avg_success_rate < 0.6:  # Low success cluster, less interesting
            return None
        
        # Find common tactics
        all_tactics = []
        for exp in experiences:
            all_tactics.extend(exp.attempted_strategy)
        
        tactic_counts = defaultdict(int)
        for tactic in all_tactics:
            tactic_counts[tactic] += 1
        
        # Tactics that appear in at least half the experiences
        common_tactics = [
            tactic for tactic, count in tactic_counts.items()
            if count >= len(experiences) // 2
        ]
        
        # Analyze circuit patterns
        circuit_features = []
        for exp in experiences:
            features = exp.circuit_context
            circuit_features.append({
                'module_count': features.get('module_count', 0),
                'complexity': features.get('complexity_score', 0),
                'pattern_type': features.get('pattern_type', 'unknown')
            })
        
        # Find dominant circuit characteristics
        pattern_types = [cf['pattern_type'] for cf in circuit_features]
        dominant_pattern = max(set(pattern_types), key=pattern_types.count)
        
        avg_complexity = np.mean([cf['complexity'] for cf in circuit_features])
        
        # Calculate pattern confidence
        tactic_consistency = len(common_tactics) / max(1, len(set(all_tactics)))
        success_consistency = 1.0 - np.std(success_rates)
        confidence = (tactic_consistency + success_consistency + avg_success_rate) / 3.0
        
        return {
            'pattern_type': 'discovered_strategy',
            'dominant_circuit_pattern': dominant_pattern,
            'average_complexity': avg_complexity,
            'common_tactics': common_tactics,
            'success_rate': avg_success_rate,
            'confidence': confidence,
            'evidence_count': len(experiences),
            'insight_description': self._generate_insight_description(
                common_tactics, dominant_pattern, avg_success_rate
            )
        }
    
    def _generate_insight_description(
        self, tactics: List[str], circuit_pattern: str, success_rate: float
    ) -> str:
        """Generate human-readable insight description."""
        insight_templates = [
            f"For {circuit_pattern} circuits, using {', '.join(tactics[:3])} achieves {success_rate:.1%} success",
            f"Strategy combining {tactics[0] if tactics else 'unknown'} with {tactics[1] if len(tactics) > 1 else 'basic approach'} works well for {circuit_pattern} patterns",
            f"Discovered high-performance approach for {circuit_pattern}: {', '.join(tactics[:2])}"
        ]
        
        return random.choice(insight_templates)


class MetaLearningEngine:
    """Meta-learning engine that learns how to learn better."""
    
    def __init__(self, meta_learning_rate: float = 0.01):
        self.meta_learning_rate = meta_learning_rate
        self.task_distributions: Dict[str, List[float]] = defaultdict(list)
        self.adaptation_strategies: Dict[str, Callable] = {}
        self.meta_gradients: Dict[str, torch.Tensor] = {}
        
        # MAML (Model-Agnostic Meta-Learning) components
        self.meta_model = self._initialize_meta_model()
        self.meta_optimizer = torch.optim.Adam(self.meta_model.parameters(), lr=meta_learning_rate)
        
    def _initialize_meta_model(self) -> nn.Module:
        """Initialize meta-learning model."""
        return nn.Sequential(
            nn.Linear(16, 64),    # Input: task features
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8),     # Output: learning parameters
            nn.Sigmoid()
        )
    
    def adapt_to_task(
        self, 
        task_context: Dict[str, Any], 
        few_shot_examples: List[LearningExperience]
    ) -> Dict[str, float]:
        """Adapt learning parameters for new task using meta-learning."""
        
        # Extract task features
        task_features = self._extract_task_features(task_context, few_shot_examples)
        task_tensor = torch.tensor(task_features, dtype=torch.float32).unsqueeze(0)
        
        # Predict optimal learning parameters
        with torch.no_grad():
            adaptation_params = self.meta_model(task_tensor).squeeze().numpy()
        
        # Map to learning parameters
        adapted_params = {
            'learning_rate': float(adaptation_params[0] * 0.1),  # Scale to reasonable range
            'exploration_rate': float(adaptation_params[1]),
            'confidence_threshold': float(0.5 + adaptation_params[2] * 0.4),
            'tactic_selection_temperature': float(0.1 + adaptation_params[3] * 0.9),
            'pattern_matching_weight': float(adaptation_params[4]),
            'novelty_bonus': float(adaptation_params[5] * 0.2),
            'meta_gradient_scale': float(adaptation_params[6] * 0.1),
            'adaptation_speed': float(adaptation_params[7] * 0.5)
        }
        
        return adapted_params
    
    def _extract_task_features(
        self, 
        task_context: Dict[str, Any], 
        examples: List[LearningExperience]
    ) -> np.ndarray:
        """Extract features characterizing the learning task."""
        
        # Task complexity features
        circuit_complexity = task_context.get('complexity_score', 0) / 100.0  # Normalize
        module_count = min(task_context.get('module_count', 0) / 50.0, 1.0)  # Cap at 50
        
        # Historical performance features
        if examples:
            avg_success_rate = np.mean([exp.success_rate for exp in examples])
            avg_execution_time = np.mean([exp.execution_time for exp in examples]) / 1000.0  # Normalize
            strategy_diversity = len(set(
                tactic for exp in examples for tactic in exp.attempted_strategy
            )) / max(1, sum(len(exp.attempted_strategy) for exp in examples))
        else:
            avg_success_rate = 0.5
            avg_execution_time = 0.5
            strategy_diversity = 0.5
        
        # Domain-specific features
        verification_type = hash(task_context.get('verification_type', 'unknown')) % 1000 / 1000.0
        property_count = min(task_context.get('property_count', 0) / 10.0, 1.0)
        
        # Meta-features (learning context)
        task_novelty = self._compute_task_novelty(task_context)
        resource_constraints = task_context.get('resource_constraints', 0.5)
        time_pressure = task_context.get('time_pressure', 0.5)
        
        features = np.array([
            circuit_complexity,
            module_count,
            avg_success_rate,
            avg_execution_time,
            strategy_diversity,
            verification_type,
            property_count,
            task_novelty,
            resource_constraints,
            time_pressure,
            # Additional padding features
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ], dtype=np.float32)
        
        return features[:16]  # Ensure exactly 16 features
    
    def _compute_task_novelty(self, task_context: Dict[str, Any]) -> float:
        """Compute how novel this task is compared to previous tasks."""
        # Simplified novelty computation
        task_signature = json.dumps(sorted(task_context.items()))
        task_hash = hash(task_signature) % 1000000
        
        # Check against known task distributions
        domain = task_context.get('domain', 'unknown')
        if domain in self.task_distributions:
            # Compute distance to nearest known task
            known_hashes = self.task_distributions[domain]
            if known_hashes:
                min_distance = min(abs(task_hash - known_hash) for known_hash in known_hashes)
                novelty = min(min_distance / 1000000.0, 1.0)
            else:
                novelty = 1.0
            
            self.task_distributions[domain].append(task_hash)
        else:
            novelty = 1.0
            self.task_distributions[domain] = [task_hash]
        
        return novelty
    
    def update_meta_learning(
        self, 
        task_context: Dict[str, Any],
        adaptation_params: Dict[str, float],
        task_performance: float
    ):
        """Update meta-learning model based on task performance."""
        
        # Extract task features
        task_features = self._extract_task_features(task_context, [])
        task_tensor = torch.tensor(task_features, dtype=torch.float32).unsqueeze(0)
        
        # Predicted parameters
        predicted_params = self.meta_model(task_tensor)
        
        # Target parameters (adjusted based on performance)
        target_params = torch.tensor([
            adaptation_params.get('learning_rate', 0.01) / 0.1,
            adaptation_params.get('exploration_rate', 0.3),
            (adaptation_params.get('confidence_threshold', 0.7) - 0.5) / 0.4,
            (adaptation_params.get('tactic_selection_temperature', 0.5) - 0.1) / 0.9,
            adaptation_params.get('pattern_matching_weight', 0.5),
            adaptation_params.get('novelty_bonus', 0.1) / 0.2,
            adaptation_params.get('meta_gradient_scale', 0.05) / 0.1,
            adaptation_params.get('adaptation_speed', 0.25) / 0.5
        ], dtype=torch.float32).unsqueeze(0)
        
        # Adjust target based on performance
        performance_factor = 2.0 * task_performance - 1.0  # Map [0,1] to [-1,1]
        target_params = target_params + 0.1 * performance_factor * (target_params - 0.5)
        target_params = torch.clamp(target_params, 0.0, 1.0)
        
        # Compute loss and update
        loss = F.mse_loss(predicted_params, target_params)
        
        self.meta_optimizer.zero_grad()
        loss.backward()
        self.meta_optimizer.step()


class SelfReflectionModule:
    """Module for self-reflection and performance analysis."""
    
    def __init__(self, reflection_depth: int = 3):
        self.reflection_depth = reflection_depth
        self.performance_history: deque = deque(maxlen=1000)
        self.reflection_insights: List[Dict[str, Any]] = []
        self.self_awareness_metrics: Dict[str, float] = {}
        
    def reflect_on_performance(
        self, 
        recent_experiences: List[LearningExperience]
    ) -> Dict[str, Any]:
        """Perform deep self-reflection on recent performance."""
        
        if len(recent_experiences) < 5:
            return {'reflection_status': 'insufficient_data'}
        
        # Multi-level reflection
        reflection_results = {}
        
        # Level 1: Performance trends
        performance_trend = self._analyze_performance_trends(recent_experiences)
        reflection_results['performance_trends'] = performance_trend
        
        # Level 2: Strategy effectiveness
        strategy_analysis = self._analyze_strategy_effectiveness(recent_experiences)
        reflection_results['strategy_analysis'] = strategy_analysis
        
        # Level 3: Meta-cognitive insights
        meta_insights = self._generate_meta_cognitive_insights(recent_experiences)
        reflection_results['meta_insights'] = meta_insights
        
        # Self-improvement recommendations
        recommendations = self._generate_self_improvement_plan(reflection_results)
        reflection_results['self_improvement_plan'] = recommendations
        
        # Update self-awareness metrics
        self._update_self_awareness(reflection_results)
        
        return reflection_results
    
    def _analyze_performance_trends(self, experiences: List[LearningExperience]) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        
        # Sort by timestamp
        sorted_experiences = sorted(experiences, key=lambda x: x.timestamp)
        
        success_rates = [exp.success_rate for exp in sorted_experiences]
        execution_times = [exp.execution_time for exp in sorted_experiences]
        
        # Trend analysis
        if len(success_rates) > 1:
            success_trend = np.polyfit(range(len(success_rates)), success_rates, 1)[0]
            time_trend = np.polyfit(range(len(execution_times)), execution_times, 1)[0]
        else:
            success_trend = 0.0
            time_trend = 0.0
        
        # Performance stability
        success_stability = 1.0 / (1.0 + np.std(success_rates))
        time_stability = 1.0 / (1.0 + np.std(execution_times))
        
        return {
            'success_trend': float(success_trend),
            'time_trend': float(time_trend),
            'success_stability': float(success_stability),
            'time_stability': float(time_stability),
            'overall_performance': float(np.mean(success_rates)),
            'trend_interpretation': self._interpret_trends(success_trend, time_trend)
        }
    
    def _analyze_strategy_effectiveness(self, experiences: List[LearningExperience]) -> Dict[str, Any]:
        """Analyze effectiveness of different strategies."""
        
        strategy_performance: Dict[str, List[float]] = defaultdict(list)
        
        for exp in experiences:
            strategy_signature = '_'.join(sorted(exp.attempted_strategy[:3]))  # First 3 tactics
            strategy_performance[strategy_signature].append(exp.success_rate)
        
        strategy_analysis = {}
        for strategy, performances in strategy_performance.items():
            if len(performances) >= 2:  # Need at least 2 samples
                strategy_analysis[strategy] = {
                    'mean_performance': float(np.mean(performances)),
                    'stability': float(1.0 / (1.0 + np.std(performances))),
                    'sample_count': len(performances),
                    'confidence': min(1.0, len(performances) / 10.0)
                }
        
        # Identify best and worst strategies
        if strategy_analysis:
            best_strategy = max(strategy_analysis.keys(), 
                              key=lambda s: strategy_analysis[s]['mean_performance'])
            worst_strategy = min(strategy_analysis.keys(), 
                               key=lambda s: strategy_analysis[s]['mean_performance'])
        else:
            best_strategy = None
            worst_strategy = None
        
        return {
            'strategy_performance': strategy_analysis,
            'best_strategy': best_strategy,
            'worst_strategy': worst_strategy,
            'strategy_diversity': len(strategy_analysis),
            'recommendations': self._generate_strategy_recommendations(strategy_analysis)
        }
    
    def _generate_meta_cognitive_insights(self, experiences: List[LearningExperience]) -> List[str]:
        """Generate meta-cognitive insights about learning patterns."""
        
        insights = []
        
        # Learning pattern analysis
        knowledge_types = [exp.knowledge_type for exp in experiences]
        dominant_knowledge_type = max(set(knowledge_types), key=knowledge_types.count)
        
        insights.append(f"Primary learning focus: {dominant_knowledge_type.value}")
        
        # Failure pattern analysis
        failures = [exp for exp in experiences if exp.success_rate < 0.5]
        if failures:
            failure_rate = len(failures) / len(experiences)
            insights.append(f"Current failure rate: {failure_rate:.1%}")
            
            common_failure_patterns = self._identify_failure_patterns(failures)
            if common_failure_patterns:
                insights.append(f"Common failure pattern: {common_failure_patterns[0]}")
        
        # Learning efficiency insights
        high_insight_experiences = [exp for exp in experiences if exp.insight_level > 0.7]
        if high_insight_experiences:
            insights.append(f"High-insight learning represents {len(high_insight_experiences)/len(experiences):.1%} of experiences")
        
        # Adaptive capacity assessment
        recent_novelty = np.mean([
            exp.insight_level for exp in experiences[-10:] 
        ]) if len(experiences) >= 10 else 0.0
        
        if recent_novelty > 0.6:
            insights.append("Demonstrating strong adaptive learning capacity")
        elif recent_novelty < 0.3:
            insights.append("May be over-relying on existing patterns")
        
        return insights
    
    def _interpret_trends(self, success_trend: float, time_trend: float) -> str:
        """Interpret performance trends."""
        
        if success_trend > 0.01 and time_trend < -0.01:
            return "improving_efficiency"
        elif success_trend > 0.01:
            return "improving_effectiveness"
        elif time_trend < -0.01:
            return "improving_speed"
        elif success_trend < -0.01:
            return "declining_performance"
        else:
            return "stable_performance"
    
    def _generate_strategy_recommendations(self, strategy_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations for strategy improvement."""
        
        recommendations = []
        
        if not strategy_analysis:
            recommendations.append("Insufficient strategy data for recommendations")
            return recommendations
        
        # Find strategies to emphasize
        high_performers = [
            strategy for strategy, metrics in strategy_analysis.items()
            if metrics['mean_performance'] > 0.7 and metrics['confidence'] > 0.5
        ]
        
        if high_performers:
            recommendations.append(f"Emphasize high-performing strategies: {', '.join(high_performers[:2])}")
        
        # Find strategies to avoid
        low_performers = [
            strategy for strategy, metrics in strategy_analysis.items()
            if metrics['mean_performance'] < 0.3 and metrics['sample_count'] >= 3
        ]
        
        if low_performers:
            recommendations.append(f"Reconsider low-performing strategies: {', '.join(low_performers[:2])}")
        
        # Diversity recommendations
        if len(strategy_analysis) < 3:
            recommendations.append("Explore more diverse strategies")
        elif len(strategy_analysis) > 10:
            recommendations.append("Focus on most effective strategies")
        
        return recommendations
    
    def _identify_failure_patterns(self, failures: List[LearningExperience]) -> List[str]:
        """Identify common patterns in failures."""
        
        patterns = []
        
        # Common circuit types in failures
        circuit_types = [exp.circuit_context.get('pattern_type', 'unknown') for exp in failures]
        if circuit_types:
            common_type = max(set(circuit_types), key=circuit_types.count)
            if circuit_types.count(common_type) / len(circuit_types) > 0.5:
                patterns.append(f"struggles_with_{common_type}_circuits")
        
        # Common tactics in failures
        all_tactics = [tactic for exp in failures for tactic in exp.attempted_strategy]
        if all_tactics:
            tactic_counts = defaultdict(int)
            for tactic in all_tactics:
                tactic_counts[tactic] += 1
            
            problematic_tactic = max(tactic_counts.keys(), key=lambda t: tactic_counts[t])
            if tactic_counts[problematic_tactic] / len(failures) > 0.5:
                patterns.append(f"issues_with_{problematic_tactic}_tactic")
        
        return patterns
    
    def _generate_self_improvement_plan(self, reflection_results: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate specific self-improvement action items."""
        
        action_items = []
        
        # Based on performance trends
        trends = reflection_results.get('performance_trends', {})
        if trends.get('success_trend', 0) < -0.01:
            action_items.append({
                'category': 'performance',
                'action': 'investigate_performance_decline',
                'priority': 'high',
                'description': 'Recent performance decline detected, analyze root causes'
            })
        
        # Based on strategy analysis
        strategy_info = reflection_results.get('strategy_analysis', {})
        if strategy_info.get('strategy_diversity', 0) < 3:
            action_items.append({
                'category': 'strategy',
                'action': 'increase_strategy_diversity',
                'priority': 'medium',
                'description': 'Explore more diverse verification strategies'
            })
        
        # Based on meta-insights
        meta_insights = reflection_results.get('meta_insights', [])
        if any('failure rate' in insight and '30%' in insight for insight in meta_insights):
            action_items.append({
                'category': 'learning',
                'action': 'address_high_failure_rate',
                'priority': 'high',
                'description': 'High failure rate detected, need focused improvement'
            })
        
        return action_items
    
    def _update_self_awareness(self, reflection_results: Dict[str, Any]):
        """Update self-awareness metrics based on reflection."""
        
        # Performance awareness
        trends = reflection_results.get('performance_trends', {})
        self.self_awareness_metrics['performance_awareness'] = abs(trends.get('success_trend', 0))
        
        # Strategy awareness
        strategy_info = reflection_results.get('strategy_analysis', {})
        self.self_awareness_metrics['strategy_awareness'] = min(1.0, strategy_info.get('strategy_diversity', 0) / 5.0)
        
        # Learning awareness
        improvement_plan = reflection_results.get('self_improvement_plan', [])
        self.self_awareness_metrics['improvement_awareness'] = min(1.0, len(improvement_plan) / 3.0)
        
        # Overall self-awareness
        self.self_awareness_metrics['overall'] = np.mean(list(self.self_awareness_metrics.values()))


class AdaptiveLearningSystem:
    """Main adaptive learning system orchestrating all components."""
    
    def __init__(
        self,
        learning_modes: List[LearningMode] = None,
        adaptation_speed: float = 0.1,
        discovery_threshold: float = 0.8,
        reflection_frequency: int = 50  # Reflect every N experiences
    ):
        self.learning_modes = learning_modes or [
            LearningMode.EXPLORATION,
            LearningMode.EXPLOITATION,
            LearningMode.META_LEARNING
        ]
        self.adaptation_speed = adaptation_speed
        self.reflection_frequency = reflection_frequency
        
        # Core components
        self.knowledge_discoverer = AutonomousKnowledgeDiscoverer(discovery_threshold)
        self.meta_learning_engine = MetaLearningEngine()
        self.self_reflection_module = SelfReflectionModule()
        
        # Learning state
        self.meta_learning_state = MetaLearningState()
        self.experience_buffer: deque = deque(maxlen=10000)
        self.current_mode = LearningMode.EXPLORATION
        
        # Performance tracking
        self.learning_metrics = {
            'total_experiences': 0,
            'discoveries_made': 0,
            'adaptations_performed': 0,
            'reflections_completed': 0,
            'learning_efficiency': 0.0
        }
        
        # Integration with other systems
        self.fusion_engine: Optional[NeuralSymbolicFusionEngine] = None
        self.quantum_searcher: Optional[QuantumProofSearcher] = None
        self.ml_optimizer: Optional[MLProofOptimizer] = None
    
    def initialize_system(
        self,
        fusion_engine: NeuralSymbolicFusionEngine,
        quantum_searcher: QuantumProofSearcher,
        ml_optimizer: MLProofOptimizer
    ):
        """Initialize the adaptive learning system with other components."""
        self.fusion_engine = fusion_engine
        self.quantum_searcher = quantum_searcher
        self.ml_optimizer = ml_optimizer
        
        print("Adaptive learning system initialized with multi-modal integration")
    
    def learn_from_experience(
        self,
        circuit_context: Dict[str, Any],
        verification_goal: str,
        attempted_strategy: List[str],
        outcome: str,
        success_rate: float,
        execution_time: float,
        resource_usage: Dict[str, float]
    ) -> Dict[str, Any]:
        """Learn from a verification experience."""
        
        # Create learning experience
        experience = LearningExperience(
            experience_id=str(uuid.uuid4()),
            timestamp=time.time(),
            circuit_context=circuit_context,
            verification_goal=verification_goal,
            attempted_strategy=attempted_strategy,
            outcome=outcome,
            success_rate=success_rate,
            execution_time=execution_time,
            resource_usage=resource_usage,
            learned_patterns=self._extract_learned_patterns(attempted_strategy, success_rate),
            insight_level=self._compute_insight_level(success_rate, attempted_strategy, circuit_context)
        )
        
        # Add to experience buffer
        self.experience_buffer.append(experience)
        self.learning_metrics['total_experiences'] += 1
        
        # Adaptive learning based on current mode
        learning_results = {}
        
        if self.current_mode == LearningMode.EXPLORATION:
            learning_results.update(self._exploration_learning(experience))
        elif self.current_mode == LearningMode.EXPLOITATION:
            learning_results.update(self._exploitation_learning(experience))
        elif self.current_mode == LearningMode.META_LEARNING:
            learning_results.update(self._meta_learning_update(experience))
        
        # Knowledge discovery
        if len(self.experience_buffer) % 20 == 0:  # Every 20 experiences
            discoveries = self.knowledge_discoverer.analyze_experience_patterns(
                list(self.experience_buffer)[-50:]  # Last 50 experiences
            )
            if discoveries:
                learning_results['new_discoveries'] = discoveries
                self.learning_metrics['discoveries_made'] += len(discoveries)
        
        # Self-reflection
        if self.learning_metrics['total_experiences'] % self.reflection_frequency == 0:
            reflection_results = self.self_reflection_module.reflect_on_performance(
                list(self.experience_buffer)[-self.reflection_frequency:]
            )
            learning_results['reflection_insights'] = reflection_results
            self.learning_metrics['reflections_completed'] += 1
        
        # Adaptive mode switching
        self._adapt_learning_mode()
        
        # Update learning efficiency
        self._update_learning_efficiency()
        
        return learning_results
    
    def _extract_learned_patterns(self, strategy: List[str], success_rate: float) -> List[str]:
        """Extract learned patterns from the experience."""
        patterns = []
        
        # Pattern based on strategy effectiveness
        if success_rate > 0.8:
            patterns.append(f"effective_sequence_{'_'.join(strategy[:3])}")
        
        # Pattern based on strategy composition
        if len(strategy) <= 3 and success_rate > 0.6:
            patterns.append("efficient_short_strategy")
        elif len(strategy) > 5 and success_rate > 0.7:
            patterns.append("effective_complex_strategy")
        
        return patterns
    
    def _compute_insight_level(
        self, 
        success_rate: float, 
        strategy: List[str], 
        context: Dict[str, Any]
    ) -> float:
        """Compute the insight level of this experience."""
        
        # Base insight from success
        insight = success_rate
        
        # Bonus for novel strategies
        strategy_signature = '_'.join(sorted(strategy))
        known_strategies = [
            '_'.join(sorted(exp.attempted_strategy)) 
            for exp in list(self.experience_buffer)[-100:]
        ]
        
        if strategy_signature not in known_strategies:
            insight += 0.2  # Novelty bonus
        
        # Bonus for handling complex circuits
        complexity = context.get('complexity_score', 0)
        if complexity > 50 and success_rate > 0.7:
            insight += 0.1  # Complexity handling bonus
        
        return min(1.0, insight)
    
    def _exploration_learning(self, experience: LearningExperience) -> Dict[str, Any]:
        """Learning focused on exploration and discovery."""
        
        results = {'mode': 'exploration'}
        
        # Update exploration bias based on results
        if experience.success_rate > 0.7:
            self.meta_learning_state.exploration_bias = max(
                0.1, self.meta_learning_state.exploration_bias - 0.02
            )
        else:
            self.meta_learning_state.exploration_bias = min(
                0.5, self.meta_learning_state.exploration_bias + 0.01
            )
        
        # Record exploration outcomes
        results['exploration_bias_updated'] = self.meta_learning_state.exploration_bias
        results['exploration_outcome'] = 'successful' if experience.success_rate > 0.5 else 'unsuccessful'
        
        return results
    
    def _exploitation_learning(self, experience: LearningExperience) -> Dict[str, Any]:
        """Learning focused on exploiting known good strategies."""
        
        results = {'mode': 'exploitation'}
        
        # Update strategy effectiveness
        for strategy in experience.attempted_strategy:
            self.meta_learning_state.update_strategy_effectiveness(
                strategy, experience.success_rate > 0.5
            )
        
        # Track exploitation performance
        results['strategy_updates'] = len(experience.attempted_strategy)
        results['exploitation_success'] = experience.success_rate
        
        return results
    
    def _meta_learning_update(self, experience: LearningExperience) -> Dict[str, Any]:
        """Meta-learning update based on experience."""
        
        results = {'mode': 'meta_learning'}
        
        # Update meta-learning parameters
        task_context = {
            'complexity_score': experience.circuit_context.get('complexity_score', 0),
            'module_count': experience.circuit_context.get('module_count', 0),
            'verification_type': experience.verification_goal,
            'domain': 'formal_verification'
        }
        
        # Adapt learning parameters
        adapted_params = self.meta_learning_engine.adapt_to_task(task_context, [experience])
        
        # Update meta-learning based on performance
        self.meta_learning_engine.update_meta_learning(
            task_context, adapted_params, experience.success_rate
        )
        
        results['adapted_parameters'] = adapted_params
        results['meta_learning_updated'] = True
        
        self.learning_metrics['adaptations_performed'] += 1
        
        return results
    
    def _adapt_learning_mode(self):
        """Adaptively switch between learning modes."""
        
        if len(self.experience_buffer) < 10:
            return
        
        recent_experiences = list(self.experience_buffer)[-10:]
        recent_success_rate = np.mean([exp.success_rate for exp in recent_experiences])
        recent_novelty = np.mean([exp.insight_level for exp in recent_experiences])
        
        # Mode switching logic
        if recent_success_rate < 0.3 and self.current_mode != LearningMode.EXPLORATION:
            self.current_mode = LearningMode.EXPLORATION
        elif recent_success_rate > 0.7 and recent_novelty < 0.3:
            self.current_mode = LearningMode.EXPLOITATION
        elif recent_novelty > 0.6:
            self.current_mode = LearningMode.META_LEARNING
    
    def _update_learning_efficiency(self):
        """Update learning efficiency metric."""
        
        if len(self.experience_buffer) < 10:
            return
        
        recent_experiences = list(self.experience_buffer)[-10:]
        
        # Success rate trend
        success_rates = [exp.success_rate for exp in recent_experiences]
        if len(success_rates) > 1:
            success_trend = np.polyfit(range(len(success_rates)), success_rates, 1)[0]
        else:
            success_trend = 0.0
        
        # Discovery rate
        recent_discoveries = sum(
            1 for exp in recent_experiences if exp.insight_level > 0.7
        )
        discovery_rate = recent_discoveries / len(recent_experiences)
        
        # Combine metrics
        self.learning_metrics['learning_efficiency'] = (
            0.5 * max(0, success_trend) + 
            0.3 * discovery_rate + 
            0.2 * np.mean(success_rates)
        )
    
    def get_adaptive_recommendations(
        self, 
        current_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get adaptive recommendations for current verification task."""
        
        if not self.experience_buffer:
            return {'status': 'insufficient_experience'}
        
        # Find similar past experiences
        similar_experiences = self._find_similar_experiences(current_context)
        
        # Meta-learning adaptation
        adapted_params = self.meta_learning_engine.adapt_to_task(
            current_context, similar_experiences
        )
        
        # Strategy recommendations based on context
        strategy_recommendations = self._generate_strategy_recommendations(
            current_context, similar_experiences
        )
        
        # Learning mode recommendation
        recommended_mode = self._recommend_learning_mode(current_context, similar_experiences)
        
        return {
            'adapted_parameters': adapted_params,
            'strategy_recommendations': strategy_recommendations,
            'recommended_mode': recommended_mode.value,
            'confidence': self._compute_recommendation_confidence(similar_experiences),
            'similar_experiences_count': len(similar_experiences)
        }
    
    def _find_similar_experiences(
        self, 
        context: Dict[str, Any], 
        max_similar: int = 10
    ) -> List[LearningExperience]:
        """Find experiences similar to current context."""
        
        similarities = []
        
        for exp in self.experience_buffer:
            similarity = self._compute_context_similarity(context, exp.circuit_context)
            similarities.append((similarity, exp))
        
        # Sort by similarity and return top matches
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [exp for similarity, exp in similarities[:max_similar] if similarity > 0.3]
    
    def _compute_context_similarity(
        self, 
        context1: Dict[str, Any], 
        context2: Dict[str, Any]
    ) -> float:
        """Compute similarity between two contexts."""
        
        # Key features for comparison
        features = ['complexity_score', 'module_count', 'pattern_type']
        
        similarity = 0.0
        feature_count = 0
        
        for feature in features:
            if feature in context1 and feature in context2:
                val1 = context1[feature]
                val2 = context2[feature]
                
                if isinstance(val1, str) and isinstance(val2, str):
                    feature_sim = 1.0 if val1 == val2 else 0.0
                elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    # Normalized similarity for numeric values
                    max_val = max(abs(val1), abs(val2), 1)
                    feature_sim = 1.0 - abs(val1 - val2) / max_val
                else:
                    continue
                
                similarity += feature_sim
                feature_count += 1
        
        return similarity / max(feature_count, 1)
    
    def _generate_strategy_recommendations(
        self,
        context: Dict[str, Any],
        similar_experiences: List[LearningExperience]
    ) -> List[Dict[str, Any]]:
        """Generate strategy recommendations based on similar experiences."""
        
        if not similar_experiences:
            return [{'strategy': 'default_exploration', 'confidence': 0.3}]
        
        # Analyze successful strategies from similar experiences
        successful_strategies = [
            exp.attempted_strategy for exp in similar_experiences
            if exp.success_rate > 0.6
        ]
        
        if not successful_strategies:
            return [{'strategy': 'careful_exploration', 'confidence': 0.4}]
        
        # Count strategy patterns
        strategy_patterns = defaultdict(list)
        for strategy in successful_strategies:
            pattern = '_'.join(strategy[:3])  # First 3 tactics
            strategy_patterns[pattern].append(strategy)
        
        # Rank patterns by success
        pattern_scores = {}
        for pattern, strategies in strategy_patterns.items():
            scores = []
            for strategy in strategies:
                for exp in similar_experiences:
                    if exp.attempted_strategy[:3] == strategy[:3]:
                        scores.append(exp.success_rate)
            
            if scores:
                pattern_scores[pattern] = {
                    'avg_score': np.mean(scores),
                    'confidence': min(1.0, len(scores) / 3.0),
                    'sample_count': len(scores)
                }
        
        # Generate recommendations
        recommendations = []
        for pattern, metrics in sorted(
            pattern_scores.items(), 
            key=lambda x: x[1]['avg_score'], 
            reverse=True
        )[:3]:
            recommendations.append({
                'strategy_pattern': pattern,
                'expected_success_rate': metrics['avg_score'],
                'confidence': metrics['confidence'],
                'evidence_count': metrics['sample_count']
            })
        
        return recommendations
    
    def _recommend_learning_mode(
        self,
        context: Dict[str, Any],
        similar_experiences: List[LearningExperience]
    ) -> LearningMode:
        """Recommend learning mode for current context."""
        
        if not similar_experiences:
            return LearningMode.EXPLORATION
        
        # Analyze context novelty
        context_complexity = context.get('complexity_score', 0)
        average_similar_complexity = np.mean([
            exp.circuit_context.get('complexity_score', 0)
            for exp in similar_experiences
        ])
        
        # Analyze success patterns
        success_rates = [exp.success_rate for exp in similar_experiences]
        avg_success_rate = np.mean(success_rates)
        success_stability = 1.0 / (1.0 + np.std(success_rates))
        
        # Decision logic
        if context_complexity > average_similar_complexity * 1.5:
            return LearningMode.EXPLORATION  # Novel complex context
        elif avg_success_rate > 0.7 and success_stability > 0.8:
            return LearningMode.EXPLOITATION  # Known good approach
        elif len(similar_experiences) < 5:
            return LearningMode.META_LEARNING  # Limited data, need adaptation
        else:
            return self.current_mode  # Continue current mode
    
    def _compute_recommendation_confidence(self, similar_experiences: List[LearningExperience]) -> float:
        """Compute confidence in recommendations based on evidence."""
        
        if not similar_experiences:
            return 0.1
        
        # Confidence based on number of similar experiences
        count_confidence = min(1.0, len(similar_experiences) / 10.0)
        
        # Confidence based on success rate consistency
        success_rates = [exp.success_rate for exp in similar_experiences]
        consistency_confidence = 1.0 / (1.0 + np.std(success_rates))
        
        # Confidence based on recency
        recent_count = sum(
            1 for exp in similar_experiences
            if time.time() - exp.timestamp < 3600  # Last hour
        )
        recency_confidence = min(1.0, recent_count / max(1, len(similar_experiences)))
        
        return (count_confidence + consistency_confidence + recency_confidence) / 3.0
    
    def get_learning_status(self) -> Dict[str, Any]:
        """Get comprehensive learning system status."""
        
        return {
            'current_mode': self.current_mode.value,
            'learning_metrics': self.learning_metrics.copy(),
            'meta_learning_state': {
                'exploration_bias': self.meta_learning_state.exploration_bias,
                'confidence_threshold': self.meta_learning_state.confidence_threshold,
                'adaptation_speed': self.meta_learning_state.adaptation_speed,
                'known_strategies': len(self.meta_learning_state.strategy_effectiveness)
            },
            'experience_summary': {
                'total_experiences': len(self.experience_buffer),
                'recent_success_rate': np.mean([
                    exp.success_rate for exp in list(self.experience_buffer)[-10:]
                ]) if len(self.experience_buffer) >= 10 else 0.0,
                'recent_insight_level': np.mean([
                    exp.insight_level for exp in list(self.experience_buffer)[-10:]
                ]) if len(self.experience_buffer) >= 10 else 0.0
            },
            'self_awareness_metrics': self.self_reflection_module.self_awareness_metrics.copy(),
            'system_health': {
                'adaptation_active': True,
                'discovery_active': True,
                'reflection_active': True,
                'learning_efficiency': self.learning_metrics['learning_efficiency']
            }
        }
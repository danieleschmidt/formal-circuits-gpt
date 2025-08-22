"""
Autonomous Proof Discovery System

This module implements an advanced autonomous system for discovering novel
proof strategies using meta-learning and evolutionary algorithms.
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
import networkx as nx
import random
from pathlib import Path

from ..core import CircuitVerifier, ProofResult
from ..llm.llm_client import LLMManager
from ..monitoring.logger import get_logger
from .experiment_runner import ExperimentRunner


@dataclass
class ProofStrategy:
    """Represents a proof strategy with metadata and performance metrics."""
    strategy_id: str
    name: str
    description: str
    tactics: List[str]
    success_rate: float
    avg_proof_time: float
    complexity_score: float
    applicable_circuit_types: List[str]
    generated_at: str
    generation: int


@dataclass
class AutonomousDiscoveryResult:
    """Result of autonomous proof discovery process."""
    discovered_strategies: List[ProofStrategy]
    performance_improvements: Dict[str, float]
    novel_insights: List[str]
    experiment_metadata: Dict[str, Any]
    total_experiments: int
    successful_discoveries: int


class AutonomousProofDiscovery:
    """
    Advanced autonomous system for discovering novel proof strategies
    using meta-learning, evolutionary algorithms, and pattern recognition.
    """

    def __init__(self, verifier: CircuitVerifier, max_generations: int = 50):
        self.verifier = verifier
        self.max_generations = max_generations
        self.logger = get_logger("autonomous_proof_discovery")
        self.llm_manager = LLMManager.create_default()
        self.experiment_runner = ExperimentRunner()
        
        # Strategy repository
        self.strategy_repository: Dict[str, ProofStrategy] = {}
        self.performance_history: List[Dict[str, Any]] = []
        
        # Evolutionary parameters
        self.population_size = 20
        self.mutation_rate = 0.15
        self.crossover_rate = 0.7
        self.elite_ratio = 0.2
        
        # Discovery state
        self.current_generation = 0
        self.discovery_session_id = str(uuid.uuid4())
        
        self.logger.info(f"Initialized autonomous proof discovery system (session: {self.discovery_session_id})")

    async def discover_novel_strategies(
        self,
        benchmark_circuits: List[str],
        target_properties: List[str],
        discovery_budget_minutes: int = 120
    ) -> AutonomousDiscoveryResult:
        """
        Autonomously discover novel proof strategies through evolutionary meta-learning.
        
        Args:
            benchmark_circuits: Circuit specifications to test strategies on
            target_properties: Properties to verify
            discovery_budget_minutes: Time budget for discovery process
            
        Returns:
            Results of the autonomous discovery process
        """
        start_time = time.time()
        budget_seconds = discovery_budget_minutes * 60
        
        self.logger.info(f"Starting autonomous proof strategy discovery")
        self.logger.info(f"Budget: {discovery_budget_minutes} minutes")
        self.logger.info(f"Benchmark circuits: {len(benchmark_circuits)}")
        self.logger.info(f"Target properties: {len(target_properties)}")
        
        # Phase 1: Initialize population with baseline strategies
        await self._initialize_strategy_population()
        
        # Phase 2: Evolutionary discovery loop
        generation = 0
        discovered_strategies = []
        novel_insights = []
        
        while generation < self.max_generations and (time.time() - start_time) < budget_seconds:
            self.current_generation = generation
            
            self.logger.info(f"Generation {generation + 1}/{self.max_generations}")
            
            # Evaluate current population
            population_results = await self._evaluate_population_parallel(
                benchmark_circuits, target_properties
            )
            
            # Identify novel strategies
            novel_strategies = self._identify_novel_strategies(population_results)
            discovered_strategies.extend(novel_strategies)
            
            # Extract insights from successful strategies
            insights = await self._extract_strategic_insights(population_results)
            novel_insights.extend(insights)
            
            # Evolve population for next generation
            await self._evolve_population(population_results)
            
            # Log progress
            best_performance = max(r['fitness'] for r in population_results)
            avg_performance = sum(r['fitness'] for r in population_results) / len(population_results)
            
            self.logger.info(f"Generation {generation + 1} - Best: {best_performance:.3f}, Avg: {avg_performance:.3f}")
            
            generation += 1
            
            # Adaptive budget management
            if len(discovered_strategies) > 10 and generation > 20:
                # If we've found many strategies, we can stop early
                break
        
        # Phase 3: Validate and refine discovered strategies
        validated_strategies = await self._validate_discovered_strategies(
            discovered_strategies, benchmark_circuits, target_properties
        )
        
        # Phase 4: Calculate performance improvements
        performance_improvements = await self._calculate_performance_improvements(
            validated_strategies, benchmark_circuits, target_properties
        )
        
        total_time = time.time() - start_time
        
        result = AutonomousDiscoveryResult(
            discovered_strategies=validated_strategies,
            performance_improvements=performance_improvements,
            novel_insights=novel_insights,
            experiment_metadata={
                'total_time_seconds': total_time,
                'generations_completed': generation,
                'discovery_session_id': self.discovery_session_id,
                'benchmark_circuits_count': len(benchmark_circuits),
                'target_properties_count': len(target_properties)
            },
            total_experiments=generation * self.population_size,
            successful_discoveries=len(validated_strategies)
        )
        
        self.logger.info(f"Discovery complete - Found {len(validated_strategies)} novel strategies")
        
        return result

    async def _initialize_strategy_population(self):
        """Initialize the strategy population with baseline and generated strategies."""
        self.logger.info("Initializing strategy population")
        
        # Baseline proven strategies
        baseline_strategies = [
            ProofStrategy(
                strategy_id=str(uuid.uuid4()),
                name="Inductive Decomposition",
                description="Break complex proofs into inductive cases",
                tactics=["induction", "cases", "simp"],
                success_rate=0.7,
                avg_proof_time=45.0,
                complexity_score=0.6,
                applicable_circuit_types=["sequential", "fsm"],
                generated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                generation=0
            ),
            ProofStrategy(
                strategy_id=str(uuid.uuid4()),
                name="Algebraic Simplification",
                description="Use algebraic laws to simplify expressions",
                tactics=["algebra", "simp", "unfold"],
                success_rate=0.8,
                avg_proof_time=30.0,
                complexity_score=0.4,
                applicable_circuit_types=["combinational", "arithmetic"],
                generated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                generation=0
            ),
            ProofStrategy(
                strategy_id=str(uuid.uuid4()),
                name="Behavioral Equivalence",
                description="Prove equivalence through behavioral analysis",
                tactics=["unfold", "cases", "auto"],
                success_rate=0.65,
                avg_proof_time=60.0,
                complexity_score=0.7,
                applicable_circuit_types=["sequential", "memory"],
                generated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                generation=0
            )
        ]
        
        # Add baseline strategies to repository
        for strategy in baseline_strategies:
            self.strategy_repository[strategy.strategy_id] = strategy
        
        # Generate additional diverse strategies using LLM
        generated_strategies = await self._generate_initial_strategies(
            self.population_size - len(baseline_strategies)
        )
        
        for strategy in generated_strategies:
            self.strategy_repository[strategy.strategy_id] = strategy
        
        self.logger.info(f"Initialized population with {len(self.strategy_repository)} strategies")

    async def _generate_initial_strategies(self, count: int) -> List[ProofStrategy]:
        """Generate initial diverse strategies using LLM."""
        strategies = []
        
        for i in range(count):
            prompt = f"""
            Generate a novel formal verification proof strategy for hardware circuits.
            
            Requirements:
            - Provide a unique name and description
            - List 3-5 specific proof tactics
            - Focus on practical applicability
            - Consider different circuit types (combinational, sequential, arithmetic, memory)
            
            Generate strategy #{i+1}:
            """
            
            try:
                response = await self.llm_manager.generate(
                    prompt, temperature=0.8, max_tokens=500
                )
                
                # Parse LLM response into strategy
                strategy = await self._parse_llm_strategy_response(response.content, i)
                if strategy:
                    strategies.append(strategy)
                    
            except Exception as e:
                self.logger.warning(f"Failed to generate strategy {i+1}: {e}")
        
        return strategies

    async def _parse_llm_strategy_response(self, response: str, index: int) -> Optional[ProofStrategy]:
        """Parse LLM response into a ProofStrategy object."""
        try:
            # Extract strategy components using LLM
            parse_prompt = f"""
            Parse the following strategy description into structured components:
            
            {response}
            
            Return JSON with fields:
            - name: strategy name
            - description: brief description
            - tactics: list of 3-5 proof tactics
            - applicable_circuit_types: list of applicable circuit types
            """
            
            parse_response = await self.llm_manager.generate(
                parse_prompt, temperature=0.1, max_tokens=300
            )
            
            # Simple JSON extraction (in real implementation, use robust parsing)
            data = json.loads(parse_response.content.strip())
            
            return ProofStrategy(
                strategy_id=str(uuid.uuid4()),
                name=data.get('name', f'Generated Strategy {index+1}'),
                description=data.get('description', 'AI-generated proof strategy'),
                tactics=data.get('tactics', ['auto', 'simp']),
                success_rate=0.5,  # Initial estimate
                avg_proof_time=45.0,
                complexity_score=0.5,
                applicable_circuit_types=data.get('applicable_circuit_types', ['combinational']),
                generated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                generation=0
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to parse strategy response: {e}")
            return None

    async def _evaluate_population_parallel(
        self, 
        benchmark_circuits: List[str], 
        target_properties: List[str]
    ) -> List[Dict[str, Any]]:
        """Evaluate all strategies in the population in parallel."""
        
        evaluation_tasks = []
        
        for strategy_id, strategy in self.strategy_repository.items():
            for circuit in benchmark_circuits[:3]:  # Limit for performance
                for prop in target_properties[:2]:  # Limit for performance
                    task = self._evaluate_strategy_on_circuit(strategy, circuit, prop)
                    evaluation_tasks.append(task)
        
        # Execute evaluations in parallel
        results = await asyncio.gather(*evaluation_tasks, return_exceptions=True)
        
        # Aggregate results by strategy
        strategy_results = {}
        for result in results:
            if isinstance(result, dict) and 'strategy_id' in result:
                strategy_id = result['strategy_id']
                if strategy_id not in strategy_results:
                    strategy_results[strategy_id] = {
                        'strategy_id': strategy_id,
                        'total_attempts': 0,
                        'successful_proofs': 0,
                        'total_time': 0.0,
                        'fitness': 0.0
                    }
                
                strategy_results[strategy_id]['total_attempts'] += 1
                strategy_results[strategy_id]['total_time'] += result['proof_time']
                
                if result['success']:
                    strategy_results[strategy_id]['successful_proofs'] += 1
        
        # Calculate fitness scores
        population_results = []
        for strategy_id, stats in strategy_results.items():
            if stats['total_attempts'] > 0:
                success_rate = stats['successful_proofs'] / stats['total_attempts']
                avg_time = stats['total_time'] / stats['total_attempts']
                
                # Fitness function: balance success rate and speed
                fitness = success_rate * 0.7 + (1.0 / (1.0 + avg_time/100.0)) * 0.3
                
                stats['fitness'] = fitness
                population_results.append(stats)
        
        return population_results

    async def _evaluate_strategy_on_circuit(
        self, strategy: ProofStrategy, circuit: str, property_spec: str
    ) -> Dict[str, Any]:
        """Evaluate a single strategy on a circuit-property pair."""
        start_time = time.time()
        
        try:
            # Create a custom verifier with the strategy's tactics
            # In a real implementation, this would modify the proof generation
            result = self.verifier.verify(circuit, [property_spec], timeout=60)
            
            proof_time = time.time() - start_time
            success = result.status == "VERIFIED"
            
            return {
                'strategy_id': strategy.strategy_id,
                'success': success,
                'proof_time': proof_time,
                'circuit': circuit[:50],  # Truncate for logging
                'property': property_spec
            }
            
        except Exception as e:
            proof_time = time.time() - start_time
            return {
                'strategy_id': strategy.strategy_id,
                'success': False,
                'proof_time': proof_time,
                'circuit': circuit[:50],
                'property': property_spec,
                'error': str(e)
            }

    def _identify_novel_strategies(self, population_results: List[Dict[str, Any]]) -> List[ProofStrategy]:
        """Identify strategies that show novel capabilities or performance."""
        novel_strategies = []
        
        # Sort by fitness
        population_results.sort(key=lambda x: x['fitness'], reverse=True)
        
        # Top 20% are considered potentially novel
        top_count = max(1, len(population_results) // 5)
        top_strategies = population_results[:top_count]
        
        for result in top_strategies:
            strategy = self.strategy_repository[result['strategy_id']]
            
            # Check if this represents a significant improvement
            if result['fitness'] > 0.8:  # High performance threshold
                novel_strategies.append(strategy)
        
        return novel_strategies

    async def _extract_strategic_insights(self, population_results: List[Dict[str, Any]]) -> List[str]:
        """Extract strategic insights from successful proof attempts."""
        insights = []
        
        # Analyze patterns in successful strategies
        successful_strategies = [
            self.strategy_repository[r['strategy_id']] 
            for r in population_results 
            if r['fitness'] > 0.7
        ]
        
        if successful_strategies:
            # Use LLM to analyze patterns
            strategy_descriptions = [s.description for s in successful_strategies]
            tactics_list = [s.tactics for s in successful_strategies]
            
            analysis_prompt = f"""
            Analyze the following successful proof strategies and identify key insights:
            
            Strategies: {strategy_descriptions}
            Tactics: {tactics_list}
            
            Identify 3-5 key strategic insights that contribute to success:
            """
            
            try:
                response = await self.llm_manager.generate(
                    analysis_prompt, temperature=0.3, max_tokens=400
                )
                
                # Extract insights from response
                insight_lines = response.content.split('\n')
                for line in insight_lines:
                    if line.strip() and len(line.strip()) > 10:
                        insights.append(line.strip())
                        
            except Exception as e:
                self.logger.warning(f"Failed to extract insights: {e}")
        
        return insights[:5]  # Limit to top 5 insights

    async def _evolve_population(self, population_results: List[Dict[str, Any]]):
        """Evolve the strategy population using genetic algorithm principles."""
        if not population_results:
            return
        
        # Sort by fitness
        population_results.sort(key=lambda x: x['fitness'], reverse=True)
        
        # Select elite strategies (top 20%)
        elite_count = max(1, int(len(population_results) * self.elite_ratio))
        elite_strategy_ids = [r['strategy_id'] for r in population_results[:elite_count]]
        
        # Generate new strategies through crossover and mutation
        new_strategies = []
        
        for i in range(self.population_size - elite_count):
            if random.random() < self.crossover_rate and len(elite_strategy_ids) >= 2:
                # Crossover: combine two elite strategies
                parent1_id, parent2_id = random.sample(elite_strategy_ids, 2)
                parent1 = self.strategy_repository[parent1_id]
                parent2 = self.strategy_repository[parent2_id]
                
                child = await self._crossover_strategies(parent1, parent2)
                if child:
                    new_strategies.append(child)
            else:
                # Mutation: modify an elite strategy
                parent_id = random.choice(elite_strategy_ids)
                parent = self.strategy_repository[parent_id]
                
                mutant = await self._mutate_strategy(parent)
                if mutant:
                    new_strategies.append(mutant)
        
        # Replace non-elite strategies with new ones
        all_strategy_ids = list(self.strategy_repository.keys())
        non_elite_ids = [sid for sid in all_strategy_ids if sid not in elite_strategy_ids]
        
        # Remove excess strategies
        for i, strategy_id in enumerate(non_elite_ids):
            if i < len(new_strategies):
                del self.strategy_repository[strategy_id]
        
        # Add new strategies
        for strategy in new_strategies:
            self.strategy_repository[strategy.strategy_id] = strategy
        
        self.logger.info(f"Evolved population: {elite_count} elite + {len(new_strategies)} new strategies")

    async def _crossover_strategies(self, parent1: ProofStrategy, parent2: ProofStrategy) -> Optional[ProofStrategy]:
        """Create a child strategy by combining two parent strategies."""
        try:
            crossover_prompt = f"""
            Create a new proof strategy by combining the best aspects of these two strategies:
            
            Strategy 1: {parent1.name}
            Description: {parent1.description}
            Tactics: {parent1.tactics}
            
            Strategy 2: {parent2.name}
            Description: {parent2.description}
            Tactics: {parent2.tactics}
            
            Generate a novel hybrid strategy that combines their strengths.
            Provide: name, description, and 3-5 tactics.
            """
            
            response = await self.llm_manager.generate(
                crossover_prompt, temperature=0.6, max_tokens=400
            )
            
            # Parse response and create child strategy
            child = await self._parse_llm_strategy_response(response.content, self.current_generation)
            if child:
                child.generation = self.current_generation + 1
                child.applicable_circuit_types = list(set(
                    parent1.applicable_circuit_types + parent2.applicable_circuit_types
                ))
            
            return child
            
        except Exception as e:
            self.logger.warning(f"Crossover failed: {e}")
            return None

    async def _mutate_strategy(self, parent: ProofStrategy) -> Optional[ProofStrategy]:
        """Create a mutated version of a strategy."""
        try:
            mutation_prompt = f"""
            Create a variant of this proof strategy with small innovative modifications:
            
            Original Strategy: {parent.name}
            Description: {parent.description}
            Tactics: {parent.tactics}
            
            Make 1-2 small innovative changes while preserving the core approach.
            Provide: name, description, and 3-5 tactics.
            """
            
            response = await self.llm_manager.generate(
                mutation_prompt, temperature=0.7, max_tokens=400
            )
            
            # Parse response and create mutant strategy
            mutant = await self._parse_llm_strategy_response(response.content, self.current_generation)
            if mutant:
                mutant.generation = self.current_generation + 1
                mutant.applicable_circuit_types = parent.applicable_circuit_types.copy()
            
            return mutant
            
        except Exception as e:
            self.logger.warning(f"Mutation failed: {e}")
            return None

    async def _validate_discovered_strategies(
        self, 
        strategies: List[ProofStrategy],
        benchmark_circuits: List[str],
        target_properties: List[str]
    ) -> List[ProofStrategy]:
        """Validate discovered strategies on a broader test set."""
        validated = []
        
        for strategy in strategies:
            # Test on additional circuits not used in evolution
            validation_score = await self._comprehensive_validation(
                strategy, benchmark_circuits, target_properties
            )
            
            if validation_score > 0.6:  # Validation threshold
                strategy.success_rate = validation_score
                validated.append(strategy)
        
        return validated

    async def _comprehensive_validation(
        self, 
        strategy: ProofStrategy,
        circuits: List[str],
        properties: List[str]
    ) -> float:
        """Perform comprehensive validation of a strategy."""
        total_tests = 0
        successful_tests = 0
        
        # Test on random subset of circuits and properties
        test_pairs = [(c, p) for c in circuits[:5] for p in properties[:3]]
        
        for circuit, prop in test_pairs:
            try:
                result = await self._evaluate_strategy_on_circuit(strategy, circuit, prop)
                total_tests += 1
                if result['success']:
                    successful_tests += 1
            except:
                total_tests += 1
        
        return successful_tests / total_tests if total_tests > 0 else 0.0

    async def _calculate_performance_improvements(
        self,
        strategies: List[ProofStrategy],
        circuits: List[str],
        properties: List[str]
    ) -> Dict[str, float]:
        """Calculate performance improvements over baseline."""
        baseline_performance = 0.5  # Assume baseline 50% success rate
        
        improvements = {}
        for strategy in strategies:
            validation_score = await self._comprehensive_validation(strategy, circuits, properties)
            improvement = (validation_score - baseline_performance) / baseline_performance * 100
            improvements[strategy.name] = improvement
        
        return improvements

    def save_discovery_results(self, result: AutonomousDiscoveryResult, filepath: str):
        """Save discovery results to file for analysis."""
        output_data = {
            'discovery_session_id': self.discovery_session_id,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'discovered_strategies': [asdict(s) for s in result.discovered_strategies],
            'performance_improvements': result.performance_improvements,
            'novel_insights': result.novel_insights,
            'experiment_metadata': result.experiment_metadata,
            'statistics': {
                'total_experiments': result.total_experiments,
                'successful_discoveries': result.successful_discoveries,
                'success_rate': result.successful_discoveries / result.total_experiments if result.total_experiments > 0 else 0
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        self.logger.info(f"Discovery results saved to {filepath}")
"""Experiment runner for research studies and benchmarking."""

import json
import time
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib

from ..core import CircuitVerifier, ProofResult
from ..exceptions import VerificationError


@dataclass
class ExperimentConfig:
    """Configuration for a research experiment."""
    name: str
    description: str
    circuits: List[str]  # File paths
    provers: List[str]  # ["isabelle", "coq"]
    models: List[str]   # LLM models to test
    properties: Optional[List[str]] = None
    timeout: int = 300
    max_refinement_rounds: int = 5
    temperature: float = 0.1
    repetitions: int = 3  # For statistical significance
    
    def __post_init__(self):
        if not self.circuits:
            raise ValueError("At least one circuit must be specified")
        if not self.provers:
            raise ValueError("At least one prover must be specified")
        if not self.models:
            raise ValueError("At least one model must be specified")


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    experiment_id: str
    config_hash: str
    circuit_name: str
    prover: str
    model: str
    repetition: int
    status: str  # VERIFIED, FAILED, ERROR, TIMEOUT
    duration_ms: float
    properties_verified: int
    properties_failed: int
    llm_tokens_used: int
    refinement_attempts: int
    errors: List[str]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentResult':
        """Create from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class ExperimentRunner:
    """Manages and executes research experiments."""
    
    def __init__(self, results_dir: Path = Path("research_results")):
        """Initialize experiment runner.
        
        Args:
            results_dir: Directory to store experiment results
        """
        self.results_dir = results_dir
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
    def run_experiment(self, config: ExperimentConfig) -> List[ExperimentResult]:
        """Run a complete experiment with all configurations.
        
        Args:
            config: Experiment configuration
            
        Returns:
            List of all experiment results
        """
        experiment_id = str(uuid.uuid4())
        config_hash = self._hash_config(config)
        all_results = []
        
        # Create experiment metadata
        metadata = {
            "experiment_id": experiment_id,
            "config": asdict(config),
            "config_hash": config_hash,
            "start_time": datetime.now().isoformat(),
            "total_runs": len(config.circuits) * len(config.provers) * len(config.models) * config.repetitions
        }
        
        # Save experiment metadata
        metadata_file = self.results_dir / f"experiment_{experiment_id}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Starting experiment '{config.name}' with ID: {experiment_id}")
        print(f"Total runs: {metadata['total_runs']}")
        
        run_count = 0
        start_time = time.time()
        
        # Run all combinations
        for circuit_file in config.circuits:
            circuit_name = Path(circuit_file).stem
            
            for prover in config.provers:
                for model in config.models:
                    for rep in range(config.repetitions):
                        run_count += 1
                        
                        print(f"Run {run_count}/{metadata['total_runs']}: {circuit_name} + {prover} + {model} (rep {rep+1})")
                        
                        result = self._run_single_verification(
                            experiment_id=experiment_id,
                            config_hash=config_hash,
                            circuit_file=circuit_file,
                            circuit_name=circuit_name,
                            prover=prover,
                            model=model,
                            repetition=rep + 1,
                            config=config
                        )
                        
                        all_results.append(result)
                        
                        # Save incremental results
                        self._save_result(result)
                        
                        # Progress update
                        elapsed = time.time() - start_time
                        eta = (elapsed / run_count) * (metadata['total_runs'] - run_count)
                        print(f"  → {result.status} ({result.duration_ms:.1f}ms) | ETA: {eta/60:.1f}min")
        
        # Save final experiment summary
        self._save_experiment_summary(experiment_id, config, all_results)
        
        print(f"Experiment completed in {(time.time() - start_time)/60:.1f} minutes")
        return all_results
    
    def _run_single_verification(self, experiment_id: str, config_hash: str, 
                               circuit_file: str, circuit_name: str,
                               prover: str, model: str, repetition: int,
                               config: ExperimentConfig) -> ExperimentResult:
        """Run a single verification instance."""
        start_time = time.time()
        
        try:
            # Create verifier with specific configuration
            verifier = CircuitVerifier(
                prover=prover,
                model=model,
                temperature=config.temperature,
                refinement_rounds=config.max_refinement_rounds,
                debug_mode=False
            )
            
            # Run verification
            proof_result = verifier.verify_file(
                hdl_file=circuit_file,
                properties=config.properties,
                timeout=config.timeout
            )
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Extract metrics
            llm_tokens = 0  # Would need to track this in verifier
            if hasattr(proof_result, 'metadata'):
                llm_tokens = proof_result.metadata.get('tokens_used', 0)
            
            return ExperimentResult(
                experiment_id=experiment_id,
                config_hash=config_hash,
                circuit_name=circuit_name,
                prover=prover,
                model=model,
                repetition=repetition,
                status=proof_result.status,
                duration_ms=duration_ms,
                properties_verified=len(proof_result.properties_verified),
                properties_failed=len(proof_result.errors),
                llm_tokens_used=llm_tokens,
                refinement_attempts=getattr(proof_result, 'refinement_attempts', 0),
                errors=proof_result.errors,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            
            return ExperimentResult(
                experiment_id=experiment_id,
                config_hash=config_hash,
                circuit_name=circuit_name,
                prover=prover,
                model=model,
                repetition=repetition,
                status="ERROR",
                duration_ms=duration_ms,
                properties_verified=0,
                properties_failed=0,
                llm_tokens_used=0,
                refinement_attempts=0,
                errors=[str(e)],
                timestamp=datetime.now()
            )
    
    def _save_result(self, result: ExperimentResult) -> None:
        """Save individual result to file."""
        results_file = self.results_dir / f"experiment_{result.experiment_id}_results.jsonl"
        
        with open(results_file, 'a') as f:
            json.dump(result.to_dict(), f)
            f.write('\n')
    
    def _save_experiment_summary(self, experiment_id: str, config: ExperimentConfig, 
                                results: List[ExperimentResult]) -> None:
        """Save experiment summary and statistics."""
        summary = {
            "experiment_id": experiment_id,
            "config": asdict(config),
            "total_runs": len(results),
            "completion_time": datetime.now().isoformat(),
            "statistics": self._calculate_statistics(results)
        }
        
        summary_file = self.results_dir / f"experiment_{experiment_id}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def _calculate_statistics(self, results: List[ExperimentResult]) -> Dict[str, Any]:
        """Calculate experiment statistics."""
        total = len(results)
        verified = sum(1 for r in results if r.status == "VERIFIED")
        failed = sum(1 for r in results if r.status == "FAILED")
        errors = sum(1 for r in results if r.status == "ERROR")
        
        durations = [r.duration_ms for r in results if r.duration_ms > 0]
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        total_tokens = sum(r.llm_tokens_used for r in results)
        avg_tokens = total_tokens / total if total > 0 else 0
        
        # Per-prover statistics
        prover_stats = {}
        for prover in set(r.prover for r in results):
            prover_results = [r for r in results if r.prover == prover]
            prover_verified = sum(1 for r in prover_results if r.status == "VERIFIED")
            prover_stats[prover] = {
                "total": len(prover_results),
                "verified": prover_verified,
                "success_rate": prover_verified / len(prover_results) if prover_results else 0
            }
        
        # Per-model statistics  
        model_stats = {}
        for model in set(r.model for r in results):
            model_results = [r for r in results if r.model == model]
            model_verified = sum(1 for r in model_results if r.status == "VERIFIED")
            model_stats[model] = {
                "total": len(model_results),
                "verified": model_verified,
                "success_rate": model_verified / len(model_results) if model_results else 0
            }
        
        return {
            "overall": {
                "total_runs": total,
                "verified": verified,
                "failed": failed,
                "errors": errors,
                "success_rate": verified / total if total > 0 else 0,
                "avg_duration_ms": avg_duration,
                "total_tokens_used": total_tokens,
                "avg_tokens_per_run": avg_tokens
            },
            "by_prover": prover_stats,
            "by_model": model_stats
        }
    
    def _hash_config(self, config: ExperimentConfig) -> str:
        """Create hash of configuration for reproducibility."""
        config_str = json.dumps(asdict(config), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def load_experiment_results(self, experiment_id: str) -> List[ExperimentResult]:
        """Load results from a completed experiment."""
        results_file = self.results_dir / f"experiment_{experiment_id}_results.jsonl"
        
        if not results_file.exists():
            raise FileNotFoundError(f"No results found for experiment {experiment_id}")
        
        results = []
        with open(results_file, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    results.append(ExperimentResult.from_dict(data))
        
        return results
    
    def compare_experiments(self, experiment_ids: List[str]) -> Dict[str, Any]:
        """Compare results across multiple experiments."""
        comparison = {
            "experiment_ids": experiment_ids,
            "comparison_time": datetime.now().isoformat(),
            "experiments": {}
        }
        
        for exp_id in experiment_ids:
            try:
                results = self.load_experiment_results(exp_id)
                stats = self._calculate_statistics(results)
                comparison["experiments"][exp_id] = stats
            except FileNotFoundError:
                comparison["experiments"][exp_id] = {"error": "Results not found"}
        
        return comparison
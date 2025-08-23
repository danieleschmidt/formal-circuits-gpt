"""
Intelligent Auto-Scaling System

Advanced auto-scaling with predictive analytics, ML-driven resource optimization,
and intelligent workload distribution for formal verification at scale.
"""

import asyncio
import json
import time
import uuid
import math
import statistics
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple, Callable
from enum import Enum
from collections import defaultdict, deque
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading

from ..monitoring.logger import get_logger
from ..llm.llm_client import LLMManager
from ..monitoring.realtime_observability import RealTimeObservability, MetricType


class ScalingDirection(Enum):
    """Direction of scaling operations."""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"


class ResourceType(Enum):
    """Types of resources that can be scaled."""
    COMPUTE = "compute"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    GPU = "gpu"


class PredictionModel(Enum):
    """Types of prediction models."""
    LINEAR_REGRESSION = "linear_regression"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    NEURAL_NETWORK = "neural_network"
    LLM_ASSISTED = "llm_assisted"


@dataclass
class ResourceMetrics:
    """Current resource utilization metrics."""
    timestamp: float
    cpu_utilization: float
    memory_utilization: float
    network_io: float
    disk_io: float
    gpu_utilization: float
    active_connections: int
    queue_depth: int
    throughput: float
    latency: float


@dataclass
class ScalingEvent:
    """Represents a scaling event."""
    event_id: str
    timestamp: float
    resource_type: ResourceType
    direction: ScalingDirection
    from_capacity: float
    to_capacity: float
    reason: str
    predicted_demand: float
    confidence: float
    execution_time: float
    success: bool


@dataclass
class PredictionResult:
    """Result of demand prediction."""
    timestamp: float
    predicted_value: float
    confidence_interval: Tuple[float, float]
    confidence_score: float
    model_used: PredictionModel
    factors: Dict[str, float]
    time_horizon: int


@dataclass
class AutoScalingConfig:
    """Configuration for auto-scaling policies."""
    resource_type: ResourceType
    min_capacity: float
    max_capacity: float
    target_utilization: float
    scale_up_threshold: float
    scale_down_threshold: float
    scale_up_cooldown: int
    scale_down_cooldown: int
    prediction_window: int
    aggressive_scaling: bool = False


class IntelligentAutoScaling:
    """
    Intelligent auto-scaling system with predictive analytics and ML-driven optimization.
    """

    def __init__(
        self,
        observability: RealTimeObservability,
        prediction_horizon: int = 600,  # 10 minutes
        learning_enabled: bool = True
    ):
        self.observability = observability
        self.prediction_horizon = prediction_horizon
        self.learning_enabled = learning_enabled
        
        self.logger = get_logger("intelligent_auto_scaling")
        self.llm_manager = LLMManager.create_default()
        
        # Auto-scaling state
        self.scaling_configs: Dict[ResourceType, AutoScalingConfig] = {}
        self.current_capacity: Dict[ResourceType, float] = {}
        self.scaling_history: List[ScalingEvent] = []
        self.resource_metrics_history: deque = deque(maxlen=10000)
        
        # Prediction models
        self.prediction_models: Dict[str, Any] = {}
        self.model_performance: Dict[str, List[float]] = defaultdict(list)
        
        # Learning and optimization
        self.learning_data: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        self.optimization_feedback: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.scaling_metrics = {
            "total_scaling_events": 0,
            "successful_predictions": 0,
            "total_predictions": 0,
            "average_prediction_accuracy": 0.0,
            "cost_savings": 0.0,
            "performance_improvements": 0.0
        }
        
        # Background tasks
        self._monitoring_task = None
        self._prediction_task = None
        self._optimization_task = None
        self._shutdown_event = asyncio.Event()
        
        self.logger.info("Intelligent auto-scaling system initialized")

    async def start(self):
        """Start the auto-scaling system."""
        self.logger.info("Starting intelligent auto-scaling system")
        
        # Initialize default scaling configurations
        await self._initialize_default_configs()
        
        # Initialize prediction models
        await self._initialize_prediction_models()
        
        # Start background tasks
        self._monitoring_task = asyncio.create_task(self._resource_monitoring_loop())
        self._prediction_task = asyncio.create_task(self._prediction_loop())
        self._optimization_task = asyncio.create_task(self._optimization_loop())
        
        self.logger.info("Auto-scaling system started successfully")

    async def stop(self):
        """Stop the auto-scaling system."""
        self.logger.info("Stopping auto-scaling system")
        
        self._shutdown_event.set()
        
        # Cancel background tasks
        for task in [self._monitoring_task, self._prediction_task, self._optimization_task]:
            if task:
                task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(
            self._monitoring_task, self._prediction_task, self._optimization_task,
            return_exceptions=True
        )

    # === CONFIGURATION MANAGEMENT ===

    def add_scaling_config(self, config: AutoScalingConfig):
        """Add or update a scaling configuration."""
        self.scaling_configs[config.resource_type] = config
        
        # Initialize current capacity if not set
        if config.resource_type not in self.current_capacity:
            self.current_capacity[config.resource_type] = config.min_capacity
        
        self.logger.info(f"Added scaling config for {config.resource_type.value}")

    def update_scaling_config(
        self, 
        resource_type: ResourceType, 
        **updates
    ):
        """Update scaling configuration parameters."""
        if resource_type in self.scaling_configs:
            config = self.scaling_configs[resource_type]
            for key, value in updates.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            self.logger.info(f"Updated scaling config for {resource_type.value}: {updates}")

    # === DEMAND PREDICTION ===

    async def predict_demand(
        self, 
        resource_type: ResourceType, 
        time_horizon: int = None
    ) -> PredictionResult:
        """Predict future resource demand using ML models."""
        time_horizon = time_horizon or self.prediction_horizon
        
        # Get historical data
        historical_data = self._get_historical_data(resource_type, lookback_seconds=3600)
        
        if len(historical_data) < 10:
            # Not enough data for prediction
            return PredictionResult(
                timestamp=time.time(),
                predicted_value=self.current_capacity.get(resource_type, 1.0),
                confidence_interval=(0.5, 1.5),
                confidence_score=0.3,
                model_used=PredictionModel.LINEAR_REGRESSION,
                factors={},
                time_horizon=time_horizon
            )
        
        # Try multiple prediction models and ensemble
        predictions = []
        
        # Linear regression prediction
        linear_pred = await self._predict_linear_regression(historical_data, time_horizon)
        predictions.append(linear_pred)
        
        # Exponential smoothing prediction
        exp_pred = await self._predict_exponential_smoothing(historical_data, time_horizon)
        predictions.append(exp_pred)
        
        # LLM-assisted prediction for complex patterns
        if self.learning_enabled and len(historical_data) >= 50:
            llm_pred = await self._predict_with_llm(historical_data, resource_type, time_horizon)
            if llm_pred:
                predictions.append(llm_pred)
        
        # Ensemble prediction
        ensemble_result = self._ensemble_predictions(predictions, time_horizon)
        
        # Record prediction for later evaluation
        self.scaling_metrics["total_predictions"] += 1
        
        return ensemble_result

    async def _predict_linear_regression(
        self, 
        historical_data: List[Tuple[float, float]], 
        time_horizon: int
    ) -> PredictionResult:
        """Simple linear regression prediction."""
        if len(historical_data) < 2:
            return None
        
        # Extract timestamps and values
        timestamps = [t for t, v in historical_data]
        values = [v for t, v in historical_data]
        
        # Normalize timestamps
        base_time = timestamps[0]
        norm_timestamps = [(t - base_time) / 3600.0 for t in timestamps]  # Hours
        
        # Simple linear regression
        n = len(norm_timestamps)
        sum_x = sum(norm_timestamps)
        sum_y = sum(values)
        sum_xy = sum(x * y for x, y in zip(norm_timestamps, values))
        sum_x2 = sum(x * x for x in norm_timestamps)
        
        if n * sum_x2 - sum_x * sum_x == 0:
            # Cannot compute slope
            predicted_value = statistics.mean(values)
        else:
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            intercept = (sum_y - slope * sum_x) / n
            
            # Predict for future time
            future_time_norm = (time.time() + time_horizon - base_time) / 3600.0
            predicted_value = slope * future_time_norm + intercept
        
        # Calculate confidence interval based on variance
        variance = statistics.variance(values) if len(values) > 1 else 0.1
        confidence_interval = (
            predicted_value - 1.96 * math.sqrt(variance),
            predicted_value + 1.96 * math.sqrt(variance)
        )
        
        return PredictionResult(
            timestamp=time.time(),
            predicted_value=max(0.0, predicted_value),
            confidence_interval=confidence_interval,
            confidence_score=0.7,
            model_used=PredictionModel.LINEAR_REGRESSION,
            factors={"trend": slope if 'slope' in locals() else 0.0},
            time_horizon=time_horizon
        )

    async def _predict_exponential_smoothing(
        self, 
        historical_data: List[Tuple[float, float]], 
        time_horizon: int
    ) -> PredictionResult:
        """Exponential smoothing prediction."""
        if len(historical_data) < 3:
            return None
        
        values = [v for t, v in historical_data]
        
        # Simple exponential smoothing
        alpha = 0.3  # Smoothing parameter
        smoothed_values = [values[0]]
        
        for i in range(1, len(values)):
            smoothed = alpha * values[i] + (1 - alpha) * smoothed_values[-1]
            smoothed_values.append(smoothed)
        
        # Prediction is the last smoothed value
        predicted_value = smoothed_values[-1]
        
        # Calculate prediction error for confidence
        errors = [abs(values[i] - smoothed_values[i]) for i in range(1, len(values))]
        mean_error = statistics.mean(errors) if errors else 0.1
        
        confidence_interval = (
            predicted_value - 1.96 * mean_error,
            predicted_value + 1.96 * mean_error
        )
        
        return PredictionResult(
            timestamp=time.time(),
            predicted_value=max(0.0, predicted_value),
            confidence_interval=confidence_interval,
            confidence_score=0.6,
            model_used=PredictionModel.EXPONENTIAL_SMOOTHING,
            factors={"smoothing_alpha": alpha},
            time_horizon=time_horizon
        )

    async def _predict_with_llm(
        self, 
        historical_data: List[Tuple[float, float]], 
        resource_type: ResourceType, 
        time_horizon: int
    ) -> Optional[PredictionResult]:
        """Use LLM for complex pattern prediction."""
        try:
            # Prepare data summary for LLM
            values = [v for t, v in historical_data[-50:]]  # Last 50 points
            
            data_summary = self._create_data_summary(values)
            
            prompt = f"""
            Analyze this time series data for {resource_type.value} resource utilization and predict future demand.
            
            Historical Data Summary:
            {data_summary}
            
            Current patterns detected:
            - Recent trend: {self._detect_trend(values)}
            - Seasonality: {self._detect_seasonality(values)}
            - Volatility: {statistics.stdev(values) if len(values) > 1 else 0:.3f}
            
            Predict the resource utilization {time_horizon // 60} minutes from now.
            
            Consider:
            1. Current trends and patterns
            2. Time of day effects
            3. Historical variations
            4. Workload characteristics
            
            Return JSON with:
            - predicted_value: float (0.0 to 1.0+)
            - confidence: float (0.0 to 1.0)
            - reasoning: string
            - key_factors: array of strings
            """
            
            response = await self.llm_manager.generate(
                prompt, temperature=0.2, max_tokens=400
            )
            
            # Parse LLM response
            prediction = self._parse_llm_prediction_response(response.content, time_horizon)
            return prediction
            
        except Exception as e:
            self.logger.warning(f"LLM prediction failed: {e}")
            return None

    def _create_data_summary(self, values: List[float]) -> str:
        """Create a summary of time series data for LLM analysis."""
        if len(values) < 5:
            return "Insufficient data for analysis"
        
        summary = f"""
        Data Points: {len(values)}
        Mean: {statistics.mean(values):.3f}
        Std Dev: {statistics.stdev(values) if len(values) > 1 else 0:.3f}
        Min: {min(values):.3f}
        Max: {max(values):.3f}
        Recent values: {values[-10:]}
        """
        
        return summary

    def _detect_trend(self, values: List[float]) -> str:
        """Detect trend in time series data."""
        if len(values) < 5:
            return "insufficient_data"
        
        # Simple trend detection using first and last values
        first_quarter = statistics.mean(values[:len(values)//4])
        last_quarter = statistics.mean(values[3*len(values)//4:])
        
        ratio = last_quarter / first_quarter if first_quarter > 0 else 1.0
        
        if ratio > 1.1:
            return "increasing"
        elif ratio < 0.9:
            return "decreasing"
        else:
            return "stable"

    def _detect_seasonality(self, values: List[float]) -> str:
        """Detect seasonal patterns in data."""
        if len(values) < 20:
            return "insufficient_data"
        
        # Simple seasonality detection
        # Check for periodic patterns (simplified)
        autocorrelations = []
        for lag in [1, 2, 5, 10]:
            if lag < len(values):
                corr = self._calculate_autocorrelation(values, lag)
                autocorrelations.append(corr)
        
        max_autocorr = max(autocorrelations) if autocorrelations else 0
        
        if max_autocorr > 0.5:
            return "strong_seasonal"
        elif max_autocorr > 0.3:
            return "weak_seasonal"
        else:
            return "no_seasonality"

    def _calculate_autocorrelation(self, values: List[float], lag: int) -> float:
        """Calculate autocorrelation at given lag."""
        if lag >= len(values):
            return 0.0
        
        n = len(values) - lag
        if n <= 1:
            return 0.0
        
        mean_val = statistics.mean(values)
        
        numerator = sum((values[i] - mean_val) * (values[i + lag] - mean_val) for i in range(n))
        denominator = sum((values[i] - mean_val) ** 2 for i in range(len(values)))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator

    def _parse_llm_prediction_response(
        self, 
        response: str, 
        time_horizon: int
    ) -> Optional[PredictionResult]:
        """Parse LLM response into prediction result."""
        try:
            # Extract JSON from response
            import json
            
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                data = json.loads(json_str)
                
                predicted_value = float(data.get("predicted_value", 0.5))
                confidence = float(data.get("confidence", 0.7))
                
                # Calculate confidence interval
                uncertainty = (1.0 - confidence) * predicted_value
                confidence_interval = (
                    predicted_value - uncertainty,
                    predicted_value + uncertainty
                )
                
                factors = {
                    "llm_reasoning": data.get("reasoning", ""),
                    "key_factors": data.get("key_factors", [])
                }
                
                return PredictionResult(
                    timestamp=time.time(),
                    predicted_value=max(0.0, predicted_value),
                    confidence_interval=confidence_interval,
                    confidence_score=confidence,
                    model_used=PredictionModel.LLM_ASSISTED,
                    factors=factors,
                    time_horizon=time_horizon
                )
        
        except Exception as e:
            self.logger.warning(f"Failed to parse LLM prediction: {e}")
        
        return None

    def _ensemble_predictions(
        self, 
        predictions: List[PredictionResult], 
        time_horizon: int
    ) -> PredictionResult:
        """Combine multiple predictions into ensemble result."""
        if not predictions:
            # Default prediction
            return PredictionResult(
                timestamp=time.time(),
                predicted_value=1.0,
                confidence_interval=(0.5, 1.5),
                confidence_score=0.3,
                model_used=PredictionModel.LINEAR_REGRESSION,
                factors={},
                time_horizon=time_horizon
            )
        
        # Weight predictions by confidence
        weights = [p.confidence_score for p in predictions]
        total_weight = sum(weights)
        
        if total_weight == 0:
            # All predictions have zero confidence
            weighted_prediction = statistics.mean([p.predicted_value for p in predictions])
            ensemble_confidence = 0.3
        else:
            weighted_prediction = sum(
                p.predicted_value * w for p, w in zip(predictions, weights)
            ) / total_weight
            
            ensemble_confidence = total_weight / len(predictions)
        
        # Ensemble confidence interval
        all_intervals = [p.confidence_interval for p in predictions]
        lower_bounds = [interval[0] for interval in all_intervals]
        upper_bounds = [interval[1] for interval in all_intervals]
        
        ensemble_interval = (
            statistics.mean(lower_bounds),
            statistics.mean(upper_bounds)
        )
        
        # Combine factors from all models
        ensemble_factors = {}
        for prediction in predictions:
            ensemble_factors.update(prediction.factors)
        ensemble_factors["ensemble_size"] = len(predictions)
        
        return PredictionResult(
            timestamp=time.time(),
            predicted_value=max(0.0, weighted_prediction),
            confidence_interval=ensemble_interval,
            confidence_score=min(1.0, ensemble_confidence),
            model_used=PredictionModel.NEURAL_NETWORK,  # Representing ensemble
            factors=ensemble_factors,
            time_horizon=time_horizon
        )

    # === SCALING DECISIONS ===

    async def make_scaling_decision(
        self, 
        resource_type: ResourceType
    ) -> Optional[ScalingEvent]:
        """Make intelligent scaling decision based on current state and predictions."""
        if resource_type not in self.scaling_configs:
            return None
        
        config = self.scaling_configs[resource_type]
        current_capacity = self.current_capacity.get(resource_type, config.min_capacity)
        
        # Get current utilization
        current_metrics = await self._get_current_resource_metrics()
        current_utilization = self._get_utilization_for_resource(current_metrics, resource_type)
        
        # Get demand prediction
        prediction = await self.predict_demand(resource_type)
        predicted_demand = prediction.predicted_value
        
        # Check if scaling is needed
        scaling_decision = self._evaluate_scaling_need(
            config, current_capacity, current_utilization, predicted_demand
        )
        
        if scaling_decision["action"] == ScalingDirection.STABLE:
            return None
        
        # Calculate new capacity
        new_capacity = self._calculate_new_capacity(
            config, current_capacity, scaling_decision, prediction
        )
        
        # Create scaling event
        scaling_event = ScalingEvent(
            event_id=str(uuid.uuid4()),
            timestamp=time.time(),
            resource_type=resource_type,
            direction=scaling_decision["action"],
            from_capacity=current_capacity,
            to_capacity=new_capacity,
            reason=scaling_decision["reason"],
            predicted_demand=predicted_demand,
            confidence=prediction.confidence_score,
            execution_time=0.0,
            success=False
        )
        
        # Execute scaling
        success = await self._execute_scaling(scaling_event)
        scaling_event.success = success
        
        if success:
            self.current_capacity[resource_type] = new_capacity
            self.scaling_history.append(scaling_event)
            self.scaling_metrics["total_scaling_events"] += 1
            
            self.logger.info(
                f"Scaling {resource_type.value} {scaling_decision['action'].value}: "
                f"{current_capacity:.2f} -> {new_capacity:.2f} "
                f"(reason: {scaling_decision['reason']})"
            )
        
        return scaling_event

    def _evaluate_scaling_need(
        self,
        config: AutoScalingConfig,
        current_capacity: float,
        current_utilization: float,
        predicted_demand: float
    ) -> Dict[str, Any]:
        """Evaluate if scaling is needed based on current and predicted metrics."""
        
        # Check cooldown periods
        last_scaling_event = self._get_last_scaling_event(config.resource_type)
        if last_scaling_event:
            time_since_last_scaling = time.time() - last_scaling_event.timestamp
            
            if (last_scaling_event.direction == ScalingDirection.UP and 
                time_since_last_scaling < config.scale_up_cooldown):
                return {"action": ScalingDirection.STABLE, "reason": "scale_up_cooldown"}
            
            if (last_scaling_event.direction == ScalingDirection.DOWN and 
                time_since_last_scaling < config.scale_down_cooldown):
                return {"action": ScalingDirection.STABLE, "reason": "scale_down_cooldown"}
        
        # Calculate effective utilization (current + predicted)
        effective_utilization = max(current_utilization, predicted_demand)
        
        # Scaling logic
        if effective_utilization > config.scale_up_threshold:
            if current_capacity < config.max_capacity:
                return {
                    "action": ScalingDirection.UP,
                    "reason": f"utilization {effective_utilization:.2f} > threshold {config.scale_up_threshold:.2f}"
                }
        
        elif effective_utilization < config.scale_down_threshold:
            if current_capacity > config.min_capacity:
                return {
                    "action": ScalingDirection.DOWN,
                    "reason": f"utilization {effective_utilization:.2f} < threshold {config.scale_down_threshold:.2f}"
                }
        
        return {"action": ScalingDirection.STABLE, "reason": "within_thresholds"}

    def _calculate_new_capacity(
        self,
        config: AutoScalingConfig,
        current_capacity: float,
        scaling_decision: Dict[str, Any],
        prediction: PredictionResult
    ) -> float:
        """Calculate optimal new capacity based on prediction and constraints."""
        
        if scaling_decision["action"] == ScalingDirection.UP:
            # Calculate capacity needed for predicted demand
            target_capacity = prediction.predicted_value / config.target_utilization
            
            # Add buffer based on confidence
            confidence_buffer = (1.0 - prediction.confidence_score) * 0.2
            target_capacity *= (1.0 + confidence_buffer)
            
            # Scale incrementally if not aggressive
            if not config.aggressive_scaling:
                increment = (target_capacity - current_capacity) * 0.5  # 50% of needed increase
                new_capacity = current_capacity + increment
            else:
                new_capacity = target_capacity
            
            # Apply constraints
            new_capacity = min(new_capacity, config.max_capacity)
            new_capacity = max(new_capacity, current_capacity * 1.1)  # Minimum 10% increase
        
        elif scaling_decision["action"] == ScalingDirection.DOWN:
            # Calculate capacity needed for predicted demand
            target_capacity = prediction.predicted_value / config.target_utilization
            
            # Add safety buffer
            safety_buffer = 0.1  # 10% safety margin
            target_capacity *= (1.0 + safety_buffer)
            
            # Scale down incrementally
            if not config.aggressive_scaling:
                decrement = (current_capacity - target_capacity) * 0.3  # 30% of possible decrease
                new_capacity = current_capacity - decrement
            else:
                new_capacity = target_capacity
            
            # Apply constraints
            new_capacity = max(new_capacity, config.min_capacity)
            new_capacity = min(new_capacity, current_capacity * 0.9)  # Maximum 10% decrease
        
        else:
            new_capacity = current_capacity
        
        return round(new_capacity, 2)

    async def _execute_scaling(self, scaling_event: ScalingEvent) -> bool:
        """Execute the actual scaling operation."""
        start_time = time.time()
        
        try:
            # Simulate scaling execution
            # In a real implementation, this would interface with container orchestrators,
            # cloud APIs, or other infrastructure management systems
            
            resource_type = scaling_event.resource_type
            from_capacity = scaling_event.from_capacity
            to_capacity = scaling_event.to_capacity
            
            self.logger.info(
                f"Executing scaling for {resource_type.value}: {from_capacity} -> {to_capacity}"
            )
            
            # Simulate scaling time based on resource type and scale
            if resource_type == ResourceType.COMPUTE:
                scaling_time = abs(to_capacity - from_capacity) * 2.0  # 2 seconds per unit
            elif resource_type == ResourceType.MEMORY:
                scaling_time = abs(to_capacity - from_capacity) * 1.0  # 1 second per unit
            else:
                scaling_time = abs(to_capacity - from_capacity) * 3.0  # 3 seconds per unit
            
            # Simulate actual scaling operation
            await asyncio.sleep(min(scaling_time, 10.0))  # Cap at 10 seconds for simulation
            
            execution_time = time.time() - start_time
            scaling_event.execution_time = execution_time
            
            # Record metrics
            self.observability.record_timer(
                "scaling_execution_time", 
                execution_time,
                tags={
                    "resource_type": resource_type.value,
                    "direction": scaling_event.direction.value
                }
            )
            
            self.observability.record_metric(
                "capacity_change",
                to_capacity - from_capacity,
                MetricType.GAUGE,
                tags={"resource_type": resource_type.value}
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Scaling execution failed: {e}")
            return False

    # === DATA COLLECTION ===

    def _get_historical_data(
        self, 
        resource_type: ResourceType, 
        lookback_seconds: int
    ) -> List[Tuple[float, float]]:
        """Get historical resource utilization data."""
        cutoff_time = time.time() - lookback_seconds
        
        # Filter relevant metrics from history
        relevant_metrics = [
            metrics for metrics in self.resource_metrics_history
            if metrics.timestamp >= cutoff_time
        ]
        
        # Extract utilization values based on resource type
        data_points = []
        for metrics in relevant_metrics:
            if resource_type == ResourceType.COMPUTE:
                value = metrics.cpu_utilization
            elif resource_type == ResourceType.MEMORY:
                value = metrics.memory_utilization
            elif resource_type == ResourceType.GPU:
                value = metrics.gpu_utilization
            elif resource_type == ResourceType.NETWORK:
                value = metrics.network_io
            else:
                value = metrics.cpu_utilization  # Default
            
            data_points.append((metrics.timestamp, value))
        
        return data_points

    async def _get_current_resource_metrics(self) -> ResourceMetrics:
        """Get current resource utilization metrics."""
        # In a real implementation, this would collect actual system metrics
        # For simulation, we'll generate realistic metrics
        
        import random
        
        # Simulate realistic metrics with some correlation
        base_cpu = 0.3 + random.random() * 0.4  # 30-70% base usage
        base_memory = 0.2 + random.random() * 0.5  # 20-70% base usage
        
        # Add some time-based patterns
        hour_of_day = (time.time() % 86400) / 3600  # 0-24
        daily_factor = 1.0 + 0.3 * math.sin((hour_of_day - 6) * math.pi / 12)  # Peak at 6 PM
        
        metrics = ResourceMetrics(
            timestamp=time.time(),
            cpu_utilization=min(1.0, base_cpu * daily_factor + random.random() * 0.1),
            memory_utilization=min(1.0, base_memory * daily_factor + random.random() * 0.1),
            network_io=random.random() * 0.8,
            disk_io=random.random() * 0.6,
            gpu_utilization=random.random() * 0.9,
            active_connections=random.randint(10, 100),
            queue_depth=random.randint(0, 20),
            throughput=50.0 + random.random() * 100.0,
            latency=10.0 + random.random() * 50.0
        )
        
        # Store for history
        self.resource_metrics_history.append(metrics)
        
        # Record in observability system
        self.observability.set_gauge("cpu_utilization", metrics.cpu_utilization)
        self.observability.set_gauge("memory_utilization", metrics.memory_utilization)
        self.observability.set_gauge("gpu_utilization", metrics.gpu_utilization)
        self.observability.set_gauge("throughput", metrics.throughput, unit="ops/sec")
        self.observability.set_gauge("latency", metrics.latency, unit="ms")
        
        return metrics

    def _get_utilization_for_resource(
        self, 
        metrics: ResourceMetrics, 
        resource_type: ResourceType
    ) -> float:
        """Extract utilization value for specific resource type."""
        if resource_type == ResourceType.COMPUTE:
            return metrics.cpu_utilization
        elif resource_type == ResourceType.MEMORY:
            return metrics.memory_utilization
        elif resource_type == ResourceType.GPU:
            return metrics.gpu_utilization
        elif resource_type == ResourceType.NETWORK:
            return metrics.network_io
        else:
            return metrics.cpu_utilization

    def _get_last_scaling_event(self, resource_type: ResourceType) -> Optional[ScalingEvent]:
        """Get the last scaling event for a resource type."""
        resource_events = [
            event for event in self.scaling_history
            if event.resource_type == resource_type
        ]
        
        if not resource_events:
            return None
        
        return max(resource_events, key=lambda e: e.timestamp)

    # === BACKGROUND TASKS ===

    async def _resource_monitoring_loop(self):
        """Continuously monitor resource utilization."""
        while not self._shutdown_event.is_set():
            try:
                # Collect current metrics
                current_metrics = await self._get_current_resource_metrics()
                
                # Check for scaling decisions for each configured resource
                for resource_type in self.scaling_configs:
                    scaling_event = await self.make_scaling_decision(resource_type)
                    
                    if scaling_event:
                        self.logger.info(f"Scaling decision made: {scaling_event}")
                
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in resource monitoring: {e}")
                await asyncio.sleep(30)

    async def _prediction_loop(self):
        """Continuously update demand predictions."""
        while not self._shutdown_event.is_set():
            try:
                # Update predictions for each resource type
                for resource_type in self.scaling_configs:
                    prediction = await self.predict_demand(resource_type)
                    
                    # Evaluate prediction accuracy for learning
                    await self._evaluate_prediction_accuracy(resource_type, prediction)
                
                await asyncio.sleep(60)  # Update predictions every minute
                
            except Exception as e:
                self.logger.error(f"Error in prediction loop: {e}")
                await asyncio.sleep(60)

    async def _optimization_loop(self):
        """Continuously optimize scaling parameters."""
        while not self._shutdown_event.is_set():
            try:
                # Perform optimization every 10 minutes
                await self._optimize_scaling_parameters()
                
                # Update model performance metrics
                await self._update_model_performance()
                
                await asyncio.sleep(600)  # Optimize every 10 minutes
                
            except Exception as e:
                self.logger.error(f"Error in optimization loop: {e}")
                await asyncio.sleep(600)

    async def _evaluate_prediction_accuracy(
        self, 
        resource_type: ResourceType, 
        prediction: PredictionResult
    ):
        """Evaluate accuracy of previous predictions for learning."""
        # Check if we have actual values to compare against predictions
        current_time = time.time()
        
        # Find predictions made around the prediction horizon ago
        target_time = current_time - prediction.time_horizon
        
        # Get actual utilization at target time
        actual_metrics = [
            m for m in self.resource_metrics_history
            if abs(m.timestamp - target_time) < 60  # Within 1 minute
        ]
        
        if actual_metrics:
            actual_value = self._get_utilization_for_resource(actual_metrics[0], resource_type)
            prediction_error = abs(prediction.predicted_value - actual_value)
            
            # Store for learning
            self.learning_data[f"{resource_type.value}_prediction_errors"].append(
                (prediction.confidence_score, prediction_error)
            )
            
            # Update performance metrics
            if prediction_error < 0.1:  # Within 10% is considered accurate
                self.scaling_metrics["successful_predictions"] += 1
            
            # Update average accuracy
            total_predictions = self.scaling_metrics["total_predictions"]
            successful_predictions = self.scaling_metrics["successful_predictions"]
            
            if total_predictions > 0:
                self.scaling_metrics["average_prediction_accuracy"] = \
                    successful_predictions / total_predictions

    async def _optimize_scaling_parameters(self):
        """Optimize scaling parameters based on historical performance."""
        if not self.learning_enabled:
            return
        
        # Analyze scaling history for optimization opportunities
        recent_events = [
            event for event in self.scaling_history
            if time.time() - event.timestamp < 3600  # Last hour
        ]
        
        if len(recent_events) < 5:
            return  # Not enough data for optimization
        
        # Analyze patterns and suggest optimizations
        optimization_suggestions = await self._analyze_scaling_patterns(recent_events)
        
        # Apply reasonable optimizations automatically
        for suggestion in optimization_suggestions:
            if suggestion["confidence"] > 0.8 and suggestion["impact"] > 0.1:
                await self._apply_optimization(suggestion)

    async def _analyze_scaling_patterns(
        self, 
        recent_events: List[ScalingEvent]
    ) -> List[Dict[str, Any]]:
        """Analyze scaling patterns to identify optimization opportunities."""
        suggestions = []
        
        # Group events by resource type
        events_by_resource = defaultdict(list)
        for event in recent_events:
            events_by_resource[event.resource_type].append(event)
        
        for resource_type, events in events_by_resource.items():
            # Analyze thrashing (frequent up/down scaling)
            if len(events) >= 4:
                directions = [e.direction for e in events[-4:]]
                if len(set(directions)) > 1:  # Mixed directions
                    suggestions.append({
                        "type": "increase_cooldown",
                        "resource_type": resource_type,
                        "reason": "Frequent direction changes detected",
                        "confidence": 0.9,
                        "impact": 0.3,
                        "parameters": {"cooldown_increase": 60}
                    })
            
            # Analyze over-provisioning
            capacities = [e.to_capacity for e in events]
            if len(capacities) >= 3:
                if all(c > events[0].predicted_demand * 1.5 for c in capacities[-3:]):
                    suggestions.append({
                        "type": "lower_scale_up_threshold",
                        "resource_type": resource_type,
                        "reason": "Consistent over-provisioning detected",
                        "confidence": 0.8,
                        "impact": 0.2,
                        "parameters": {"threshold_decrease": 0.05}
                    })
        
        return suggestions

    async def _apply_optimization(self, suggestion: Dict[str, Any]):
        """Apply an optimization suggestion."""
        resource_type = suggestion["resource_type"]
        
        if resource_type not in self.scaling_configs:
            return
        
        config = self.scaling_configs[resource_type]
        
        if suggestion["type"] == "increase_cooldown":
            cooldown_increase = suggestion["parameters"]["cooldown_increase"]
            config.scale_up_cooldown += cooldown_increase
            config.scale_down_cooldown += cooldown_increase
            
            self.logger.info(
                f"Optimization applied: Increased cooldown for {resource_type.value} "
                f"by {cooldown_increase}s"
            )
        
        elif suggestion["type"] == "lower_scale_up_threshold":
            threshold_decrease = suggestion["parameters"]["threshold_decrease"]
            config.scale_up_threshold = max(0.1, config.scale_up_threshold - threshold_decrease)
            
            self.logger.info(
                f"Optimization applied: Lowered scale-up threshold for {resource_type.value} "
                f"by {threshold_decrease}"
            )

    async def _update_model_performance(self):
        """Update performance metrics for prediction models."""
        # Calculate model-specific performance metrics
        for model_type in PredictionModel:
            error_data = self.learning_data.get(f"{model_type.value}_errors", [])
            
            if len(error_data) >= 10:
                recent_errors = error_data[-10:]
                avg_error = statistics.mean([error for _, error in recent_errors])
                
                self.model_performance[model_type.value].append(avg_error)
                
                # Keep only recent performance data
                if len(self.model_performance[model_type.value]) > 100:
                    self.model_performance[model_type.value] = \
                        self.model_performance[model_type.value][-100:]

    async def _initialize_default_configs(self):
        """Initialize default scaling configurations."""
        default_configs = [
            AutoScalingConfig(
                resource_type=ResourceType.COMPUTE,
                min_capacity=1.0,
                max_capacity=10.0,
                target_utilization=0.7,
                scale_up_threshold=0.8,
                scale_down_threshold=0.3,
                scale_up_cooldown=300,  # 5 minutes
                scale_down_cooldown=600,  # 10 minutes
                prediction_window=600,
                aggressive_scaling=False
            ),
            AutoScalingConfig(
                resource_type=ResourceType.MEMORY,
                min_capacity=1.0,
                max_capacity=8.0,
                target_utilization=0.75,
                scale_up_threshold=0.85,
                scale_down_threshold=0.4,
                scale_up_cooldown=180,  # 3 minutes
                scale_down_cooldown=900,  # 15 minutes
                prediction_window=300,
                aggressive_scaling=False
            )
        ]
        
        for config in default_configs:
            self.add_scaling_config(config)

    async def _initialize_prediction_models(self):
        """Initialize prediction models."""
        # Initialize simple statistical models
        self.prediction_models = {
            "linear_regression": {"initialized": True},
            "exponential_smoothing": {"alpha": 0.3},
            "ensemble": {"weights": [0.4, 0.4, 0.2]}  # LR, ES, LLM
        }
        
        self.logger.info("Prediction models initialized")

    # === PUBLIC API ===

    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling system status."""
        return {
            "configured_resources": list(self.scaling_configs.keys()),
            "current_capacities": self.current_capacity.copy(),
            "recent_scaling_events": len([
                e for e in self.scaling_history 
                if time.time() - e.timestamp < 3600
            ]),
            "performance_metrics": self.scaling_metrics.copy(),
            "prediction_models": list(self.prediction_models.keys()),
            "learning_enabled": self.learning_enabled
        }

    def get_resource_forecast(
        self, 
        resource_type: ResourceType, 
        hours_ahead: int = 2
    ) -> Dict[str, Any]:
        """Get resource utilization forecast."""
        # Generate forecast for multiple time points
        forecast_points = []
        
        for i in range(1, hours_ahead + 1):
            time_horizon = i * 3600  # Convert hours to seconds
            prediction = asyncio.run(self.predict_demand(resource_type, time_horizon))
            
            forecast_points.append({
                "hours_ahead": i,
                "predicted_utilization": prediction.predicted_value,
                "confidence_interval": prediction.confidence_interval,
                "confidence_score": prediction.confidence_score
            })
        
        return {
            "resource_type": resource_type.value,
            "current_capacity": self.current_capacity.get(resource_type, 1.0),
            "forecast": forecast_points,
            "last_updated": time.time()
        }
"""
Real-time Observability System

Advanced observability platform for formal verification with distributed tracing,
real-time metrics, anomaly detection, and predictive analytics.
"""

import asyncio
import json
import time
import uuid
import statistics
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Callable, Union, Tuple
from enum import Enum
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor
import traceback

from .logger import get_logger
from .metrics import MetricsCollector
from ..llm.llm_client import LLMManager


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class TraceSpan:
    """Represents a span in distributed tracing."""
    span_id: str
    trace_id: str
    parent_span_id: Optional[str]
    operation_name: str
    start_time: float
    end_time: Optional[float]
    duration: Optional[float]
    tags: Dict[str, Any]
    logs: List[Dict[str, Any]]
    status: str = "active"
    service_name: str = "formal_circuits_gpt"


@dataclass
class Metric:
    """Represents a system metric."""
    name: str
    metric_type: MetricType
    value: Union[float, int]
    timestamp: float
    tags: Dict[str, str]
    unit: str = ""


@dataclass
class Alert:
    """Represents a system alert."""
    alert_id: str
    name: str
    severity: AlertSeverity
    message: str
    timestamp: float
    source: str
    metric_name: Optional[str]
    threshold_value: Optional[float]
    current_value: Optional[float]
    tags: Dict[str, str]
    resolution_time: Optional[float] = None


@dataclass
class AnomalyDetection:
    """Anomaly detection result."""
    metric_name: str
    timestamp: float
    value: float
    expected_range: Tuple[float, float]
    anomaly_score: float
    confidence: float
    pattern_type: str


class RealTimeObservability:
    """
    Advanced real-time observability system for formal verification
    with distributed tracing, metrics collection, and intelligent alerting.
    """

    def __init__(
        self,
        service_name: str = "formal_circuits_gpt",
        sampling_rate: float = 1.0,
        retention_hours: int = 24
    ):
        self.service_name = service_name
        self.sampling_rate = sampling_rate
        self.retention_hours = retention_hours
        
        self.logger = get_logger("realtime_observability")
        self.llm_manager = LLMManager.create_default()
        
        # Tracing system
        self.active_traces: Dict[str, List[TraceSpan]] = {}
        self.completed_traces: Dict[str, List[TraceSpan]] = {}
        self.span_index: Dict[str, TraceSpan] = {}
        
        # Metrics system
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.metric_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Alerting system
        self.alerts: Dict[str, Alert] = {}
        self.alert_rules: List[Dict[str, Any]] = []
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        
        # Anomaly detection
        self.anomaly_detectors: Dict[str, 'AnomalyDetector'] = {}
        self.baseline_data: Dict[str, List[float]] = defaultdict(list)
        
        # Real-time dashboards
        self.dashboard_subscribers: List[Callable[[Dict[str, Any]], None]] = []
        
        # Performance optimization
        self.async_processing = True
        self.batch_size = 100
        self.flush_interval = 5.0
        
        # Background tasks
        self._background_tasks = []
        self._shutdown_event = asyncio.Event()
        
        self.logger.info(f"Real-time observability system initialized for {service_name}")

    async def start(self):
        """Start the observability system background tasks."""
        self.logger.info("Starting observability system")
        
        # Start background processing tasks
        self._background_tasks = [
            asyncio.create_task(self._metrics_processor()),
            asyncio.create_task(self._anomaly_detector_runner()),
            asyncio.create_task(self._alert_processor()),
            asyncio.create_task(self._trace_processor()),
            asyncio.create_task(self._cleanup_old_data()),
            asyncio.create_task(self._dashboard_updater())
        ]
        
        # Initialize anomaly detectors for common metrics
        await self._initialize_anomaly_detectors()
        
        # Setup default alert rules
        self._setup_default_alert_rules()

    async def stop(self):
        """Stop the observability system."""
        self.logger.info("Stopping observability system")
        
        self._shutdown_event.set()
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self._background_tasks, return_exceptions=True)

    # === DISTRIBUTED TRACING ===

    def start_trace(self, operation_name: str, parent_span_id: Optional[str] = None) -> str:
        """Start a new distributed trace."""
        trace_id = str(uuid.uuid4())
        span_id = self.start_span(trace_id, operation_name, parent_span_id)
        return span_id

    def start_span(
        self, 
        trace_id: str, 
        operation_name: str, 
        parent_span_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None
    ) -> str:
        """Start a new span within a trace."""
        span_id = str(uuid.uuid4())
        
        span = TraceSpan(
            span_id=span_id,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            start_time=time.time(),
            end_time=None,
            duration=None,
            tags=tags or {},
            logs=[]
        )
        
        # Store span
        self.span_index[span_id] = span
        
        if trace_id not in self.active_traces:
            self.active_traces[trace_id] = []
        self.active_traces[trace_id].append(span)
        
        self.logger.debug(f"Started span {span_id} for operation {operation_name}")
        return span_id

    def add_span_tag(self, span_id: str, key: str, value: Any):
        """Add a tag to an active span."""
        if span_id in self.span_index:
            self.span_index[span_id].tags[key] = value

    def add_span_log(self, span_id: str, message: str, level: str = "info", **kwargs):
        """Add a log entry to an active span."""
        if span_id in self.span_index:
            log_entry = {
                "timestamp": time.time(),
                "level": level,
                "message": message,
                **kwargs
            }
            self.span_index[span_id].logs.append(log_entry)

    def finish_span(self, span_id: str, status: str = "success"):
        """Finish an active span."""
        if span_id in self.span_index:
            span = self.span_index[span_id]
            span.end_time = time.time()
            span.duration = span.end_time - span.start_time
            span.status = status
            
            self.logger.debug(f"Finished span {span_id} with status {status}, duration: {span.duration:.3f}s")
            
            # Move to completed traces if this was the root span
            if span.parent_span_id is None and span.trace_id in self.active_traces:
                self.completed_traces[span.trace_id] = self.active_traces.pop(span.trace_id)

    def get_trace(self, trace_id: str) -> Optional[List[TraceSpan]]:
        """Get a complete trace by ID."""
        return self.completed_traces.get(trace_id) or self.active_traces.get(trace_id)

    # === METRICS COLLECTION ===

    def record_metric(
        self, 
        name: str, 
        value: Union[float, int], 
        metric_type: MetricType = MetricType.GAUGE,
        tags: Optional[Dict[str, str]] = None,
        unit: str = ""
    ):
        """Record a metric value."""
        metric = Metric(
            name=name,
            metric_type=metric_type,
            value=value,
            timestamp=time.time(),
            tags=tags or {},
            unit=unit
        )
        
        self.metrics[name].append(metric)
        
        # Store metadata
        if name not in self.metric_metadata:
            self.metric_metadata[name] = {
                "type": metric_type.value,
                "unit": unit,
                "tags": set()
            }
        
        # Update tag set
        self.metric_metadata[name]["tags"].update(tags.keys() if tags else [])

    def increment_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        self.record_metric(name, value, MetricType.COUNTER, tags)

    def set_gauge(self, name: str, value: Union[float, int], tags: Optional[Dict[str, str]] = None, unit: str = ""):
        """Set a gauge metric value."""
        self.record_metric(name, value, MetricType.GAUGE, tags, unit)

    def record_timer(self, name: str, duration: float, tags: Optional[Dict[str, str]] = None):
        """Record a timer metric."""
        self.record_metric(name, duration, MetricType.TIMER, tags, "seconds")

    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None, unit: str = ""):
        """Record a histogram metric."""
        self.record_metric(name, value, MetricType.HISTOGRAM, tags, unit)

    def get_metric_stats(self, name: str, time_window: int = 300) -> Dict[str, float]:
        """Get statistical summary of a metric over time window."""
        if name not in self.metrics:
            return {}
        
        cutoff_time = time.time() - time_window
        recent_values = [
            m.value for m in self.metrics[name]
            if m.timestamp >= cutoff_time
        ]
        
        if not recent_values:
            return {}
        
        return {
            "count": len(recent_values),
            "min": min(recent_values),
            "max": max(recent_values),
            "mean": statistics.mean(recent_values),
            "median": statistics.median(recent_values),
            "stddev": statistics.stdev(recent_values) if len(recent_values) > 1 else 0.0
        }

    # === ALERTING SYSTEM ===

    def add_alert_rule(
        self,
        name: str,
        metric_name: str,
        condition: str,  # "gt", "lt", "eq", "ne"
        threshold: float,
        severity: AlertSeverity = AlertSeverity.WARNING,
        evaluation_window: int = 300,
        min_duration: int = 60
    ):
        """Add an alert rule."""
        rule = {
            "name": name,
            "metric_name": metric_name,
            "condition": condition,
            "threshold": threshold,
            "severity": severity,
            "evaluation_window": evaluation_window,
            "min_duration": min_duration,
            "last_evaluation": 0,
            "triggered_at": None
        }
        
        self.alert_rules.append(rule)
        self.logger.info(f"Added alert rule: {name}")

    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add a callback function for alert notifications."""
        self.alert_callbacks.append(callback)

    def trigger_alert(
        self,
        name: str,
        severity: AlertSeverity,
        message: str,
        source: str,
        metric_name: Optional[str] = None,
        current_value: Optional[float] = None,
        threshold_value: Optional[float] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """Manually trigger an alert."""
        alert_id = str(uuid.uuid4())
        
        alert = Alert(
            alert_id=alert_id,
            name=name,
            severity=severity,
            message=message,
            timestamp=time.time(),
            source=source,
            metric_name=metric_name,
            threshold_value=threshold_value,
            current_value=current_value,
            tags=tags or {}
        )
        
        self.alerts[alert_id] = alert
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")
        
        self.logger.warning(f"Alert triggered: {name} - {message}")
        return alert_id

    def resolve_alert(self, alert_id: str):
        """Resolve an active alert."""
        if alert_id in self.alerts:
            self.alerts[alert_id].resolution_time = time.time()
            self.logger.info(f"Alert {alert_id} resolved")

    # === ANOMALY DETECTION ===

    async def detect_anomalies(self, metric_name: str) -> List[AnomalyDetection]:
        """Detect anomalies in a metric using ML techniques."""
        if metric_name not in self.anomaly_detectors:
            return []
        
        detector = self.anomaly_detectors[metric_name]
        return await detector.detect(self.metrics[metric_name])

    async def _initialize_anomaly_detectors(self):
        """Initialize anomaly detectors for key metrics."""
        key_metrics = [
            "verification_duration",
            "llm_response_time", 
            "memory_usage",
            "cpu_usage",
            "error_rate",
            "success_rate"
        ]
        
        for metric_name in key_metrics:
            detector = AnomalyDetector(metric_name, self.llm_manager)
            self.anomaly_detectors[metric_name] = detector
            await detector.initialize()

    # === DASHBOARD INTEGRATION ===

    def add_dashboard_subscriber(self, callback: Callable[[Dict[str, Any]], None]):
        """Add a dashboard update callback."""
        self.dashboard_subscribers.append(callback)

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data."""
        current_time = time.time()
        
        # System overview
        overview = {
            "active_traces": len(self.active_traces),
            "completed_traces": len(self.completed_traces),
            "total_metrics": sum(len(metrics) for metrics in self.metrics.values()),
            "active_alerts": len([a for a in self.alerts.values() if a.resolution_time is None]),
            "timestamp": current_time
        }
        
        # Key metrics summary
        key_metrics = {}
        for metric_name in ["verification_duration", "llm_response_time", "success_rate", "error_rate"]:
            stats = self.get_metric_stats(metric_name, 300)  # Last 5 minutes
            if stats:
                key_metrics[metric_name] = stats
        
        # Recent alerts
        recent_alerts = [
            asdict(alert) for alert in self.alerts.values()
            if current_time - alert.timestamp < 3600  # Last hour
        ]
        
        # Anomalies
        recent_anomalies = []
        for detector in self.anomaly_detectors.values():
            recent_anomalies.extend(detector.get_recent_anomalies(3600))
        
        return {
            "overview": overview,
            "key_metrics": key_metrics,
            "recent_alerts": recent_alerts,
            "anomalies": recent_anomalies,
            "trace_samples": self._get_trace_samples()
        }

    def _get_trace_samples(self) -> List[Dict[str, Any]]:
        """Get sample traces for dashboard display."""
        samples = []
        
        # Get recent completed traces
        recent_traces = sorted(
            self.completed_traces.items(),
            key=lambda x: max(span.start_time for span in x[1]),
            reverse=True
        )[:10]
        
        for trace_id, spans in recent_traces:
            root_span = next((s for s in spans if s.parent_span_id is None), spans[0])
            
            samples.append({
                "trace_id": trace_id,
                "operation": root_span.operation_name,
                "duration": root_span.duration,
                "span_count": len(spans),
                "status": root_span.status,
                "start_time": root_span.start_time
            })
        
        return samples

    # === BACKGROUND PROCESSING ===

    async def _metrics_processor(self):
        """Background task for processing metrics."""
        while not self._shutdown_event.is_set():
            try:
                # Process metrics in batches
                await self._process_metric_batch()
                await asyncio.sleep(self.flush_interval)
            except Exception as e:
                self.logger.error(f"Error in metrics processor: {e}")
                await asyncio.sleep(5)

    async def _process_metric_batch(self):
        """Process a batch of metrics for analysis."""
        # Update baseline data for anomaly detection
        for metric_name, metric_deque in self.metrics.items():
            if len(metric_deque) > 0:
                recent_values = [m.value for m in list(metric_deque)[-100:]]
                self.baseline_data[metric_name].extend(recent_values)
                
                # Keep only recent baseline data
                if len(self.baseline_data[metric_name]) > 1000:
                    self.baseline_data[metric_name] = self.baseline_data[metric_name][-1000:]

    async def _anomaly_detector_runner(self):
        """Background task for running anomaly detection."""
        while not self._shutdown_event.is_set():
            try:
                # Run anomaly detection on all metrics
                for metric_name in list(self.anomaly_detectors.keys()):
                    if metric_name in self.metrics and len(self.metrics[metric_name]) > 0:
                        anomalies = await self.detect_anomalies(metric_name)
                        
                        # Create alerts for significant anomalies
                        for anomaly in anomalies:
                            if anomaly.confidence > 0.8 and anomaly.anomaly_score > 2.0:
                                self.trigger_alert(
                                    name=f"Anomaly Detected: {metric_name}",
                                    severity=AlertSeverity.WARNING,
                                    message=f"Anomalous value {anomaly.value:.2f} detected for {metric_name} (confidence: {anomaly.confidence:.2f})",
                                    source="anomaly_detector",
                                    metric_name=metric_name,
                                    current_value=anomaly.value
                                )
                
                await asyncio.sleep(30)  # Run every 30 seconds
            except Exception as e:
                self.logger.error(f"Error in anomaly detector: {e}")
                await asyncio.sleep(30)

    async def _alert_processor(self):
        """Background task for processing alert rules."""
        while not self._shutdown_event.is_set():
            try:
                current_time = time.time()
                
                for rule in self.alert_rules:
                    # Check if it's time to evaluate this rule
                    if current_time - rule["last_evaluation"] >= 30:  # Evaluate every 30 seconds
                        await self._evaluate_alert_rule(rule, current_time)
                        rule["last_evaluation"] = current_time
                
                await asyncio.sleep(10)
            except Exception as e:
                self.logger.error(f"Error in alert processor: {e}")
                await asyncio.sleep(10)

    async def _evaluate_alert_rule(self, rule: Dict[str, Any], current_time: float):
        """Evaluate a single alert rule."""
        metric_name = rule["metric_name"]
        
        if metric_name not in self.metrics or len(self.metrics[metric_name]) == 0:
            return
        
        # Get recent values
        window_start = current_time - rule["evaluation_window"]
        recent_values = [
            m.value for m in self.metrics[metric_name]
            if m.timestamp >= window_start
        ]
        
        if not recent_values:
            return
        
        # Calculate current value (average of recent values)
        current_value = statistics.mean(recent_values)
        
        # Evaluate condition
        condition = rule["condition"]
        threshold = rule["threshold"]
        
        condition_met = False
        if condition == "gt":
            condition_met = current_value > threshold
        elif condition == "lt":
            condition_met = current_value < threshold
        elif condition == "eq":
            condition_met = abs(current_value - threshold) < 0.001
        elif condition == "ne":
            condition_met = abs(current_value - threshold) >= 0.001
        
        if condition_met:
            # Check if we need to trigger alert (respecting min_duration)
            if rule["triggered_at"] is None:
                rule["triggered_at"] = current_time
            elif current_time - rule["triggered_at"] >= rule["min_duration"]:
                # Trigger alert
                self.trigger_alert(
                    name=rule["name"],
                    severity=rule["severity"],
                    message=f"Metric {metric_name} {condition} {threshold} (current: {current_value:.2f})",
                    source="alert_rule",
                    metric_name=metric_name,
                    current_value=current_value,
                    threshold_value=threshold
                )
                rule["triggered_at"] = None  # Reset for next trigger
        else:
            # Condition not met, reset trigger
            rule["triggered_at"] = None

    async def _trace_processor(self):
        """Background task for processing traces."""
        while not self._shutdown_event.is_set():
            try:
                # Clean up old active traces
                current_time = time.time()
                timeout_threshold = 3600  # 1 hour
                
                for trace_id, spans in list(self.active_traces.items()):
                    oldest_span_time = min(span.start_time for span in spans)
                    if current_time - oldest_span_time > timeout_threshold:
                        # Move to completed traces (timed out)
                        self.completed_traces[trace_id] = self.active_traces.pop(trace_id)
                        self.logger.warning(f"Trace {trace_id} timed out and moved to completed")
                
                await asyncio.sleep(60)
            except Exception as e:
                self.logger.error(f"Error in trace processor: {e}")
                await asyncio.sleep(60)

    async def _cleanup_old_data(self):
        """Background task for cleaning up old data."""
        while not self._shutdown_event.is_set():
            try:
                current_time = time.time()
                retention_cutoff = current_time - (self.retention_hours * 3600)
                
                # Clean up old metrics
                for metric_name, metric_deque in self.metrics.items():
                    # Convert to list, filter, and convert back
                    filtered_metrics = deque(
                        [m for m in metric_deque if m.timestamp >= retention_cutoff],
                        maxlen=metric_deque.maxlen
                    )
                    self.metrics[metric_name] = filtered_metrics
                
                # Clean up old traces
                old_trace_ids = [
                    trace_id for trace_id, spans in self.completed_traces.items()
                    if max(span.start_time for span in spans) < retention_cutoff
                ]
                
                for trace_id in old_trace_ids:
                    del self.completed_traces[trace_id]
                
                # Clean up old alerts
                old_alert_ids = [
                    alert_id for alert_id, alert in self.alerts.items()
                    if alert.timestamp < retention_cutoff
                ]
                
                for alert_id in old_alert_ids:
                    del self.alerts[alert_id]
                
                if old_trace_ids or old_alert_ids:
                    self.logger.info(f"Cleaned up {len(old_trace_ids)} old traces and {len(old_alert_ids)} old alerts")
                
                await asyncio.sleep(3600)  # Run every hour
            except Exception as e:
                self.logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(3600)

    async def _dashboard_updater(self):
        """Background task for updating dashboards."""
        while not self._shutdown_event.is_set():
            try:
                # Get current dashboard data
                dashboard_data = self.get_dashboard_data()
                
                # Send to all subscribers
                for callback in self.dashboard_subscribers:
                    try:
                        callback(dashboard_data)
                    except Exception as e:
                        self.logger.error(f"Error in dashboard callback: {e}")
                
                await asyncio.sleep(5)  # Update every 5 seconds
            except Exception as e:
                self.logger.error(f"Error in dashboard updater: {e}")
                await asyncio.sleep(5)

    def _setup_default_alert_rules(self):
        """Setup default alert rules for common issues."""
        default_rules = [
            {
                "name": "High Verification Duration",
                "metric_name": "verification_duration",
                "condition": "gt",
                "threshold": 300.0,  # 5 minutes
                "severity": AlertSeverity.WARNING,
                "evaluation_window": 300,
                "min_duration": 60
            },
            {
                "name": "High Error Rate",
                "metric_name": "error_rate",
                "condition": "gt", 
                "threshold": 0.1,  # 10%
                "severity": AlertSeverity.ERROR,
                "evaluation_window": 300,
                "min_duration": 30
            },
            {
                "name": "Low Success Rate",
                "metric_name": "success_rate",
                "condition": "lt",
                "threshold": 0.5,  # 50%
                "severity": AlertSeverity.WARNING,
                "evaluation_window": 600,
                "min_duration": 120
            },
            {
                "name": "High Memory Usage",
                "metric_name": "memory_usage",
                "condition": "gt",
                "threshold": 0.9,  # 90%
                "severity": AlertSeverity.CRITICAL,
                "evaluation_window": 120,
                "min_duration": 30
            }
        ]
        
        for rule_config in default_rules:
            self.add_alert_rule(**rule_config)


class AnomalyDetector:
    """ML-based anomaly detector for metrics."""
    
    def __init__(self, metric_name: str, llm_manager: LLMManager):
        self.metric_name = metric_name
        self.llm_manager = llm_manager
        self.logger = get_logger(f"anomaly_detector_{metric_name}")
        
        # Simple statistical model (in production, use more sophisticated ML)
        self.baseline_mean = 0.0
        self.baseline_std = 1.0
        self.baseline_samples = 0
        self.anomalies = deque(maxlen=1000)
        
        # Seasonal patterns detection
        self.seasonal_patterns = {}
        self.pattern_detection_enabled = True
    
    async def initialize(self):
        """Initialize the anomaly detector."""
        self.logger.info(f"Initialized anomaly detector for {self.metric_name}")
    
    async def detect(self, metric_data: deque) -> List[AnomalyDetection]:
        """Detect anomalies in metric data."""
        if len(metric_data) < 10:
            return []
        
        anomalies = []
        recent_data = list(metric_data)[-100:]  # Last 100 points
        
        # Update baseline statistics
        await self._update_baseline(recent_data)
        
        # Detect point anomalies
        for metric in recent_data[-10:]:  # Check last 10 points
            anomaly_score = self._calculate_anomaly_score(metric.value)
            
            if anomaly_score > 2.0:  # Threshold for anomaly
                confidence = min(0.99, anomaly_score / 5.0)
                expected_range = (
                    self.baseline_mean - 2 * self.baseline_std,
                    self.baseline_mean + 2 * self.baseline_std
                )
                
                anomaly = AnomalyDetection(
                    metric_name=self.metric_name,
                    timestamp=metric.timestamp,
                    value=metric.value,
                    expected_range=expected_range,
                    anomaly_score=anomaly_score,
                    confidence=confidence,
                    pattern_type="point_anomaly"
                )
                
                anomalies.append(anomaly)
                self.anomalies.append(anomaly)
        
        # Detect pattern anomalies using LLM
        if self.pattern_detection_enabled and len(recent_data) >= 50:
            pattern_anomalies = await self._detect_pattern_anomalies(recent_data)
            anomalies.extend(pattern_anomalies)
        
        return anomalies
    
    async def _update_baseline(self, data_points: List[Metric]):
        """Update baseline statistics."""
        values = [m.value for m in data_points]
        
        if len(values) < 5:
            return
        
        # Exponential moving average for baseline
        alpha = 0.1  # Smoothing factor
        
        current_mean = statistics.mean(values)
        current_std = statistics.stdev(values) if len(values) > 1 else 1.0
        
        if self.baseline_samples == 0:
            self.baseline_mean = current_mean
            self.baseline_std = current_std
        else:
            self.baseline_mean = alpha * current_mean + (1 - alpha) * self.baseline_mean
            self.baseline_std = alpha * current_std + (1 - alpha) * self.baseline_std
        
        self.baseline_samples += len(values)
    
    def _calculate_anomaly_score(self, value: float) -> float:
        """Calculate anomaly score for a value."""
        if self.baseline_std == 0:
            return 0.0
        
        # Z-score based anomaly score
        z_score = abs(value - self.baseline_mean) / self.baseline_std
        return z_score
    
    async def _detect_pattern_anomalies(self, data_points: List[Metric]) -> List[AnomalyDetection]:
        """Use LLM to detect complex pattern anomalies."""
        try:
            # Prepare data for LLM analysis
            values = [m.value for m in data_points[-50:]]
            timestamps = [m.timestamp for m in data_points[-50:]]
            
            # Create time series description
            time_series_desc = self._create_time_series_description(values, timestamps)
            
            prompt = f"""
            Analyze this time series data for {self.metric_name} and identify any anomalous patterns:
            
            {time_series_desc}
            
            Look for:
            1. Sudden trend changes
            2. Unusual cyclical patterns
            3. Sustained deviations from normal behavior
            4. Pattern breaks or discontinuities
            
            Return JSON with anomalies found, including:
            - timestamp (array index)
            - pattern_type
            - severity (0.0-1.0)
            - description
            
            Only return significant anomalies with severity > 0.7.
            """
            
            response = await self.llm_manager.generate(
                prompt, temperature=0.1, max_tokens=500
            )
            
            # Parse LLM response (simplified)
            anomalies = self._parse_llm_anomaly_response(response.content, data_points)
            return anomalies
            
        except Exception as e:
            self.logger.warning(f"LLM pattern detection failed: {e}")
            return []
    
    def _create_time_series_description(self, values: List[float], timestamps: List[float]) -> str:
        """Create a textual description of time series data."""
        if len(values) < 5:
            return "Insufficient data for analysis"
        
        # Basic statistics
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values) if len(values) > 1 else 0
        min_val = min(values)
        max_val = max(values)
        
        # Trend analysis
        trend = "stable"
        if len(values) >= 10:
            first_half = values[:len(values)//2]
            second_half = values[len(values)//2:]
            first_mean = statistics.mean(first_half)
            second_mean = statistics.mean(second_half)
            
            if second_mean > first_mean * 1.1:
                trend = "increasing"
            elif second_mean < first_mean * 0.9:
                trend = "decreasing"
        
        description = f"""
        Time Series Data for {self.metric_name}:
        - Data points: {len(values)}
        - Mean: {mean_val:.2f}
        - Std Dev: {std_val:.2f}
        - Range: {min_val:.2f} to {max_val:.2f}
        - Trend: {trend}
        - Recent values: {values[-10:]}
        """
        
        return description
    
    def _parse_llm_anomaly_response(
        self, 
        response: str, 
        data_points: List[Metric]
    ) -> List[AnomalyDetection]:
        """Parse LLM response into anomaly objects."""
        anomalies = []
        
        try:
            # Simple parsing for JSON response
            # In production, use more robust JSON parsing
            import json
            
            # Extract JSON from response
            start_idx = response.find('[')
            end_idx = response.rfind(']') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                anomaly_data = json.loads(json_str)
                
                for item in anomaly_data:
                    if item.get("severity", 0) > 0.7:
                        timestamp_idx = item.get("timestamp", 0)
                        if 0 <= timestamp_idx < len(data_points):
                            metric = data_points[timestamp_idx]
                            
                            anomaly = AnomalyDetection(
                                metric_name=self.metric_name,
                                timestamp=metric.timestamp,
                                value=metric.value,
                                expected_range=(self.baseline_mean - 2 * self.baseline_std,
                                              self.baseline_mean + 2 * self.baseline_std),
                                anomaly_score=item.get("severity", 0.8) * 5.0,
                                confidence=item.get("severity", 0.8),
                                pattern_type=item.get("pattern_type", "pattern_anomaly")
                            )
                            
                            anomalies.append(anomaly)
        
        except Exception as e:
            self.logger.warning(f"Failed to parse LLM anomaly response: {e}")
        
        return anomalies
    
    def get_recent_anomalies(self, time_window: int) -> List[AnomalyDetection]:
        """Get anomalies within time window."""
        cutoff_time = time.time() - time_window
        return [a for a in self.anomalies if a.timestamp >= cutoff_time]


# Context manager for automatic span management
class TraceSpanContext:
    """Context manager for automatic span lifecycle management."""
    
    def __init__(
        self, 
        observability: RealTimeObservability,
        operation_name: str,
        trace_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None
    ):
        self.observability = observability
        self.operation_name = operation_name
        self.trace_id = trace_id
        self.parent_span_id = parent_span_id
        self.tags = tags or {}
        self.span_id = None
    
    def __enter__(self):
        if self.trace_id:
            self.span_id = self.observability.start_span(
                self.trace_id, self.operation_name, self.parent_span_id, self.tags
            )
        else:
            self.span_id = self.observability.start_trace(self.operation_name, self.parent_span_id)
        
        return self.span_id
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.span_id:
            status = "error" if exc_type else "success"
            if exc_type:
                self.observability.add_span_log(
                    self.span_id, 
                    f"Exception: {exc_val}", 
                    "error",
                    exception_type=str(exc_type.__name__) if exc_type else None
                )
            self.observability.finish_span(self.span_id, status)
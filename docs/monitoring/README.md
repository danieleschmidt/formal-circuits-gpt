# Monitoring and Observability

## Overview

Formal-Circuits-GPT provides comprehensive monitoring and observability features to help you track verification performance, system health, and usage patterns.

## Monitoring Components

### 1. Health Checks
- **Application Health**: Basic health status and readiness checks
- **Component Health**: Individual component status (parsers, LLM, provers)
- **Dependency Health**: External service availability (LLM APIs, theorem provers)

### 2. Metrics Collection
- **Performance Metrics**: Verification times, success rates, resource usage
- **Business Metrics**: Verification counts, circuit complexity, property coverage
- **System Metrics**: Memory usage, CPU utilization, API request counts

### 3. Structured Logging
- **Request Tracing**: Complete verification request lifecycle
- **Error Tracking**: Detailed error information with context
- **Audit Logging**: Security and compliance logging

### 4. Alerting
- **Performance Degradation**: Slow verification times or high error rates
- **System Issues**: Resource exhaustion or service unavailability
- **Security Events**: Authentication failures or unusual access patterns

## Quick Start

### Enable Monitoring
```python
from formal_circuits_gpt import CircuitVerifier
from formal_circuits_gpt.monitoring import MetricsCollector, HealthChecker

# Initialize with monitoring
verifier = CircuitVerifier(
    enable_metrics=True,
    enable_health_checks=True,
    metrics_port=9090
)

# Verify with automatic metrics collection
result = verifier.verify("design.v")
```

### Health Check Endpoint
```bash
# Check application health
curl http://localhost:8080/health

# Detailed health information
curl http://localhost:8080/health/detailed

# Readiness check
curl http://localhost:8080/ready
```

### Metrics Endpoint
```bash
# Prometheus metrics
curl http://localhost:9090/metrics

# JSON metrics
curl http://localhost:8080/metrics
```

## Configuration

### Environment Variables
```bash
# Enable monitoring features
FORMAL_CIRCUITS_METRICS_ENABLED=true
FORMAL_CIRCUITS_HEALTH_CHECKS_ENABLED=true
FORMAL_CIRCUITS_TRACING_ENABLED=true

# Monitoring endpoints
FORMAL_CIRCUITS_METRICS_PORT=9090
FORMAL_CIRCUITS_HEALTH_PORT=8080

# External monitoring
PROMETHEUS_GATEWAY=http://localhost:9091
JAEGER_ENDPOINT=http://localhost:14268/api/traces
SENTRY_DSN=https://your-sentry-dsn
```

### Configuration File
```yaml
# config.yaml
monitoring:
  enabled: true
  
  metrics:
    enabled: true
    port: 9090
    interval: 30
    retention: "7d"
    
  health:
    enabled: true
    port: 8080
    timeout: 10
    
  logging:
    level: INFO
    format: json
    file: /var/log/formal-circuits-gpt.log
    
  tracing:
    enabled: true
    service_name: "formal-circuits-gpt"
    jaeger_endpoint: "http://localhost:14268/api/traces"
    
  alerting:
    enabled: true
    channels:
      - type: webhook
        url: "https://hooks.slack.com/services/..."
      - type: email
        smtp_server: "smtp.example.com"
        recipients: ["admin@example.com"]
```

## Health Checks

### Health Check Types

#### Basic Health
```json
{
  "status": "healthy",
  "timestamp": "2025-08-02T16:45:00Z",
  "version": "0.1.0",
  "uptime": 3600
}
```

#### Detailed Health
```json
{
  "status": "healthy",
  "timestamp": "2025-08-02T16:45:00Z",
  "components": {
    "parser": {
      "status": "healthy",
      "details": "Verilog and VHDL parsers operational"
    },
    "llm_provider": {
      "status": "healthy", 
      "details": "OpenAI API accessible",
      "response_time": 250
    },
    "theorem_provers": {
      "isabelle": {
        "status": "healthy",
        "version": "Isabelle2024",
        "path": "/usr/local/bin/isabelle"
      },
      "coq": {
        "status": "healthy",
        "version": "8.17.1",
        "path": "/usr/bin/coq"
      }
    },
    "cache": {
      "status": "healthy",
      "size": "245MB",
      "hit_rate": 0.78
    }
  }
}
```

#### Readiness Check
```json
{
  "ready": true,
  "timestamp": "2025-08-02T16:45:00Z",
  "dependencies": {
    "llm_api": "ready",
    "theorem_provers": "ready",
    "cache": "ready"
  }
}
```

### Custom Health Checks
```python
from formal_circuits_gpt.monitoring import HealthChecker

class CustomHealthCheck:
    def check(self):
        # Custom health check logic
        return {
            "status": "healthy",
            "details": "Custom component operational"
        }

# Register custom health check
health_checker = HealthChecker()
health_checker.register("custom_component", CustomHealthCheck())
```

## Metrics

### Available Metrics

#### Performance Metrics
- `verification_duration_seconds`: Time taken for verification
- `verification_success_rate`: Percentage of successful verifications
- `llm_request_duration_seconds`: LLM API request latency
- `theorem_prover_duration_seconds`: Theorem prover execution time

#### System Metrics
- `memory_usage_bytes`: Current memory usage
- `cpu_usage_percent`: CPU utilization
- `cache_hit_rate`: Cache effectiveness
- `active_verifications`: Currently running verifications

#### Business Metrics
- `verifications_total`: Total number of verifications
- `circuits_processed_total`: Total circuits processed
- `properties_verified_total`: Total properties verified
- `llm_tokens_used_total`: Total LLM tokens consumed

### Prometheus Integration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'formal-circuits-gpt'
    static_configs:
      - targets: ['localhost:9090']
    scrape_interval: 30s
    metrics_path: /metrics
```

### Grafana Dashboard
```json
{
  "dashboard": {
    "title": "Formal-Circuits-GPT Monitoring",
    "panels": [
      {
        "title": "Verification Success Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(verification_success_total[5m]) / rate(verifications_total[5m])"
          }
        ]
      },
      {
        "title": "Average Verification Time",
        "type": "graph",
        "targets": [
          {
            "expr": "avg(verification_duration_seconds)"
          }
        ]
      }
    ]
  }
}
```

## Logging

### Log Configuration
```python
import logging
from formal_circuits_gpt.monitoring import configure_logging

# Configure structured logging
configure_logging(
    level=logging.INFO,
    format="json",
    output="/var/log/formal-circuits-gpt.log"
)
```

### Log Formats

#### Structured JSON Logging
```json
{
  "timestamp": "2025-08-02T16:45:00Z",
  "level": "INFO",
  "logger": "formal_circuits_gpt.verifier",
  "message": "Verification completed successfully",
  "verification_id": "ver_123456789",
  "circuit_name": "adder.v",
  "duration": 45.2,
  "prover": "isabelle",
  "properties_verified": 3
}
```

#### Request Tracing
```json
{
  "timestamp": "2025-08-02T16:45:00Z",
  "level": "INFO",
  "trace_id": "abc123",
  "span_id": "def456",
  "operation": "verify_circuit",
  "phase": "parsing",
  "message": "Parsing Verilog file",
  "file_size": 2048,
  "parse_time": 1.2
}
```

### Log Aggregation

#### ELK Stack Integration
```yaml
# logstash.conf
input {
  file {
    path => "/var/log/formal-circuits-gpt.log"
    codec => json
  }
}

filter {
  if [logger] == "formal_circuits_gpt" {
    mutate {
      add_tag => ["formal-circuits"]
    }
  }
}

output {
  elasticsearch {
    hosts => ["localhost:9200"]
    index => "formal-circuits-gpt-%{+YYYY.MM.dd}"
  }
}
```

#### Fluentd Configuration
```xml
<source>
  @type tail
  path /var/log/formal-circuits-gpt.log
  pos_file /var/log/fluentd/formal-circuits-gpt.log.pos
  tag formal-circuits-gpt
  format json
</source>

<match formal-circuits-gpt>
  @type elasticsearch
  host localhost
  port 9200
  index_name formal-circuits-gpt
  type_name verification
</match>
```

## Tracing

### Distributed Tracing Setup
```python
from formal_circuits_gpt.monitoring import TracingConfig

# Enable tracing
tracing_config = TracingConfig(
    service_name="formal-circuits-gpt",
    jaeger_endpoint="http://localhost:14268/api/traces",
    sample_rate=0.1
)

verifier = CircuitVerifier(tracing_config=tracing_config)
```

### Trace Spans
- **Verification Request**: Complete verification lifecycle
- **Parsing**: HDL parsing and AST generation
- **Translation**: AST to formal specification
- **LLM Interaction**: Proof generation and refinement
- **Theorem Proving**: Proof verification

### Jaeger Integration
```yaml
# jaeger-docker-compose.yml
version: '3.8'
services:
  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"
      - "14268:14268"
    environment:
      - COLLECTOR_ZIPKIN_HTTP_PORT=9411
```

## Alerting

### Alert Rules
```yaml
# alerts.yml
groups:
  - name: formal-circuits-gpt
    rules:
      - alert: HighVerificationFailureRate
        expr: rate(verification_failures_total[5m]) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High verification failure rate"
          description: "Verification failure rate is {{ $value }} failures/sec"
      
      - alert: SlowVerificationTime
        expr: avg(verification_duration_seconds) > 300
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Slow verification times"
          description: "Average verification time is {{ $value }} seconds"
      
      - alert: LLMAPIDown
        expr: up{job="formal-circuits-gpt"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "LLM API unavailable"
          description: "LLM API has been down for more than 1 minute"
```

### Notification Channels

#### Slack Integration
```python
from formal_circuits_gpt.monitoring import SlackNotifier

slack_notifier = SlackNotifier(
    webhook_url="https://hooks.slack.com/services/...",
    channel="#alerts",
    username="formal-circuits-gpt"
)

# Send alert
slack_notifier.send_alert(
    title="High Failure Rate",
    message="Verification failure rate exceeded threshold",
    severity="warning"
)
```

#### Email Notifications
```python
from formal_circuits_gpt.monitoring import EmailNotifier

email_notifier = EmailNotifier(
    smtp_server="smtp.example.com",
    username="alerts@example.com",
    password="password",
    recipients=["admin@example.com"]
)
```

## Security Monitoring

### Security Events
- Authentication failures
- Unauthorized access attempts
- Unusual API usage patterns
- Configuration changes
- Data access patterns

### Audit Logging
```json
{
  "timestamp": "2025-08-02T16:45:00Z",
  "event_type": "authentication",
  "result": "success",
  "user_id": "user123",
  "ip_address": "192.168.1.100",
  "user_agent": "formal-circuits-gpt-cli/0.1.0"
}
```

### Compliance Reporting
```python
from formal_circuits_gpt.monitoring import ComplianceReporter

reporter = ComplianceReporter()

# Generate compliance report
report = reporter.generate_report(
    start_date="2025-08-01",
    end_date="2025-08-31",
    report_type="security_audit"
)
```

## Deployment

### Monitoring Stack with Docker Compose
```yaml
# monitoring-stack.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
  
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana
  
  jaeger:
    image: jaegertracing/all-in-one
    ports:
      - "16686:16686"
      - "14268:14268"
  
  elasticsearch:
    image: elasticsearch:7.10.1
    environment:
      - discovery.type=single-node
    ports:
      - "9200:9200"
  
  kibana:
    image: kibana:7.10.1
    ports:
      - "5601:5601"
    depends_on:
      - elasticsearch

volumes:
  grafana-storage:
```

### Kubernetes Monitoring
```yaml
# monitoring-deployment.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
    scrape_configs:
      - job_name: 'formal-circuits-gpt'
        kubernetes_sd_configs:
          - role: pod
        relabel_configs:
          - source_labels: [__meta_kubernetes_pod_label_app]
            action: keep
            regex: formal-circuits-gpt

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prometheus
spec:
  replicas: 1
  selector:
    matchLabels:
      app: prometheus
  template:
    metadata:
      labels:
        app: prometheus
    spec:
      containers:
      - name: prometheus
        image: prom/prometheus
        ports:
        - containerPort: 9090
        volumeMounts:
        - name: config
          mountPath: /etc/prometheus
      volumes:
      - name: config
        configMap:
          name: prometheus-config
```

## Best Practices

### Monitoring Guidelines
1. **Start Simple**: Begin with basic health checks and key metrics
2. **Monitor User Journey**: Track the complete verification workflow
3. **Set Meaningful Alerts**: Avoid alert fatigue with thoughtful thresholds
4. **Regular Review**: Periodically review and update monitoring configuration

### Performance Considerations
1. **Sampling**: Use sampling for high-volume tracing
2. **Buffering**: Buffer metrics and logs to reduce I/O impact
3. **Retention**: Set appropriate retention periods for different data types
4. **Resource Limits**: Monitor the monitoring system itself

### Security Considerations
1. **Access Control**: Secure monitoring endpoints and dashboards
2. **Data Privacy**: Avoid logging sensitive information
3. **Audit Trail**: Maintain audit logs for monitoring configuration changes
4. **Encryption**: Use TLS for monitoring data transmission

---

For more detailed information, see:
- [Metrics Reference](metrics-reference.md)
- [Alert Runbooks](../runbooks/README.md)
- [Troubleshooting Guide](troubleshooting.md)
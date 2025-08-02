# Runbook: Performance Degradation

## Overview
This runbook addresses performance issues in formal-circuits-gpt, including slow verification times, high resource usage, and system bottlenecks.

## Performance Baselines

### Expected Performance Metrics
| Circuit Size | Expected Time | Memory Usage | Success Rate |
|-------------|---------------|--------------|--------------|
| Small (<100 gates) | <30 seconds | <500MB | >95% |
| Medium (100-1000 gates) | <300 seconds | <2GB | >90% |
| Large (1000+ gates) | <1800 seconds | <4GB | >85% |

### Key Performance Indicators
- **Verification Time**: Time from request to completion
- **Throughput**: Verifications per hour
- **Resource Utilization**: CPU, memory, I/O usage
- **Cache Hit Rate**: Percentage of cache hits
- **LLM Response Time**: API call latency

## Detection and Alerting

### Automated Alerts
```promql
# Slow verification times
avg_over_time(verification_duration_seconds[10m]) > 600

# High CPU usage
rate(cpu_usage_total[5m]) > 0.8

# Memory pressure
memory_usage_bytes / memory_total_bytes > 0.9

# Low cache hit rate
cache_hit_rate < 0.5

# High LLM latency
avg_over_time(llm_request_duration_seconds[5m]) > 30
```

### Manual Detection
```bash
# Check current performance
formal-circuits-gpt status --performance

# Monitor active verifications
ps aux | grep formal-circuits-gpt
top -p $(pgrep -f formal-circuits-gpt)

# Check system load
uptime
iostat 1 5
```

## Diagnostic Procedures

### Step 1: Identify Performance Bottleneck

#### System Resource Check
```bash
# CPU usage
top -bn1 | grep "Cpu(s)"
mpstat 1 5

# Memory usage
free -h
cat /proc/meminfo

# I/O performance
iostat -x 1 5
iotop -o

# Network (for LLM API calls)
netstat -i
ss -tuln
```

#### Application Metrics
```bash
# Check verification queue
formal-circuits-gpt queue status

# Cache statistics
formal-circuits-gpt cache stats

# Component performance
formal-circuits-gpt profile --component all
```

### Step 2: Component Analysis

#### LLM Provider Performance
```bash
# Test LLM response time
time curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"model":"gpt-3.5-turbo","messages":[{"role":"user","content":"test"}]}' \
     https://api.openai.com/v1/chat/completions

# Check API status
curl -s https://status.openai.com/api/v2/status.json

# Monitor concurrent requests
netstat -an | grep :443 | wc -l
```

#### Theorem Prover Performance
```bash
# Profile Isabelle performance
time echo 'theory Test imports Main begin lemma "True" by simp end' | isabelle process

# Profile Coq performance  
time echo 'Check True.' | coq

# Check prover resource usage
ps aux | grep -E "(isabelle|coq)"
```

#### Parser Performance
```bash
# Profile parsing time
time formal-circuits-gpt parse large_design.v --profile

# Check parse tree size
formal-circuits-gpt parse design.v --dump-ast | wc -l

# Memory usage during parsing
valgrind --tool=massif formal-circuits-gpt parse design.v
```

### Step 3: Historical Analysis
```bash
# Review performance trends
grep "verification_duration" /var/log/formal-circuits-gpt.log | tail -100

# Check for memory leaks
grep -i "memory\|leak" /var/log/formal-circuits-gpt.log

# Analyze error patterns
grep -i "timeout\|error" /var/log/formal-circuits-gpt.log | tail -50
```

## Common Performance Issues

### Issue 1: High LLM API Latency

**Symptoms:**
- Verification taking much longer than usual
- High `llm_request_duration_seconds` metric
- Timeouts in LLM requests

**Investigation:**
```bash
# Check LLM provider status
curl -s https://status.openai.com/api/v2/status.json | jq '.status.indicator'

# Test API latency
for i in {1..5}; do
  time curl -s -H "Authorization: Bearer $OPENAI_API_KEY" \
       https://api.openai.com/v1/models > /dev/null
done

# Check concurrent requests
lsof -i :443 | grep formal-circuits | wc -l
```

**Resolution:**
```bash
# Switch to faster model
export FORMAL_CIRCUITS_LLM_MODEL=gpt-3.5-turbo

# Reduce token usage
export FORMAL_CIRCUITS_MAX_TOKENS=2000

# Enable request batching
export FORMAL_CIRCUITS_BATCH_REQUESTS=true

# Use multiple providers for load balancing
cat > ~/.formal-circuits-gpt/config.yaml << EOF
llm:
  providers:
    - name: openai
      weight: 50
    - name: anthropic
      weight: 50
EOF
```

### Issue 2: Memory Exhaustion

**Symptoms:**
- Out of memory errors
- System becoming unresponsive
- Swap usage increasing

**Investigation:**
```bash
# Monitor memory usage
watch -n 1 'free -h; ps aux | grep formal-circuits-gpt | head -5'

# Check memory leaks
valgrind --leak-check=full formal-circuits-gpt verify design.v

# Analyze heap usage
pmap -d $(pgrep formal-circuits-gpt)
```

**Resolution:**
```bash
# Increase system memory
# Configure memory limits
echo 'vm.overcommit_memory=2' >> /etc/sysctl.conf
echo 'vm.overcommit_ratio=80' >> /etc/sysctl.conf
sysctl -p

# Tune application memory usage
export FORMAL_CIRCUITS_MAX_MEMORY=4096
export FORMAL_CIRCUITS_CACHE_SIZE=1024

# Enable memory optimization
export FORMAL_CIRCUITS_MEMORY_OPTIMIZE=true

# Restart with memory limits
systemctl edit formal-circuits-gpt.service
# Add:
# [Service]
# MemoryLimit=6G
# MemoryAccounting=yes
```

### Issue 3: Theorem Prover Bottlenecks

**Symptoms:**
- High `theorem_prover_duration_seconds`
- Provers using excessive CPU/memory
- Frequent prover timeouts

**Investigation:**
```bash
# Profile theorem prover execution
strace -c -p $(pgrep isabelle)
perf top -p $(pgrep isabelle)

# Check prover configuration
isabelle getenv ISABELLE_TOOL_OPTIONS
coq config

# Monitor prover resource usage
pidstat -r -p $(pgrep -f "isabelle|coq") 1 10
```

**Resolution:**
```bash
# Optimize Isabelle JVM settings
export ISABELLE_JVM_OPTIONS="-Xms2g -Xmx6g -XX:+UseG1GC"

# Tune Coq performance
export COQ_OPTIONS="-bt -noglob"

# Increase prover timeouts
export FORMAL_CIRCUITS_PROVER_TIMEOUT=600

# Use parallel proving
export FORMAL_CIRCUITS_PARALLEL_PROVERS=true

# Optimize proof search
export FORMAL_CIRCUITS_PROOF_STRATEGY=optimized
```

### Issue 4: Cache Performance Issues

**Symptoms:**
- Low cache hit rate
- Cache operations taking too long
- Frequent cache misses for recent items

**Investigation:**
```bash
# Analyze cache performance
formal-circuits-gpt cache analyze

# Check cache directory
du -sh ~/.formal-circuits-gpt/cache/
ls -la ~/.formal-circuits-gpt/cache/ | wc -l

# Monitor cache I/O
iotop -o | grep cache
```

**Resolution:**
```bash
# Optimize cache settings
export FORMAL_CIRCUITS_CACHE_SIZE=2048
export FORMAL_CIRCUITS_CACHE_TTL=86400

# Move cache to faster storage
mv ~/.formal-circuits-gpt/cache /tmp/fast-cache
ln -s /tmp/fast-cache ~/.formal-circuits-gpt/cache

# Rebuild cache index
formal-circuits-gpt cache rebuild

# Enable cache compression
export FORMAL_CIRCUITS_CACHE_COMPRESS=true
```

### Issue 5: I/O Bottlenecks

**Symptoms:**
- High I/O wait times
- Slow file operations
- Storage running out of space

**Investigation:**
```bash
# Check I/O performance
iostat -x 1 5
iotop -o

# Monitor disk usage
df -h
du -sh /var/log /tmp ~/.formal-circuits-gpt

# Check inode usage
df -i
```

**Resolution:**
```bash
# Clean up temporary files
find /tmp -name "formal-circuits-*" -mtime +1 -delete
find ~/.formal-circuits-gpt/cache -atime +30 -delete

# Optimize logging
export FORMAL_CIRCUITS_LOG_LEVEL=WARNING
logrotate -f /etc/logrotate.d/formal-circuits-gpt

# Move to faster storage
mv ~/.formal-circuits-gpt /fast-storage/
ln -s /fast-storage/.formal-circuits-gpt ~/.formal-circuits-gpt

# Enable async I/O
export FORMAL_CIRCUITS_ASYNC_IO=true
```

## Performance Optimization

### System-Level Optimizations

#### CPU Optimization
```bash
# Set CPU governor to performance
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Disable CPU throttling
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo

# Pin processes to specific cores
taskset -c 0-3 formal-circuits-gpt verify design.v
```

#### Memory Optimization
```bash
# Tune kernel parameters
echo 'vm.swappiness=10' >> /etc/sysctl.conf
echo 'vm.vfs_cache_pressure=50' >> /etc/sysctl.conf
echo 'vm.dirty_ratio=15' >> /etc/sysctl.conf
sysctl -p

# Enable transparent huge pages
echo always > /sys/kernel/mm/transparent_hugepage/enabled
```

#### I/O Optimization
```bash
# Tune I/O scheduler
echo deadline > /sys/block/sda/queue/scheduler

# Increase I/O readahead
echo 4096 > /sys/block/sda/queue/read_ahead_kb

# Mount with performance options
mount -o noatime,nodiratime /dev/sda1 /mnt/fast-storage
```

### Application-Level Optimizations

#### Parallel Processing
```yaml
# config.yaml
performance:
  parallel_workers: 4
  max_concurrent_verifications: 8
  parallel_parsing: true
  parallel_translation: true
```

#### Caching Strategy
```yaml
# config.yaml
cache:
  enabled: true
  size: 4096  # MB
  ttl: 86400  # seconds
  compression: true
  strategy: lru
  
  # Cache levels
  levels:
    - name: memory
      size: 1024
      ttl: 3600
    - name: disk
      size: 8192
      ttl: 86400
```

#### LLM Optimization
```yaml
# config.yaml
llm:
  optimization:
    request_batching: true
    connection_pooling: true
    timeout: 60
    max_retries: 3
    backoff_factor: 2
    
  # Model selection
  models:
    small_circuits: gpt-3.5-turbo
    medium_circuits: gpt-4
    large_circuits: gpt-4-32k
```

### Monitoring Performance Improvements

#### Before/After Metrics
```bash
# Baseline measurement
formal-circuits-gpt benchmark --circuits examples/ --iterations 10

# Apply optimizations
# ... configuration changes ...

# Measure improvements  
formal-circuits-gpt benchmark --circuits examples/ --iterations 10 --compare baseline.json
```

#### Continuous Monitoring
```yaml
# prometheus-alerts.yml
groups:
  - name: performance
    rules:
      - alert: PerformanceDegradation
        expr: avg_over_time(verification_duration_seconds[1h]) > 1.5 * avg_over_time(verification_duration_seconds[24h] offset 24h)
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Performance degradation detected"
```

## Capacity Planning

### Growth Projections
```bash
# Analyze usage trends
formal-circuits-gpt analytics usage-trends --period 30d

# Project resource requirements
formal-circuits-gpt capacity-plan --growth-rate 20% --period 1y
```

### Scaling Strategies

#### Horizontal Scaling
```yaml
# docker-compose.scale.yml
version: '3.8'
services:
  formal-circuits-gpt:
    deploy:
      replicas: 4
    environment:
      - FORMAL_CIRCUITS_WORKER_MODE=true
      - FORMAL_CIRCUITS_LOAD_BALANCER=redis://localhost:6379
```

#### Vertical Scaling
```bash
# Increase resource limits
systemctl edit formal-circuits-gpt.service
# Add:
# [Service]
# CPUQuota=400%
# MemoryLimit=8G
```

## Recovery Procedures

### Performance Recovery
```bash
# Quick performance fixes
systemctl restart formal-circuits-gpt
formal-circuits-gpt cache clear
formal-circuits-gpt optimize --quick

# Full performance reset
systemctl stop formal-circuits-gpt
rm -rf /tmp/formal-circuits-*
formal-circuits-gpt cache rebuild
systemctl start formal-circuits-gpt
```

### Emergency Procedures
```bash
# If system is unresponsive
killall -9 formal-circuits-gpt isabelle coq
systemctl start formal-circuits-gpt

# If memory exhaustion
sync
echo 3 > /proc/sys/vm/drop_caches
systemctl restart formal-circuits-gpt
```

## Prevention Strategies

### Regular Maintenance
```bash
# Weekly performance check
formal-circuits-gpt health-check --performance
formal-circuits-gpt cache cleanup
formal-circuits-gpt log rotate

# Monthly optimization
formal-circuits-gpt optimize --full
formal-circuits-gpt benchmark --save-baseline
```

### Monitoring Setup
```bash
# Set up performance monitoring
formal-circuits-gpt monitor setup --metrics prometheus
formal-circuits-gpt monitor setup --alerts slack

# Configure capacity alerts
formal-circuits-gpt monitor alert-rule "cpu_usage > 80% for 10m"
formal-circuits-gpt monitor alert-rule "memory_usage > 90% for 5m"
```

---

**Last Updated:** August 2025  
**Next Review:** September 2025  
**Owner:** Performance Team
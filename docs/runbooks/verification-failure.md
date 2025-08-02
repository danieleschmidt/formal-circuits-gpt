# Runbook: Verification Failure Investigation

## Overview
This runbook provides step-by-step procedures for investigating and resolving verification failures in formal-circuits-gpt.

## Severity Classification

### Critical (P0)
- Complete system failure
- All verifications failing
- Security breach or data loss

### High (P1) 
- High failure rate (>20%)
- Performance degradation (>5x normal time)
- LLM API unavailable

### Medium (P2)
- Moderate failure rate (5-20%)
- Specific circuit types failing
- Non-critical component failures

### Low (P3)
- Individual verification failures
- Minor performance issues
- Warning alerts

## Initial Response

### 1. Assess Severity
```bash
# Check overall health
curl http://localhost:8080/health

# Check recent metrics
curl http://localhost:9090/metrics | grep verification_success_rate

# Review recent logs
tail -n 100 /var/log/formal-circuits-gpt.log | grep ERROR
```

### 2. Gather Information
- When did the issue start?
- What changed recently?
- Which circuits/properties are affected?
- Error patterns in logs

### 3. Immediate Actions (P0/P1)
```bash
# Check system resources
top
df -h
free -m

# Restart service if necessary
systemctl restart formal-circuits-gpt

# Enable debug logging
export FORMAL_CIRCUITS_DEBUG=true
export FORMAL_CIRCUITS_LOG_LEVEL=DEBUG
```

## Diagnostic Procedures

### Step 1: System Health Check
```bash
# Check service status
systemctl status formal-circuits-gpt

# Verify dependencies
formal-circuits-gpt --check-setup

# Test theorem provers
isabelle version
coq --version

# Check disk space
df -h /var/log
df -h /tmp
```

### Step 2: Component Testing
```bash
# Test LLM connectivity
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     https://api.openai.com/v1/models

# Test simple verification
formal-circuits-gpt verify examples/simple_adder.v --debug

# Check cache status
ls -la ~/.formal-circuits-gpt/cache/
```

### Step 3: Log Analysis
```bash
# Search for error patterns
grep -i error /var/log/formal-circuits-gpt.log | tail -50

# Check for timeout issues
grep -i timeout /var/log/formal-circuits-gpt.log | tail -20

# Look for memory issues
grep -i "memory\|oom" /var/log/formal-circuits-gpt.log

# API rate limiting
grep -i "rate.limit\|429" /var/log/formal-circuits-gpt.log
```

## Common Issues and Solutions

### Issue 1: LLM API Failures
**Symptoms:**
- HTTP 401/403 errors
- "API key invalid" messages
- High latency or timeouts

**Investigation:**
```bash
# Check API key
echo $OPENAI_API_KEY | cut -c1-10
echo $ANTHROPIC_API_KEY | cut -c1-10

# Test API directly
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"model":"gpt-3.5-turbo","messages":[{"role":"user","content":"test"}]}' \
     https://api.openai.com/v1/chat/completions

# Check rate limits
grep "rate.limit" /var/log/formal-circuits-gpt.log | tail -10
```

**Resolution:**
```bash
# Verify API key is valid and has sufficient credits
# Implement exponential backoff
# Switch to alternative LLM provider
# Increase timeout values

# Update configuration
cat > ~/.formal-circuits-gpt/config.yaml << EOF
llm:
  openai:
    timeout: 120
    max_retries: 3
    backoff_factor: 2
EOF
```

### Issue 2: Theorem Prover Failures
**Symptoms:**
- "Proof verification failed" messages
- Prover timeout errors
- "Command not found" errors

**Investigation:**
```bash
# Check prover installation
which isabelle
which coq

# Test prover directly
echo 'theory Test imports Main begin end' | isabelle process
echo 'Check 1 + 1.' | coq

# Check permissions
ls -la /usr/local/bin/isabelle
ls -la /usr/bin/coq

# Monitor resource usage during proof
top -p $(pgrep isabelle)
```

**Resolution:**
```bash
# Reinstall theorem provers
sudo apt-get update
sudo apt-get install --reinstall coq

# For Isabelle (if manual installation needed)
wget https://isabelle.in.tum.de/dist/Isabelle2024.tar.gz
tar -xzf Isabelle2024.tar.gz
sudo mv Isabelle2024 /opt/
sudo ln -sf /opt/Isabelle2024/bin/isabelle /usr/local/bin/

# Increase timeout
export FORMAL_CIRCUITS_PROVER_TIMEOUT=300

# Clear prover cache
rm -rf ~/.isabelle/Isabelle2024/heaps
```

### Issue 3: Memory/Performance Issues
**Symptoms:**
- Out of memory errors
- Very slow verification times
- System becoming unresponsive

**Investigation:**
```bash
# Monitor memory usage
free -m
cat /proc/meminfo

# Check for memory leaks
ps aux | grep formal-circuits-gpt
pmap $(pgrep -f formal-circuits-gpt)

# Monitor swap usage
swapon --show
vmstat 1 10

# Check for large files
du -sh ~/.formal-circuits-gpt/
du -sh /tmp/formal-circuits-*
```

**Resolution:**
```bash
# Increase system memory limits
ulimit -m 4194304  # 4GB

# Tune JVM for Isabelle
export ISABELLE_JVM_OPTIONS="-Xms2g -Xmx4g"

# Enable parallel processing limits
export FORMAL_CIRCUITS_MAX_WORKERS=2

# Clear cache
rm -rf ~/.formal-circuits-gpt/cache/*

# Restart with limited memory
systemctl edit formal-circuits-gpt.service
# Add:
# [Service]
# MemoryLimit=4G
# MemoryAccounting=yes
```

### Issue 4: Parsing Failures
**Symptoms:**
- "Parse error" messages
- "Unsupported syntax" errors
- Crashes during circuit parsing

**Investigation:**
```bash
# Test parsing standalone
formal-circuits-gpt parse design.v --debug

# Check syntax with external tools
verilator --lint-only design.v
ghdl -s design.vhd

# Validate file encoding
file design.v
hexdump -C design.v | head

# Check for special characters
grep -P "[^\x00-\x7F]" design.v
```

**Resolution:**
```bash
# Convert file encoding if needed
iconv -f ISO-8859-1 -t UTF-8 design.v > design_utf8.v

# Remove unsupported constructs
# Use simpler syntax
# Update parser if needed

# Test with minimal example
echo 'module test(input a, output b); assign b = a; endmodule' > test.v
formal-circuits-gpt verify test.v
```

### Issue 5: Cache Corruption
**Symptoms:**
- Inconsistent results
- "Cache miss" for recently cached items
- Verification slower than expected

**Investigation:**
```bash
# Check cache directory
ls -la ~/.formal-circuits-gpt/cache/
du -sh ~/.formal-circuits-gpt/cache/

# Check cache integrity
find ~/.formal-circuits-gpt/cache/ -name "*.cache" -exec file {} \;

# Monitor cache hit rate
grep "cache.hit" /var/log/formal-circuits-gpt.log | tail -20
```

**Resolution:**
```bash
# Clear cache completely
rm -rf ~/.formal-circuits-gpt/cache/*

# Rebuild cache directory structure
mkdir -p ~/.formal-circuits-gpt/cache/{proofs,lemmas,translations}

# Disable cache temporarily
export FORMAL_CIRCUITS_CACHE_ENABLED=false

# Test without cache
formal-circuits-gpt verify design.v --no-cache
```

## Escalation Procedures

### Level 1: Self-Service
- Check runbooks
- Review documentation
- Basic troubleshooting

### Level 2: Team Support
- Contact development team
- Provide logs and diagnostic information
- Implement temporary workarounds

### Level 3: Vendor Support
- Contact LLM provider support
- Engage theorem prover community
- Critical system failures

## Recovery Procedures

### Service Recovery
```bash
# Graceful restart
systemctl reload formal-circuits-gpt

# Full restart
systemctl restart formal-circuits-gpt

# Clear all state
systemctl stop formal-circuits-gpt
rm -rf /tmp/formal-circuits-*
rm -rf ~/.formal-circuits-gpt/cache/*
systemctl start formal-circuits-gpt
```

### Data Recovery
```bash
# Restore from backup
cp /backup/formal-circuits-gpt-config.yaml ~/.formal-circuits-gpt/config.yaml

# Rebuild cache from scratch
formal-circuits-gpt cache rebuild

# Verify system integrity
formal-circuits-gpt verify examples/ --test-mode
```

## Post-Incident Actions

### 1. Root Cause Analysis
- Identify root cause
- Document findings
- Update monitoring to prevent recurrence

### 2. Update Procedures
- Update runbooks based on lessons learned
- Improve monitoring and alerting
- Add automation for common fixes

### 3. Communication
- Notify stakeholders of resolution
- Provide post-mortem report
- Schedule follow-up reviews

## Monitoring and Alerting

### Key Metrics to Watch
```promql
# Verification success rate
rate(verification_success_total[5m]) / rate(verifications_total[5m])

# Average verification time
avg(verification_duration_seconds)

# Error rate by component
rate(errors_total[5m]) by (component)

# Memory usage
process_memory_usage_bytes

# API response times
avg(llm_request_duration_seconds)
```

### Alert Thresholds
- Success rate < 80%: Warning
- Success rate < 50%: Critical
- Avg verification time > 300s: Warning
- Memory usage > 80%: Warning
- API errors > 10/min: Critical

## Contact Information

### Emergency Contacts
- On-call Engineer: +1-555-0123
- Team Lead: engineer@example.com
- System Admin: admin@example.com

### Escalation Matrix
| Severity | Response Time | Contact |
|----------|---------------|---------|
| P0 | 15 minutes | On-call + Team Lead |
| P1 | 1 hour | On-call |
| P2 | 4 hours | Team member |
| P3 | Next business day | Team member |

---

**Last Updated:** August 2025  
**Next Review:** September 2025  
**Owner:** Development Team
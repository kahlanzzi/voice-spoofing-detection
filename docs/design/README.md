# Task Management API – k6 Performance Testing
## CS 428 – Spring 2025

Everything you need to run the full test suite from scratch.

---

## 📁 Project Structure

```
task-api/
├── server.js                    ← Express API (4 CRUD endpoints)
├── package.json
├── report.md                    ← Section C: Analysis & Reporting
└── k6-tests/
    ├── performance-test.js      ← A1: Baseline Load Test (50 VUs, 6 min)
    ├── spike-test.js            ← A2: Spike Test (10→500 VUs)
    └── stress-test.js           ← A3: Stress Test (ramp to 200 VUs)
```

---

## Step 1 – Install Prerequisites

### Node.js API
```bash
# In the task-api/ folder
npm install
```

### k6 (performance testing tool)
```bash
# macOS (Homebrew)
brew install k6

# Windows (Chocolatey)
choco install k6

# Linux (Debian/Ubuntu)
sudo gpg -k
sudo gpg --no-default-keyring --keyring /usr/share/keyrings/k6-archive-keyring.gpg \
    --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D69
echo "deb [signed-by=/usr/share/keyrings/k6-archive-keyring.gpg] https://dl.k6.io/deb stable main" \
    | sudo tee /etc/apt/sources.list.d/k6.list
sudo apt-get update && sudo apt-get install k6
```

---

## Step 2 – Start the API Server

Open **Terminal 1** and run:
```bash
cd task-api
node server.js
# ✅  Task API running on http://localhost:3000
```

**Verify it works:**
```bash
curl http://localhost:3000/health
curl http://localhost:3000/api/tasks
```

---

## Step 3 – Run the k6 Tests

Open **Terminal 2** (keep the server running in Terminal 1).

### 🔵 A1 – Baseline Load Test (50 VUs, 6 min)
```bash
k6 run k6-tests/performance-test.js
```

### 🔴 A2 – Spike Test (10 → 500 VUs)
```bash
k6 run k6-tests/spike-test.js
```

### 🟠 A3 – Stress Test (gradual ramp to 200 VUs)
```bash
k6 run k6-tests/stress-test.js
```

---

## Step 4 – Save Results to JSON (Optional but recommended)

```bash
# Baseline
k6 run --out json=results/baseline-results.json k6-tests/performance-test.js

# Spike
k6 run --out json=results/spike-results.json k6-tests/spike-test.js

# Stress
k6 run --out json=results/stress-results.json k6-tests/stress-test.js
```

---

## Step 5 – Verbose Output with Summary

```bash
# Run all three back-to-back with a shell script:
#!/bin/bash
echo "=== BASELINE ===" && k6 run k6-tests/performance-test.js
echo "=== SPIKE ==="    && k6 run k6-tests/spike-test.js
echo "=== STRESS ==="   && k6 run k6-tests/stress-test.js
```

---

## Key Metrics to Watch in Output

| Metric                  | What it measures                              | Target (Baseline) |
|-------------------------|-----------------------------------------------|-------------------|
| `http_req_duration p95` | 95th percentile response time                 | < 500 ms          |
| `http_req_failed`       | % of requests with HTTP errors                | < 1%              |
| `errors`                | Custom error rate (check failures)            | < 1%              |
| `http_req_blocked`      | Time waiting for a free TCP connection        | < 50 ms           |
| `http_req_waiting`      | TTFB – server processing time                 | < 200 ms          |
| `vus_max`               | Maximum VUs active during the test            | per scenario      |
| `iterations`            | Total number of script executions             | –                 |

---

## Quick API Reference

| Method | Endpoint            | Body                                   | Expected |
|--------|---------------------|----------------------------------------|----------|
| GET    | /api/tasks          | –                                      | 200      |
| POST   | /api/tasks          | `{"title":"x","completed":false}`      | 201      |
| PUT    | /api/tasks/:id      | `{"completed":true}`                   | 200      |
| DELETE | /api/tasks/:id      | –                                      | 200      |
| GET    | /health             | –                                      | 200      |

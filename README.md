# FraudGraph Investigator

An agentic fraud investigation system built on **TigerGraph**. Given an alert (a model score, a customer complaint or an
analyst request) it opens a case, investigates the transaction graph, decides whether it knows enough, asks for more
evidence when it does not, and recommends the next best action with the approval route the bank's policy requires. Every
case is written back into the graph as memory for the next one.

Built for the TigerGraph × Hacker House Goa 2026 challenge on the HHGOA_IEEE dataset.

Architecture diagram and design notes: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Highlights

- **TigerGraph as the investigation substrate**: 590,742 transactions, 14,215 cards, 9,706 device profiles, billing
  regions, email domains, a per-card `NEXT` chain, 5,565 closed cases and the agent's own cases in one graph
  (`gsql/schema.gsql`).
- **12 GSQL installed queries as agent tools** (`gsql/queries/investigation.gsql`): card windows, behavioural baselines,
  device neighbours, a windowed device-ring expansion, region clusters, recurring charges, 2-hop case memory and two
  **TigerVector** searches.
- **TigerGraph MCP**: the same tools are served through `tigergraph-mcp`; by default the agent calls
  `tigergraph__run_installed_query` over MCP.
- **GraphRAG**: policy rules, the five typologies and regulatory guidance (FinCEN, FATF, FFIEC) are chunked into
  `DocChunk` vectors; closed cases are embedded into `ClosedCase.emb`. Retrieved context grounds the explanations and SARs.
- **Learning from case memory**: a model trained on the closed cases beats the bank's risk score
  (holdout AUC **0.955 vs 0.866**; confirmed-vs-cleared **0.877 vs 0.052**) and becomes one evidence family.
- **Uncertainty-aware loop**: independent evidence families, the policy's stopping rule, R1 verify-before-block, a
  simulated step-up / customer reply with the assumption recorded, and an explicit *before → after* recommendation.
- **Policy as code**: actions, auto/L1/L2 routes, R1–R10, case-vs-SAR (§3a) in `fraudagent/agent/policy.py`; the API
  refuses approvals from the wrong role.
- **Graph algorithms for monitoring**: `ring_components` runs weakly-connected-components (min-label propagation) over
  proxied card↔device links in GSQL, and `band_bursts` finds threshold-structuring candidates; the monitor confirms
  and investigates each hit.
- **Two undocumented patterns detected**: sub-$500 structuring bursts and an anonymous-proxy device ring. The monitor
  found 30 more cases in November–December beyond the case pack (`monitor/`).
- **LLM chain: Gemini 3.5 Flash-Lite → local Qwen2.5 3B (Ollama)**. Gemini (free tier) is used first; when its quota
  runs out the agent switches to the local model automatically. The LLM re-ranks tools and writes the case
  summaries and SAR narratives from the retrieved evidence and policy text; an ID guard rejects any output that
  mentions an entity not in the evidence, and the policy engine alone decides actions and routes.
- **Analyst dashboard**: live agent timeline (SSE), evidence with entity IDs, evidence graph, next best action with
  approvals, SAR and case memory.

## Results on the 20 benchmark cases

| Case | Trigger | Verdict | p | Pattern | Exposure | Evidence asked | Final actions (route) | SAR |
|---|---|---|---|---|---|---|---|---|
| HHG-001 | risk score | legitimate | 0.06 | none | $0.00 | customer_validation | ALLOW_TRANSACTION (auto), CREATE_CASE (auto), CLOSE_NO_FRAUD (auto) | no |
| HHG-002 | risk score | fraud | 0.93 | card not present fraud | $292.36 | – | BLOCK_CARD (L1), CREATE_CASE (auto) | no |
| HHG-003 | customer report | legitimate | 0.06 | none | $0.00 | customer_validation | ALLOW_TRANSACTION (auto), CREATE_CASE (auto), CLOSE_NO_FRAUD (auto) | no |
| HHG-004 | customer report | legitimate | 0.05 | none | $0.00 | customer_validation | ALLOW_TRANSACTION (auto), CREATE_CASE (auto), CLOSE_NO_FRAUD (auto) | no |
| HHG-005 | risk score | legitimate | 0.07 | none | $0.00 | – | ALLOW_TRANSACTION (auto), CLOSE_NO_FRAUD (auto) | no |
| HHG-006 | customer report | fraud | 0.98 | undocumented | $1,906.07 | – | BLOCK_CARD (L1), CREATE_CASE (auto), FILE_REPORT (L2), ESCALATE_TO_ANALYST (auto) | yes |
| HHG-007 | risk score | fraud | 0.88 | account takeover | $148.89 | – | BLOCK_CARD (L1), CREATE_CASE (auto) | no |
| HHG-008 | customer report | fraud | 0.90 | card not present new device | $166.97 | – | BLOCK_CARD (L1), CREATE_CASE (auto) | no |
| HHG-009 | customer report | fraud | 0.85 | card not present fraud | $30.02 | – | BLOCK_CARD (L1), CREATE_CASE (auto) | no |
| HHG-010 | risk score | legitimate | 0.03 | none | $0.00 | step_up_auth | ALLOW_TRANSACTION (auto), CREATE_CASE (auto), CLOSE_NO_FRAUD (auto) | no |
| HHG-011 | customer report | fraud | 0.96 | card not present new device | $131.30 | customer_validation | BLOCK_CARD (L1), CREATE_CASE (auto) | no |
| HHG-012 | risk score | legitimate | 0.06 | none | $0.00 | – | ALLOW_TRANSACTION (auto), CLOSE_NO_FRAUD (auto) | no |
| HHG-013 | risk score | legitimate | 0.02 | none | $0.00 | step_up_auth | ALLOW_TRANSACTION (auto), CREATE_CASE (auto), CLOSE_NO_FRAUD (auto) | no |
| HHG-014 | analyst request | fraud | 0.98 | undocumented | $439.61 | step_up_auth | BLOCK_CARD (L1), CREATE_CASE (auto), MONITOR_CONNECTED_CARDS (auto), FILE_REPORT (L2), ESCALATE_TO_ANALYST (auto) | yes |
| HHG-015 | risk score | legitimate | 0.07 | none | $0.00 | step_up_auth | ALLOW_TRANSACTION (auto), CREATE_CASE (auto), CLOSE_NO_FRAUD (auto) | no |
| HHG-016 | customer report | fraud | 0.88 | card not present new device | $59.67 | – | BLOCK_CARD (L1), CREATE_CASE (auto) | no |
| HHG-017 | risk score | legitimate | 0.03 | none | $0.00 | customer_validation | ALLOW_TRANSACTION (auto), CREATE_CASE (auto), CLOSE_NO_FRAUD (auto) | no |
| HHG-018 | customer report | legitimate | 0.04 | none | $0.00 | customer_validation | ALLOW_TRANSACTION (auto), CREATE_CASE (auto), CLOSE_NO_FRAUD (auto) | no |
| HHG-019 | risk score | fraud | 0.97 | card not present new device | $99.92 | step_up_auth | BLOCK_CARD (L1), CREATE_CASE (auto) | no |
| HHG-020 | risk score | legitimate | 0.10 | none | $0.00 | – | ALLOW_TRANSACTION (auto), CLOSE_NO_FRAUD (auto) | no |

Screenshots: [`docs/img/`](docs/img). Answer files: [`cases/`](cases). Autonomous monitoring (optional, Innovation): [`monitor/`](monitor).
Back-test on 200 October closed cases (graph evidence only, no customer contact): 72.5% verdict accuracy, AUC 0.75
([`docs/backtest_october.json`](docs/backtest_october.json)).

## Testing and robustness

| Check | Result |
|---|---|
| `pytest tests/` | 23 tests: policy rules R1–R10, SAR test, routes, stopping rule, detectors, answer-file validation |
| `scripts/stress_test.py` | 144 random alerts across channels, months and trigger types: 0 crashes, 0 invalid outputs, p50 0.09 s / p95 0.2 s per investigation on TigerGraph |
| Edge cases | single-transaction cards, the largest aggregated card, in-person rows without a region, unknown transaction IDs (clean "not found") |
| Determinism | all 20 cases give identical verdicts, episodes and actions when re-run, in forward or reverse order |
| API | 200 concurrent reads in 1 s; 3 simultaneous live investigations; strict input validation; an approval from the wrong role is refused (403); a decided action cannot be approved twice (409) |
| LLM outage | with no LLM the agent completes with templates; with Gemini out of quota it falls back to local Qwen2.5 3B |
| `scripts/validate_answers.py` | 20/20 answer files match the format and reference only IDs that exist in the dataset |

Stress testing found and fixed three real bugs: per-investigation state leaking into the next case's similar-case
query, non-deterministic ordering of tied vector-search results, and approvals that accepted unknown roles.

## Quick start

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt tigergraph-mcp
cp .env.example .env            # add a free Gemini/Groq key if you want LLM-written narratives

# 1. put the HHGOA_IEEE files in data/ (transactions.csv, identity.csv, closed_cases_history.csv, case_pack.csv)

# 2. TigerGraph Community Edition (Docker). On Apple Silicon run it under Rosetta:
#    colima start tg --arch aarch64 --vm-type vz --vz-rosetta --cpu 6 --memory 9
docker run -d --name tigergraph --platform linux/amd64 -p 14240:14240 -p 9000:9000 \
  -v "$PWD":/home/tigergraph/project tigergraph/community:4.2.5

# 3. build the graph and memory
python scripts/prepare_graph_data.py          # cards, devices, edges, closed-case profiles
python scripts/train_case_model.py            # case-memory model (writes docs/case_model_metrics.json)
scripts/gsql.sh gsql/schema.gsql && scripts/gsql.sh gsql/vectors.gsql
python scripts/load_graph_rest.py             # ~2 minutes over RESTPP
python scripts/build_memory.py --push         # embeddings -> TigerVector
scripts/gsql.sh gsql/queries/investigation.gsql && scripts/gsql.sh gsql/queries/monitoring.gsql
scripts/gsql.sh gsql/install.gsql

# optional local LLM fallback
brew install ollama && ollama serve & ollama pull qwen2.5:3b

# 4. run
python scripts/run_benchmark.py               # writes cases/HHG-*.json
python scripts/monitor.py                     # optional sweep -> monitor/
uvicorn fraudagent.api.server:app --port 8000 # dashboard at http://localhost:8000
```

`GRAPH_BACKEND=mcp` routes every tool call through the TigerGraph MCP server; `GRAPH_BACKEND=local` answers the same
tools from the prepared parquet files (used for tests and UI work without a database).

## How an investigation runs

1. **Trigger** → open case → `txn_detail`.
2. **Investigate**: the planner chooses the next query from what it already knows (device present? card-present?
   dispute?); an LLM, when configured, may re-rank the remaining tools.
3. **Detect**: card testing (R5), structuring, device rings (R6/R9), region clusters, new device / proxy, first-time
   region, amount and product anomalies, recurring charges (R7), recent fraud on the card, similar closed cases, and the
   case-memory model score.
4. **Assess**: log-odds per evidence family, capped so correlated evidence counts once; probability and the number of
   independent families.
5. **Gather more evidence** when policy §6 is not met: record the initial action, request step-up or customer
   validation, simulate the reply from the posterior and state the assumption.
6. **Decide**: the policy engine maps the facts to actions and routes; SAR only when §3a holds.
7. **Explain**: summary, SAR narrative, pattern description, stop reason.
8. **Remember**: `InvestigationCase` + edges + embedding written to TigerGraph.

Details: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Repository layout

```
fraudagent/
  agent/       investigator (state machine), detectors, policy engine, simulator, narrative, llm
  graph/       TigerGraph REST client, graph tools (TigerGraph / MCP / local), profile helpers
  memory/      embedder, knowledge chunking, case store (graph write-back)
  api/         FastAPI server: cases, SSE investigations, evidence graph, approvals
gsql/          schema, vector attributes, loading job, installed queries
knowledge/     fraud policy, typologies, regulatory guidance (GraphRAG corpus)
scripts/       data prep, model training, loader, memory build, benchmark, monitor, back-test
ui/            analyst dashboard (vanilla JS + vis-network)
cases/         the 20 answer files
monitor/       autonomous monitoring investigations
docs/          architecture, blog, demo script, metrics
```

## Notes and assumptions

- Card IDs are not a column in the data. They are rebuilt as customer + card network, numbered from the most recently
  first-seen network; this reproduces 99.3% of the labelled card IDs and the labels override the rest.
- Customer, step-up and analyst responses are simulated (policy §5); every assumption is in `evidence_requests`.
- Actions are simulated APIs: `auto` actions are "executed" by the agent, `L1`/`L2` actions wait in the approval queue.
- Vesta's V/C/D/M/id features are used only as unnamed model inputs; evidence never claims to know what they mean.
- The original public IEEE-CIS files were not used.

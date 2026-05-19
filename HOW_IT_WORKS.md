# How the AI Agent Works: Complete Step-by-Step Guide

## Overview

The Agricultural AI Agent is a **dataset-first, expert-assisted review system** that processes agricultural queries through multiple stages of routing, retrieval, and validation. This document explains every step from user input to final response.

---

## Complete Processing Pipeline

### Phase 1: Query Input & Initialization

**Step 1.1: User Submits Query**
- User enters a question via CLI, API, or batch mode
- Example: `"What crops are grown in Andhra Pradesh?"`

**Step 1.2: Agent Initialization (First Run Only)**
```
Agent.__init__()
  ├─ Load configuration (config.py)
  │  └─ Model whitelist, system prompt, thresholds
  ├─ Initialize POP Bank Loader
  │  ├─ Load pop_bank_index.json (30 states, 574 crops)
  │  └─ Build state lookup + aliases
  │     └─ "Andhra Pradesh" → "Andra Pradesh" (fuzzy match)
  ├─ Initialize Memory
  │  ├─ Connect to SQLite (reports/chat_memory.sqlite3)
  │  └─ Load prior conversation history
  └─ Select Ollama model (from approved whitelist)
     └─ Fallback to available model if preferred unavailable
```

**Step 1.3: Query Added to Memory**
```python
memory.add("user", query)
# Written to SQLite immediately
# Timestamp recorded: 2026-05-19T11:30:45
```

---

### Phase 2: Query Classification

**Step 2.1: Classify Query Intent**
```
_classify_query(user_input)
  ├─ Check if lookup query
  │  └─ "What crops are grown in Andhra Pradesh?" → YES
  │  └─ Route: "pop_lookup"
  │
  ├─ Check if summarization
  │  └─ "Summarize crop trends in Punjab" → YES
  │  └─ Route: "summarization"
  │
  ├─ Check if reasoning
  │  └─ "Why is soil fertility low?" → YES
  │  └─ Route: "reasoning"
  │
  └─ Default: "general" for other queries
```

**Step 2.2: Lookup Query Classification Details**
```
_is_lookup_query()
  ├─ Extract query tokens
  │  └─ user_lower = query.lower()
  │
  ├─ Check for non-lookup markers (why, how, analyze, etc.)
  │  └─ If found → reject lookup route
  │
  ├─ Check for state mentions
  │  └─ Call _find_state_mentions(user_input)
  │
  └─ Check for lookup keywords (what crops, list, show, etc.)
     └─ If state + lookup keyword → Route: "pop_lookup"
```

**Classification Result:**
```
Query: "What crops are grown in Andhra Pradesh?"
Route: "pop_lookup"
State Mention: "Andhra Pradesh" (resolved to "Andra Pradesh")
```

---

### Phase 3: Attempt Direct Lookup (Fast Path)

**Step 3.1: Check for Direct Answer**
```python
if route == "pop_lookup":
    direct_answer = _build_direct_pop_bank_answer(user_input)
```

**Step 3.2: Build Direct Answer from Index**
```
_build_direct_pop_bank_answer()
  ├─ Get state matches via _find_state_mentions()
  │  └─ "Andhra Pradesh" → resolve to "Andra Pradesh"
  │
  ├─ Fetch crop list for state
  │  └─ state_crops["Andra Pradesh"] = [40 crops]
  │  └─ Extract: Blackgram, Cotton, Groundnut, ...
  │
  ├─ Format answer
  │  └─ "Andra Pradesh: 40 crop profiles are indexed."
  │  └─ "Examples: Blackgram Varieties, Cotton, ..."
  │
  └─ Return: (answer_text, confidence=1.0)
```

**Step 3.3: If Direct Answer Found**
```
if direct_answer:
    ├─ Add to memory: memory.add("assistant", direct_answer)
    ├─ Log interaction (see Phase 5)
    └─ RETURN answer immediately (< 100ms)
       └─ Skip all further steps!
```

**For this query, direct answer exists:**
```
Answer: "Andra Pradesh: 40 crop profiles are indexed. Examples: Blackgram Varieties, Cotton Varieties, Cotton, Cotton_v1, ... +28 more."
Confidence: 1.0
Time: < 50ms
Route: COMPLETE ✓
```

---

### Phase 4: Dataset Semantic Search (If No Direct Answer)

**Step 4.1: Tokenize Query**
```
_tokenize_query("Tell me about rice in Karnataka")
  └─ Extract words > 2 chars: ["tell", "about", "rice", "karnataka"]
```

**Step 4.2: Search Across All Cached States**
```
retrieve_from_dataset(query, top_k=3)
  ├─ For each state in 30 states:
  │  ├─ _load_state_data(state)  # Lazy load from cache/PDF
  │  │  ├─ Check if already loaded → skip
  │  │  └─ Load from pop_bank_cache.json or extract PDF
  │  │
  │  └─ For each crop in state:
  │     ├─ content_tokens = tokenize(content[:500])
  │     ├─ matches = query_tokens ∩ content_tokens
  │     ├─ confidence = len(matches) / len(query_tokens)
  │     └─ If matches > 0 → add to results
  │
  └─ Sort results by confidence (descending)
```

**Step 4.3: Calculate Relevance Score**
```
Query: "Tell me about rice in Karnataka"
Query tokens: {tell, about, rice, karnataka}

Result 1: Karnataka - Rice Farming
  Content tokens found: {rice, farming}
  Matches: 1 (rice)
  Confidence: 1/4 = 0.25

Result 2: Odisha - Rice Farming
  Content tokens found: {rice, farming}
  Matches: 1 (rice)
  Confidence: 1/4 = 0.25

Result 3: Punjab - Citrus Fruit Drop
  Content tokens found: {} (no matches)
  Skipped
```

**Step 4.4: Get Best Dataset Answer**
```
get_best_dataset_answer(query, min_confidence=0.3)
  ├─ Top result: (Odisha, Rice Farming, confidence=0.25)
  ├─ Check confidence >= 0.3? NO (0.25 < 0.3)
  └─ Return ("", 0.0)  # Too low confidence
```

**Result: Dataset search found match but below threshold**
```
Best match confidence: 0.25
Minimum threshold: 0.3
Decision: ESCALATE TO LLM
```

---

### Phase 5: LLM Fallback (If Confidence Too Low)

**Step 5.1: Select Runtime Model by Route**
```
_select_runtime_model(route)
  ├─ If route == "summarization"
  │  └─ Prefer: phi4-mini:3.8b → llama3:latest
  │
  ├─ If route == "reasoning"
  │  └─ Prefer: qwen3:4b → qwen2.5:1.5b → llama3:latest
  │
  └─ Default: use agent.model_name
     └─ phi4-mini:3.8b (from config/CLI flag)
```

**For low-confidence dataset match:**
```
Route: "general" (fallback)
Selected model: phi4-mini:3.8b
```

**Step 5.2: Build LLM Prompt**
```
build_messages(user_input)
  ├─ Start with system message (config.SYSTEM_PROMPT)
  │  └─ "You are a direct, careful agricultural AI assistant..."
  │
  ├─ Add relevant knowledge from POP Bank (if any)
  │  └─ _extract_relevant_knowledge(user_input)
  │  └─ Search for crop mentions: "rice" → found in states
  │  └─ Inject: "[Rice Information] ..."
  │
  ├─ Add conversation history
  │  └─ memory.get_all()
  │  └─ Prior user messages + assistant responses
  │
  └─ Return: [system_msg, user1, assistant1, user2, ...]
```

**Step 5.3: Call LLM**
```python
result = ollama.chat(
    model="phi4-mini:3.8b",
    messages=[system_msg, ...history...]
)
response = result["message"]["content"]
```

**Example LLM Response:**
```
"Rice cultivation in Karnataka is a significant agricultural activity,
particularly in regions like the Krishna valley. The state produces
both short-grained and long-grained rice varieties. Key growing regions
include Tumkur, Mandya, and Hassan districts..."
```

**Step 5.4: Error Handling**
```
if LLM call fails:
    ├─ Catch: Exception (timeout, model not available, etc.)
    ├─ Return: "Data not found: selected Ollama model is not available..."
    └─ Log as error
```

---

### Phase 6: Response Storage & Logging

**Step 6.1: Add Response to Memory**
```python
memory.add("assistant", response)
# Immediately written to SQLite
```

**Step 6.2: Log Interaction**
```
_log_interaction(
    question="Tell me about rice in Karnataka",
    route="general",                    # Final route taken
    model="phi4-mini:3.8b",             # Model used
    response=response,                  # Full response text
    started_at=datetime_obj             # Query start time
)
```

**Step 6.3: Calculate Metrics**
```
elapsed_time = (datetime.now() - started_at).total_seconds()
  └─ Example: 2.145 seconds

accuracy_estimate = 1.0 if route=="pop_lookup" else ...
                  = 0.9 if response found else ...
                  = 0.0 if "not found" markers detected
  └─ Example: 0.9 (good LLM response, no "I don't know")
```

**Step 6.4: Write JSONL Log Entry**
```json
{
  "timestamp": "2026-05-19T11:30:47",
  "question": "Tell me about rice in Karnataka",
  "route": "general",
  "model": "phi4-mini:3.8b",
  "response_time": 2.145,
  "accuracy": 0.9
}
```

**File: `reports/interaction_log.jsonl`**
```
(append mode - each query adds one line)
```

---

### Phase 7: Return Response to User

**Step 7.1: Generate Report (for one-shot mode)**
```
generate_report(user_input)
  ├─ Check if direct answer (pop_lookup route)
  │  └─ Format: "# AI Agent Response\n## Question\n...\n## Mode: POP Bank"
  │
  └─ Otherwise:
     ├─ Call chat() for LLM response
     ├─ Format: "# AI Agent Response\n## Question\n...\n## Answer\n...\n## Model"
     └─ Return markdown
```

**Output to User:**
```markdown
# AI Agent Response

## Question
Tell me about rice in Karnataka

## Answer
Rice cultivation in Karnataka is a significant agricultural activity...

## Model
phi4-mini:3.8b
```

**Step 7.2: Save Session (on exit)**
```
save_session_memory()
  ├─ Export conversation to: reports/session_memory_20260519_113047.md
  │  └─ Human-readable markdown format
  │
  └─ Export to: reports/session_memory_20260519_113047.json
     └─ Machine-readable JSON format
     └─ All messages with roles + content
```

**Example session_memory_*.md:**
```markdown
# Session Memory

## User
What crops are grown in Andhra Pradesh?

## Assistant
Andra Pradesh: 40 crop profiles are indexed...

## User
Tell me about rice in Karnataka

## Assistant
Rice cultivation in Karnataka...
```

---

## Decision Flow Diagram

```
┌─ Query Input ─┐
│ "What crops   │
│  in Andhra    │
│  Pradesh?"    │
└──────┬────────┘
       │
       ▼
┌──────────────────────────┐
│ Classify Query Intent    │
│ Route: pop_lookup        │
└──────┬───────────────────┘
       │
       ▼
    ┌─ Direct Lookup? ────────────┐
    │ _build_direct_pop_bank()    │
    │ → Answer found              │
    └──────┬─────────────────────┬┘
    YES    │                     │    NO
           │                     │
    ┌──────▼─────────┐    ┌──────▼──────────────┐
    │ Return FAST    │    │ Semantic Search    │
    │ (< 100ms)      │    │ (retrieve_from...)  │
    │                │    │ → Confidence: 0.25  │
    └────────┬───────┘    └──────┬───────────┬──┘
             │                   │           │
             │          Conf >= 0.3?      NO (Low)
             │            YES / NO         │
             │             │               │
             │        ┌────▼────┐    ┌─────▼──────┐
             │        │Return   │    │ LLM        │
             │        │Dataset  │    │ Fallback   │
             │        │Answer   │    │ (Ollama)   │
             │        └────┬────┘    └─────┬──────┘
             │             │              │
             └─────┬───────┴──────────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │ Log Interaction      │
        │ (JSONL + SQLite)     │
        └──────┬───────────────┘
               │
               ▼
        ┌──────────────────────┐
        │ Return to User       │
        │ Save Session Memory  │
        └──────────────────────┘
```

---

## Data Structures

### Query Classification States
```python
route = one of:
    "pop_lookup"       # Direct state/crop lookup (FASTEST)
    "dataset_retrieval" # Semantic search on POP Bank
    "summarization"    # Use specialized model (phi4-mini)
    "reasoning"        # Use reasoning model (qwen)
    "general"          # Default LLM path
```

### Interaction Log Entry
```json
{
  "timestamp": "ISO 8601",           # When query was processed
  "question": "User question text",
  "route": "Route classification",
  "model": "Model used (or 'POP Bank direct')",
  "response_time": 0.0,              # Seconds
  "accuracy": 0.0                    # 0.0 to 1.0 estimate
}
```

### Memory Storage
```
SQLite: reports/chat_memory.sqlite3
  └─ Table: messages
     ├─ id (primary key)
     ├─ role (user / assistant)
     ├─ content (message text)
     └─ created_at (timestamp)
```

### POP Bank Index
```json
{
  "Andra Pradesh": {
    "Blackgram Varieties": "path/to/file.pdf",
    "Cotton": "path/to/file.pdf",
    ...
  },
  "Arunachal Pradesh": {...},
  ...
}
```

---

## Performance Characteristics

### Latency by Route

| Route | Typical Latency | Example |
|-------|-----------------|---------|
| `pop_lookup` | < 100ms | Direct state query |
| `dataset_retrieval` | 500ms - 2s | Semantic search |
| `reasoning` (LLM) | 2 - 5s | "Why" questions |
| `summarization` (LLM) | 1 - 3s | Summary questions |

### Memory Usage

| Component | Size |
|-----------|------|
| POP Bank index | ~500 KB |
| Cached PDFs | ~50 MB (first load) |
| SQLite chat DB | ~10 KB per 1000 messages |
| Interaction log | ~1 KB per query |

---

## Error Handling

### Common Errors

**Scenario 1: Ollama Not Running**
```
agent.chat("What crops...?")
  └─ LLM call times out
  └─ Catch exception
  └─ Return: "Data not found: Ollama service not ready"
  └─ Log: route, model, error status
```

**Scenario 2: Low Dataset Confidence**
```
dataset search confidence: 0.15 (< 0.3 threshold)
  └─ Skip dataset answer
  └─ Try LLM fallback
  └─ Log: dataset_retrieval attempt with low score
```

**Scenario 3: Malformed Query**
```
user_input = ""
  └─ Tokenize → empty list
  └─ No matches
  └─ Route to LLM with empty context
  └─ LLM responds naturally or with "unclear"
```

---

## Example: Full Query Trace

### Input
```
User: "What crops are grown in Andhra Pradesh?"
Model: phi4-mini:3.8b
Time: 2026-05-19 11:30:47
```

### Processing Steps

1. **Initialization** (100ms)
   - Load POP Bank index
   - Connect SQLite
   - Load conversation history

2. **Add to Memory** (10ms)
   - Write user query to SQLite

3. **Classification** (5ms)
   - Detect "crops", "Andhra Pradesh"
   - Route: `pop_lookup`

4. **Direct Lookup** (20ms)
   - Resolve: "Andhra Pradesh" → "Andra Pradesh"
   - Fetch: state_crops["Andra Pradesh"]
   - Build answer: "40 crop profiles indexed..."

5. **Answer Found** → SKIP remaining steps

6. **Logging** (30ms)
   - Write to memory: SQLite
   - Write to log: JSONL entry
   - Record: accuracy=1.0, time=0.165s

7. **Return** (5ms)
   - Format markdown report
   - Display to user

### Total Time: ~165ms

### Logs Generated

**SQLite entry:**
```sql
INSERT INTO messages (role, content, created_at)
VALUES ('user', 'What crops are grown in Andhra Pradesh?', '2026-05-19T11:30:47');

INSERT INTO messages (role, content, created_at)
VALUES ('assistant', 'Andra Pradesh: 40 crop profiles...', '2026-05-19T11:30:47');
```

**JSONL entry:**
```json
{
  "timestamp": "2026-05-19T11:30:47",
  "question": "What crops are grown in Andhra Pradesh?",
  "route": "pop_lookup",
  "model": "POP Bank direct answer",
  "response_time": 0.165,
  "accuracy": 1.0
}
```

---

## Configuration & Customization

### Key Thresholds (in `pop_bank_loader.py`)

```python
min_confidence = 0.3  # Minimum score to return dataset answer
top_k = 3             # Number of top results to consider
```

### System Prompt (in `config.py`)

```python
SYSTEM_PROMPT = (
    "You are a direct, careful agricultural AI assistant. "
    "Answer from your own reasoning without retrieving documents..."
)
```

### Approved Models (in `config.py`)

```python
AVAILABLE_MODELS = [
    "phi4-mini:3.8b",      # Default
    "qwen3:4b",
    "qwen2.5:1.5b",
    "llama3:latest",
    "lfm2.5:1.2b",
]
```

---

## Summary

The AI Agent processes queries through a **smart pipeline** that:

1. ✅ **Classifies** intent quickly
2. ✅ **Prioritizes dataset** over LLM for reproducibility
3. ✅ **Uses confidence thresholds** to gate answers
4. ✅ **Falls back gracefully** to LLM when needed
5. ✅ **Logs everything** for audit & evaluation
6. ✅ **Persists memory** across sessions
7. ✅ **Formats output** for human consumption

**Result:** A reproducible, auditable, dataset-backed expert-assistance system suitable for AJRASAKSHA review workflows.

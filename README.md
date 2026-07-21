# my--framework-v1-1

A personal agent framework project for learning, comparison, and long-term reuse.

## Overview

This repository is used to study existing agent frameworks and gradually build a reusable framework of my own.

The current version focuses on the basic framework building blocks:
- a `core` layer for agent config, messages, and LLM access
- a `memory` layer for conversation history
- a reusable `runtime` layer for bounded model/tool execution and run traces
- a lightweight tool registry for local tools and MCP-style adapters
- a `my_tools` directory for concrete local tools such as `bash`
- a runnable example entry point in `main.py`
- a split configuration model:
  - `.env` for secrets and connection settings
  - `config.yaml` for runtime behavior

## Runtime Model

The framework currently uses a single protocol across all model backends:
- chat requests follow the OpenAI-compatible message format
- tool calling follows the OpenAI-compatible function calling format

`provider` is still kept in config, but it only labels the backend source:
- cloud API such as `openai` or `deepseek`
- local model gateway such as `ollama` or `lmstudio`

In other words, this framework does not try to unify multiple vendor-native protocols. It standardizes on one OpenAI-compatible protocol and routes it to different backends.

## Current Capabilities

- reusable single-agent interaction runtime
- bounded multi-step model/tool execution with typed stop reasons
- in-memory run traces for reconstructing model and tool events
- normalized model responses with per-call and per-run token usage
- safe model parameter snapshots in model-call trace events
- conversation history memory
- native OpenAI-compatible tool calling
- tool registration and argument validation
- local tools and MCP-style tool adapters
- concrete local tools kept under `my_tools/`
- runnable example entry in `main.py`

## Project Structure

```text
.
|- core/
|  |- agent.py
|  |- config.py
|  |- llm.py
|  `- message.py
|- memory/
|  |- base.py
|  `- conversation.py
|- runtime/
|  |- events.py
|  |- runtime.py
|  `- state.py
|- my_tools/
|  `- bash.py
|- rag/
|  |- embedding.py
|  |- graph_store.py
|  |- retriever.py
|  `- vector_store.py
|- tools/
|  |- base.py
|  |- mcp.py
|  `- registry.py
|- config.yaml
|- main.py
`- .env
```

## Configuration

### `.env`

Use `.env` for secrets and endpoint settings:

```env
LLM_API_KEY=YOUR-API-KEY
LLM_BASE_URL=YOUR-OPENAI-COMPATIBLE-ENDPOINT
```

### `config.yaml`

Use `config.yaml` for runtime behavior:

```yaml
llm:
  provider: deepseek
  model: deepseek-v4-flash
  temperature: 0.7
  max_tokens: 2048
  timeout: 30

agent:
  max_history_length: 20
  max_tool_steps: 3
  debug: false
```

`provider` indicates where the model comes from. The request format is still OpenAI-compatible regardless of whether the backend is local or cloud-hosted.

## Run

Install the required dependencies in your environment, then run:

```bash
python main.py
```

The current entry script will:
- create an `OpenAICompatibleLLM`
- register a local `bash` tool from `my_tools/`
- create an `Agent` strategy
- execute it through the reusable `Runtime`
- run one example request through the multi-step interaction loop

## Tool Calling

Tool orchestration is implemented by `runtime.Runtime`. Its current loop is:

1. send messages and tool schemas to the model
2. parse tool calls from the response
3. execute requested tools locally
4. append tool results back into the conversation
5. repeat until the model returns a final answer or the configured step limit is reached

`tools/` contains the framework-side tool abstractions, tool registry, and MCP adapter.
Concrete local tools live in `my_tools/`. Runtime traces currently use an in-memory
recorder; external observability exporters are planned for v2.

## Current Limitations

- checkpointing and resume are not implemented
- external trace export, structured metrics, and retry policies are not implemented
- cancellation is cooperative between blocking model and tool operations
- tool implementations are responsible for enforcing their own execution timeouts
- the framework assumes an OpenAI-compatible API shape
- provider differences beyond that protocol are not abstracted
- streaming is not implemented yet
- the RAG modules exist but are not wired into the main runtime
- vector store, graph store, and embedding backends are still placeholders

## Design Direction

The framework should be centered on a reusable execution runtime rather than a large
agent base class:

- `Agent` defines strategy: model, instructions, available tools, permissions, and stop conditions.
- `Runtime` owns execution: model calls, tool loops, run lifecycle, limits, cancellation, and errors.
- `Context` builds the working message set for each model call from history, memory, retrieval, and runtime state.
- `ToolRegistry` is the execution boundary for schema validation, permissions, timeouts, and audit data.
- `Trace` records events so a run's execution order and outcomes can be inspected and evaluated.

MCP and model providers remain adapters around stable internal contracts. Multi-agent
behavior should later be implemented as delegation on top of the same runtime, not as
a separate execution foundation.

## Roadmap

The roadmap is ordered by dependency: first make execution correct, then make it
observable and resilient, then manage context and memory, and only then add advanced
execution strategies. Later stages should extend the same runtime contracts rather
than introduce separate agent loops.

### v1: Reusable Runtime

Status: complete.

Goal: move the proven loop out of `main.py` and make sequential model/tool execution
correct, bounded, and reusable.

- introduce explicit run state, lifecycle events, result, executor, and trace types under `runtime/`
- give every run and event stable identifiers, parent relationships, sequence numbers,
  timestamps, status, attempt, and typed error fields
- define a backend-neutral `EventSink` interface with an in-memory trace recorder as the
  default implementation
- execute repeated model/tool steps up to `max_tool_steps`
- persist user, assistant, tool-call, tool-result, and final assistant messages
- normalize model and tool failures and define explicit stop reasons
- support cooperative cancellation between blocking operations and tool-defined timeouts
- keep concrete agents focused on strategy and configuration
- add focused tests for successful runs, tool failures, malformed arguments, step limits,
  cancellation, missing usage, conversation continuity, and event sink isolation

Exit criteria: two different agent definitions can use the same runtime without
reimplementing the model/tool loop, and the ordered model calls, tool calls, outcomes, and
failures of a completed run can be inspected from its event trace.

### v2: Reliability, Transport, and Observability

Goal: make model communication resilient and expose enough runtime data to operate,
debug, and budget real applications.

- provide synchronous and asynchronous model interfaces with a shared response contract
- expose streaming as runtime events so terminals and web applications share one path
- add classified retries for rate limits, timeouts, and transient server failures, with
  exponential backoff, jitter, and `Retry-After` support
- add structured JSON logging and token, latency, tool, error, and cost metrics
- add optional exporters for local files and external observability backends such as
  OpenTelemetry or LangFuse without coupling the runtime to their SDKs
- export telemetry asynchronously through bounded buffers so a slow or unavailable
  observability backend cannot block or fail an agent run
- define payload allowlists, redaction, truncation, and secret/PII handling for prompts,
  tool arguments, results, and errors
- add configurable sampling that preserves all failed runs, plus retention and deletion
  policies for stored traces
- keep metric labels low-cardinality; store user queries, run identifiers, and detailed
  errors in traces rather than metric dimensions
- extend normalized usage with cost accounting, budget enforcement, and provider-specific metrics
- define token-counter and budget interfaces, context-window metadata, output reservation,
  and configurable run limits without implementing automatic summarization yet
- add checkpoint and resume support for long-running tasks
- add approval and permission policies for tools with side effects
- support strict structured outputs where available, backed by local schema validation and
  bounded repair retries rather than assuming provider output is always semantically valid
- correlate traces with prompt, model, configuration, and application versions so evaluation
  results can be compared across changes
- build replay-based regression tests and task-level evaluations

Exit criteria: transient failures are handled according to an explicit policy, streamed
and non-streamed runs produce equivalent traces, actual usage is measurable, sensitive
tool calls can require approval, sensitive telemetry payloads are controlled, exporter
failures do not affect run results, and representative tasks can be replayed and evaluated.

### v3: Context, Memory, and Retrieval

Goal: keep long-running conversations and tasks within a controlled token budget while
preserving the information most relevant to the current step.

- build a context pipeline for selection, deduplication, tool-result filtering, trimming,
  and summarization
- implement provider counting, model tokenizer, and conservative fallback strategies behind
  the token-counter interface introduced in v2
- trigger compaction with configurable high and low watermarks while reserving output tokens
- preserve system instructions, recent turns, current task state, and complete tool-call/result
  groups when compacting history
- distinguish working state, session history, summaries, retrieved knowledge, and durable
  long-term memory
- expose a structured run notepad for goals, plan status, findings, and important data; keep
  the runtime state authoritative and inject only a compact model-facing projection
- connect the existing RAG interfaces and add evaluated hybrid retrieval, using score
  normalization or rank fusion instead of fixed unvalidated weights
- support metadata, retention, provenance, and deletion policies for durable memories

Exit criteria: long runs remain inside their configured context budget, compaction does not
break tool-call integrity, retrieved memories have traceable provenance, and context policies
can be changed without modifying the runtime execution loop.

### v4: Advanced Execution and Composition

Goal: improve throughput and reasoning flexibility after execution, safety, and context
management have stable contracts.

- execute independent tool calls concurrently with dependency checks, concurrency limits,
  deterministic result ordering, cancellation, and side-effect policies
- add a sandboxed CodeAct executor with explicit filesystem, process, network, credential,
  time, and resource boundaries
- route deterministic side effects through function calling and reserve CodeAct for tasks
  that genuinely benefit from code execution
- add sub-agent delegation, workflows, and durable human-in-the-loop tasks on top of the same
  runtime and trace model
- treat multi-candidate generation and self-consistency as opt-in strategies for evaluation
  or high-value tasks because all generated candidates consume tokens
- evaluate quality, latency, and cost before enabling advanced strategies by default

Exit criteria: concurrent execution is demonstrably faster without changing task semantics,
CodeAct cannot escape its declared permissions, and composition features reuse the existing
runtime, context, budget, and trace contracts.

## Non-Goals for v1

- supporting every vendor-native model protocol
- making multi-agent orchestration a core primitive
- adding RAG before context construction has a stable interface
- hiding execution behavior behind a large all-purpose `Agent` abstraction

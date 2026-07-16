# my--framework-v1-1

A personal agent framework project for learning, comparison, and long-term reuse.

## Overview

This repository is used to study existing agent frameworks and gradually build a reusable framework of my own.

The current version focuses on a small but explicit runtime:
- a `core` layer for config, memory, messages, and llm access
- a `CodeAgent` that uses native OpenAI-compatible tool calling
- a lightweight tool registry for local tools and MCP-style adapters
- a `my_tools` directory for concrete local tools such as `bash`
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

- single-agent interaction loop
- conversation history memory
- native OpenAI-compatible tool calling
- tool registration and argument validation
- local tools and MCP-style tool adapters
- concrete local tools kept under `my_tools/`
- interactive run entry in `main.py`

## Project Structure

```text
.
|- agent/
|  |- codeagent.py
|- core/
|  |- agent.py
|  |- config.py
|  |- llm.py
|  `- message.py
|- memory/
|  |- base.py
|  `- conversation.py
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
- register a local time tool
- register a local `bash` tool from `my_tools/`
- register a demo MCP-style echo tool
- create a `CodeAgent`
- enter interactive mode

## Tool Calling

`CodeAgent` uses provider-native tool calling through the OpenAI-compatible `tools` field.

The runtime loop is:
1. send messages and tool schemas to the model
2. parse tool calls from the response
3. execute requested tools locally
4. append tool results back into the conversation
5. continue until the model returns a final answer

`tools/` only contains framework-side abstractions, the tool registry, and the MCP adapter. Concrete local tools live in `my_tools/`.

## Current Limitations

- the framework assumes an OpenAI-compatible API shape
- provider differences beyond that protocol are not abstracted
- streaming is not implemented yet
- the RAG modules exist but are not wired into the main runtime
- vector store, graph store, and embedding backends are still placeholders

## Next Steps

- clean up config and runtime naming around backend metadata
- add streaming support
- add richer tool execution controls and logging
- wire RAG into the main agent runtime

# my--framework-v1-1

A personal agent framework project for learning, comparison, and long-term reuse.

## Goal

This project is used to study the strengths of existing agent frameworks and gradually build a reusable framework of my own.

The focus is not only to make an agent run, but to design a clean architecture that can evolve over time.

## Current Status

This repository is at an early stage.

Planned first steps:
- define the core agent abstraction
- separate model interaction from agent logic
- design a simple tool calling interface
- keep the first version minimal and easy to iterate

## Design Principles

- minimal first, extend later
- clear module boundaries
- low coupling between core logic and model providers
- easy to test, debug, and refactor
- built for learning first, reuse second

## Roadmap

### Phase 1
- single-agent loop
- basic message abstraction
- model client interface
- simple tool registry

### Phase 2
- memory abstraction
- prompt management
- logging and tracing
- retries and error handling

### Phase 3
- multi-agent collaboration
- workflow orchestration
- persistence and evaluation

## Notes

This project is intentionally built step by step.
Architecture decisions will be compared against ideas from other frameworks during development.
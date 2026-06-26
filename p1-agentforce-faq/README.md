# P1 — Agentforce FAQ Bot

## What I built
A multi-agent Salesforce Agentforce system for TaskFlow customer 
support with specialised subagents for Features, Pricing, and Account 
Help questions.

## Architecture
## What I learned

### 1. Multi-agent design patterns
- Specialist agents outperform single generalist agents
- Router → specialist delegation is fundamental
- Escalation and fallback handlers are non-optional for enterprise

### 2. RAG in practice
Saw firsthand how Agentforce implements RAG:
- Knowledge Articles = grounding data
- Data Categories = metadata for retrieval scoping
- Citations Enabled = explainability layer
- This is exactly what I'll build manually in P3 with Python

### 3. Real enterprise constraints
- Data Cloud licensing gates advanced retrieval features
- Permission models apply to AI agents like human users
- Platform evolution (UI changes frequently) creates instability
- This taught me to value stable, portable fundamentals

### 4. Governance at the retrieval layer
- Data Categories control what agents retrieve (not just what they do with it)
- Citations make retrieval transparent
- Honesty guardrails (say "I don't know") prevent hallucination
- These patterns will deepen in P9

## If I were doing this for a real company, I'd:
- Implement Data Cloud for semantic search at scale (>1000 articles)
- Build custom metadata fields for article-level LLM guidance
- Create audit logging on retrieval and response quality
- Design feedback loops to improve category tagging over time

## Architectural decision made
Chose standard Knowledge search + Data Categories over Data Cloud 
Knowledge Space for learning scope. At enterprise scale (thousands of 
articles), Data Cloud's vector search is the correct approach.
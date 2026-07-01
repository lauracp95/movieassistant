---
description: Review a pull request for correctness, test coverage, architectural alignment, and documentation completeness.
alwaysApply: false
---

# PR Review Skill

## When to Use

- User asks to review a pull request or a branch diff
- User says "review this PR", "is this ready to merge?", "check this branch"
- User shares a GitHub PR URL or asks to prepare changes for review
- Before merging a feature branch into main

## Context to Inspect First

1. **Full diff from base branch**: `git diff main...HEAD` (or the target branch)
2. **Commit history**: `git log --oneline main..HEAD` — understand the progression
3. **Test results**: Run the test suite to confirm green
4. **Modified areas**: Identify which subsystems are touched (agents, workflow, RAG, guardrails, schemas, routes)
5. **README/docs**: Check if documentation was updated for user-facing changes

## Workflow

### Step 1: Understand the PR scope

```bash
git log --oneline main..HEAD
git diff --stat main...HEAD
```

Categorize the change:
- **Feature**: New capability or agent
- **Enhancement**: Improvement to existing behavior
- **Bug fix**: Corrects broken behavior
- **Refactor**: Restructures without changing behavior
- **Docs**: Documentation only

### Step 2: Review the diff systematically

Read the full diff:

```bash
git diff main...HEAD
```

Review in this order:
1. **Schemas** (`api/app/schemas/`) — Data model changes affect everything downstream
2. **State** (`api/app/workflow/state.py`) — New fields must be initialized in `create_initial_state()`
3. **Agents** (`api/app/agents/`) — Core logic changes
4. **Workflow** (`api/app/workflow/`) — Graph structure, nodes, routing
5. **RAG** (`api/app/rag/`) — Retrieval pipeline changes
6. **Routes** (`api/app/routers/`) — API contract changes
7. **Tests** (`api/test/`) — Coverage for all the above
8. **Docs** — README, knowledge base, `.env.example`

### Step 3: Check for common issues

#### Breaking changes
- [ ] API response shape changed? (breaks UI)
- [ ] New required env var without default? (breaks deployment)
- [ ] `MovieNightState` field added but not in `create_initial_state()`?
- [ ] Import paths changed? (breaks other modules)

#### Test quality
- [ ] New code has corresponding tests
- [ ] Tests mock LLM calls (never hit Azure OpenAI)
- [ ] Tests cover happy path AND error paths
- [ ] No tests depend on execution order

#### Prompt changes
- [ ] Prompt modifications in `api/app/llm/prompts.py` are intentional
- [ ] Prompt changes don't break structured output parsing
- [ ] Evaluator prompt changes don't inadvertently lower/raise the quality bar

#### Workflow integrity
- [ ] All nodes registered in `graph_builder.py`
- [ ] All conditional edges have complete branch mappings
- [ ] Retry loop still functions (evaluate → write_recommendation → evaluate)
- [ ] `MAX_RETRIES` and `PASS_THRESHOLD` not changed without justification

#### RAG consistency
- [ ] Knowledge base files updated if behavior changed
- [ ] ChromaDB ingestion still works (no broken markdown)
- [ ] Retriever parameters (`top_k`, `min_score`) not changed without reason

### Step 4: Run the test suite

```bash
cd api && uv run pytest -v
```

All tests must pass. If any fail, report them as blockers.

### Step 5: Produce the review

Structure the review as:

```
## PR Review: {brief description}

### Summary
{1-2 sentences on what this PR does}

### Scope
- Files changed: {count}
- Subsystems touched: {list}

### ✓ Looks Good
- {things done well}

### ✗ Blockers (must fix before merge)
- {critical issues}

### △ Suggestions (non-blocking)
- {optional improvements}

### Checklist
- [ ] Tests pass
- [ ] No breaking API changes (or UI updated)
- [ ] Docs updated if user-facing
- [ ] No secrets in diff
- [ ] Knowledge base current
```

## Expected Output

A structured review comment with clear blockers vs. suggestions, ready to post on the PR or use for self-review.

## Checks Before Reviewing

- [ ] On the correct branch
- [ ] Understand what the PR is supposed to accomplish
- [ ] Have the full diff (not just latest commit — check ALL commits)

## Checks After Reviewing

- [ ] All blockers addressed before approving
- [ ] Tests pass: `cd api && uv run pytest -v`
- [ ] Re-review any fixes made in response to feedback

## Commands

| Action | Command |
|--------|---------|
| See full branch diff | `git diff main...HEAD` |
| Commit history | `git log --oneline main..HEAD` |
| Files changed | `git diff --stat main...HEAD` |
| Run tests | `cd api && uv run pytest -v` |
| Check for secrets | Search diff for key/secret/password/token patterns |

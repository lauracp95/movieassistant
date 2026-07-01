---
description: Self-review completed code changes against project conventions and quality standards before committing.
alwaysApply: false
---

# Code Review Skill

## When to Use

- After completing a feature or bug fix implementation
- User says "review this", "check my code", "is this ready?"
- Before creating a commit or PR
- When verifying that changes follow project patterns

## Context to Inspect First

1. **Changed files**: Run `git diff` and `git status` to see what was modified
2. **Related test files**: Verify tests exist for modified modules
3. **Project conventions**: Check neighboring files in the same package for style consistency
4. **Import structure**: Confirm imports follow `from app.{package}.{module} import {Symbol}` pattern
5. **Schema definitions**: If domain models changed, check `api/app/schemas/`

## Workflow

### Step 1: Identify the scope of changes

```bash
git status
git diff --stat
```

List all modified, added, and deleted files.

### Step 2: Review each file against standards

For each changed file, check:

#### Python Style
- [ ] Python 3.14 syntax used: `var` keyword not applicable, but use `|` union types, pattern matching where appropriate
- [ ] Type hints on all function signatures (args and return)
- [ ] No code comments that just narrate what the code does
- [ ] Descriptive variable and function names
- [ ] Functions are focused (single responsibility)

#### Architecture
- [ ] Follows Controller → Service → Repository pattern (routes → agents → integrations)
- [ ] New agents follow the existing pattern: constructor takes LLM, methods are synchronous
- [ ] No direct LLM calls outside of agent classes
- [ ] State mutations happen only in workflow nodes, not in agents
- [ ] Agents receive data as parameters, not by reading global state

#### Imports and Dependencies
- [ ] No circular imports
- [ ] Uses `from __future__ import annotations` for forward references
- [ ] `TYPE_CHECKING` block used for type-only imports
- [ ] No new external dependencies added without updating `pyproject.toml`

#### Security
- [ ] No hardcoded API keys, secrets, or credentials
- [ ] No `.env` files committed
- [ ] Guardrail patterns not weakened

#### LangGraph Workflow
- [ ] `MovieNightState` fields added properly (with `total=False`)
- [ ] New nodes registered in `graph_builder.py`
- [ ] Conditional edges have all branches mapped
- [ ] Retry loop logic preserved (evaluate → write_recommendation cycle)

### Step 3: Check test coverage

- For each modified source file in `api/app/`, verify a corresponding test file exists in `api/test/`
- If new public methods were added, verify they have test cases
- If behavior changed, verify existing tests were updated

### Step 4: Run tests

```bash
cd api && uv run pytest -v
```

All tests must pass.

### Step 5: Report findings

Produce a summary with:
- **Pass**: Things that look good
- **Issues**: Problems that must be fixed before committing
- **Suggestions**: Optional improvements (style, naming, minor refactors)

## Expected Output

A structured review report:

```
## Review Summary

### ✓ Passes
- [list of things that are correct]

### ✗ Issues (must fix)
- [file:line] Description of the problem

### △ Suggestions (optional)
- [file:line] Suggestion for improvement
```

## Checks Before Reviewing

- [ ] All files saved
- [ ] Understand the intent of the change (read PR description or commit message)
- [ ] Read the diff, not just the final file state

## Checks After Reviewing

- [ ] All "must fix" issues resolved
- [ ] Tests pass after fixes: `cd api && uv run pytest -v`
- [ ] No untracked files left behind accidentally

## Commands

| Action | Command |
|--------|---------|
| See changes | `git diff` |
| See staged changes | `git diff --cached` |
| Run tests | `cd api && uv run pytest -v` |
| Check for secrets | `git diff --cached \| grep -i "key\|secret\|password\|token"` |

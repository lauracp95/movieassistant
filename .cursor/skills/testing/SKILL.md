---
description: Write, run, and fix pytest tests for the Movie Night Assistant API following project conventions.
alwaysApply: false
---

# Testing Skill

## When to Use

- User asks to write tests, add test coverage, or test a feature
- User creates or modifies application code and wants corresponding tests
- User says "test this", "add tests", "cover this with tests"
- User modifies files in `api/app/` and wants to verify correctness

## Context to Inspect First

1. **Existing test structure**: `api/test/` mirrors `api/app/` (e.g., `api/app/agents/` → `api/test/agents/`)
2. **Shared fixtures**: `api/test/conftest.py` — provides `mock_input_agent`, `mock_movie_finder`, `in_memory_movie_finder`, `recommendation_writer`, `evaluator`, `rag_agent`, `mock_rag_retriever`, and `make_movie()` factory
3. **The module under test**: Read the source file to understand its interface, dependencies, and edge cases
4. **Existing tests for that module**: Check if a test file already exists and follow its patterns
5. **pytest config**: `api/pyproject.toml` → `[tool.pytest.ini_options]` (testpaths = `["test"]`, addopts = `["--import-mode=prepend"]`)

## Workflow

### Step 1: Identify what to test

- Read the source module to understand its public API
- List functions/methods/classes that need coverage
- Identify edge cases: empty inputs, None values, error paths, boundary conditions

### Step 2: Determine test file location

- Mirror the source path: `api/app/{package}/{module}.py` → `api/test/{package}/test_{module}.py`
- If the test file exists, add tests to it; do not create a duplicate

### Step 3: Write tests following project conventions

- Import path style: `from app.{package}.{module} import {Class}`
- Use `unittest.mock.MagicMock` for LLM dependencies (never call Azure OpenAI)
- Use `InMemoryMovieFinderAgent` for movie data (never call TMDB)
- Use fixtures from `conftest.py` when available
- Use `make_movie()` from `conftest.py` to create test `MovieResult` objects
- Name tests descriptively: `test_{method}_{scenario}_{expected_outcome}`
- Keep tests focused: one assertion per logical concept
- No `@pytest.mark.asyncio` — all code is synchronous

### Step 4: Run tests

```bash
cd api && uv run pytest {test_file_path} -v
```

If running all tests:

```bash
cd api && uv run pytest -v
```

On Windows with Git Bash:

```bash
cd api && uv run python -m pytest {test_file_path} -v
```

### Step 5: Fix failures

- Read the full error output
- Distinguish between:
  - **Import errors**: Wrong path or missing dependency
  - **Assertion errors**: Logic bug in test or source
  - **Mock errors**: Incorrect mock setup (wrong spec, missing return_value)
  - **Type errors**: Wrong argument types or missing fields
- Fix the test (or propose a source fix if the test reveals a bug)
- Re-run only the failing test to confirm the fix

### Step 6: Verify

- Run the full test file once all individual tests pass
- Ensure no test bleeds state into another (no shared mutable fixtures)

## Test Patterns Reference

### Mocking an LLM agent

```python
from unittest.mock import MagicMock
from langchain_openai import AzureChatOpenAI

llm = MagicMock(spec=AzureChatOpenAI)
llm.invoke.return_value = MagicMock(content="response text")
agent = SomeAgent(llm)
```

### Mocking structured output (evaluator)

```python
structured = MagicMock()
structured.invoke.return_value = EvaluationResult(
    passed=True, score=0.85, feedback="Good.",
    constraint_violations=[], improvement_suggestions=[],
)
llm.with_structured_output.return_value = structured
```

### Creating test movies

```python
from test.conftest import make_movie

movie = make_movie("1", "Test Movie", genres=["sci-fi"], runtime_minutes=110, year=2020)
```

### Testing workflow nodes

```python
from app.workflow.state import MovieNightState

state: MovieNightState = {
    "user_message": "test message",
    "route": None,
    "constraints": None,
    "search_query": None,
    "needs_recommendation": False,
    "rag_query": None,
    "candidate_movies": [],
    "retrieved_contexts": [],
    "draft_recommendation": None,
    "evaluation_result": None,
    "retry_count": 0,
    "rejected_titles": [],
    "final_response": None,
    "error": None,
}
```

## Expected Output

- One or more test functions in the correct test file
- All tests passing when run with `uv run pytest`
- If a test reveals a source bug, report it clearly with the fix

## Checks Before Writing Tests

- [ ] Read the source module being tested
- [ ] Check if a test file already exists for this module
- [ ] Verify imports work (correct path structure)
- [ ] Identify which fixtures from `conftest.py` are reusable

## Checks After Writing Tests

- [ ] Run the specific test file: `cd api && uv run pytest {path} -v`
- [ ] All tests pass (green)
- [ ] No warnings about unresolved mocks or deprecations
- [ ] If failures occur, fix and re-run until green

## Commands

| Action | Command |
|--------|---------|
| Run specific test file | `cd api && uv run pytest test/{path}/test_{name}.py -v` |
| Run single test | `cd api && uv run pytest test/{path}/test_{name}.py::test_function -v` |
| Run all tests | `cd api && uv run pytest -v` |
| Run with output | `cd api && uv run pytest -v -s` |

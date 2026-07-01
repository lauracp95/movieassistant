---
description: Update project documentation (README, knowledge base, env vars) when features or architecture change.
alwaysApply: false
---

# Documentation Updates Skill

## When to Use

- After adding or modifying a feature that affects the public API or architecture
- When environment variables are added or changed
- When the project structure changes (new modules, renamed files)
- When workflow behavior changes (new nodes, routes, state fields)
- User says "update docs", "update README", "document this"
- When knowledge base content becomes stale after code changes

## Context to Inspect First

1. **README.md** (root): Architecture, env vars table, setup instructions, project structure tree, API endpoint examples
2. **Knowledge base files**: `api/app/knowledge_base/*.md` — these are ingested into ChromaDB for RAG responses
3. **The change being documented**: Read the modified source files to understand what changed
4. **`.env.example`**: Must match the env vars table in README
5. **`docker-compose.yml`**: Environment variables listed here should match `.env.example`

## Workflow

### Step 1: Identify what documentation is affected

Map the code change to documentation sections:

| Code Change | Docs to Update |
|-------------|----------------|
| New/changed env var | README env vars table + `.env.example` |
| New agent | README Architecture + Project Structure + knowledge base |
| New API endpoint | README API Endpoints section |
| New workflow node/route | README "How It Works" + `routing_logic.md` |
| Changed evaluation logic | `evaluation_logic.md` |
| Changed data sources | `data_sources.md` |
| New limitation or removed limitation | README "Current Limitations" + `known_limitations.md` |
| Changed recommendation rules | `recommendation_rules.md` |
| New file/directory | README "Project Structure" tree |
| Changed setup steps | README "Run Locally" or "Run with Docker" |

### Step 2: Update README.md

When editing the README:
- Keep the existing tone (technical but approachable)
- Update the project structure tree to reflect actual file layout
- Keep the env vars table sorted: required first, then optional
- API examples should show real request/response shapes
- Architecture bullet points should be concise (one line each)

### Step 3: Update knowledge base (if affected)

Knowledge base files in `api/app/knowledge_base/` are the source of truth for RAG answers. When updating:

- **`system_overview.md`**: High-level architecture description
- **`recommendation_rules.md`**: How recommendations are generated and grounded
- **`evaluation_logic.md`**: Evaluator behavior, thresholds, retry logic
- **`data_sources.md`**: Where movie data comes from (TMDB, in-memory)
- **`routing_logic.md`**: How input orchestrator classifies routes
- **`known_limitations.md`**: Current system limitations

Rules for knowledge base edits:
- Write in plain markdown (no code blocks unless explaining config)
- Keep content factual and current — these documents generate user-facing RAG answers
- Each file should be self-contained (a retrieval chunk may come from any file)
- Keep paragraphs short (chunking splits at ~500 chars)

### Step 4: Update `.env.example` (if env vars changed)

- Add new variables with a descriptive comment
- Use placeholder values (never real keys)
- Mark required vs optional clearly

### Step 5: Verify consistency

Cross-check that:
- README env vars table matches `.env.example`
- README project structure matches actual `ls` output
- Knowledge base content matches current code behavior
- Docker Compose environment list covers all required vars

## Expected Output

- Updated markdown files with accurate, current content
- No stale references to removed features
- Knowledge base reflects the actual system behavior

## Checks Before Editing

- [ ] Read the current state of the doc being updated
- [ ] Understand what code change triggered the update
- [ ] Check if multiple docs need updating (often README + knowledge base)

## Checks After Editing

- [ ] README renders correctly (no broken markdown)
- [ ] Env vars table has consistent column alignment
- [ ] Project structure tree matches reality
- [ ] Knowledge base files are factual (no aspirational features)
- [ ] `.env.example` has no real secrets
- [ ] Run tests to ensure nothing broke: `cd api && uv run pytest -v`

## Commands

| Action | Command |
|--------|---------|
| Verify project structure | `ls -R api/app/` |
| Check env vars in code | Search `settings.py` for all `Field(...)` definitions |
| Run tests after doc changes | `cd api && uv run pytest -v` |

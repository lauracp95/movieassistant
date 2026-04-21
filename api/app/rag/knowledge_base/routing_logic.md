# Routing Logic

## Route Types

The InputOrchestratorAgent classifies every user message into one of four routes:

### 1. `movies` Route
**Purpose**: Pure movie recommendation requests

**Triggers**:
- Requests for movie suggestions
- Genre-specific queries
- Mood-based movie requests
- Constraint-based searches (runtime, etc.)

**Examples**:
- "Recommend a good horror movie"
- "What should I watch tonight?"
- "I want something funny and under 2 hours"
- "Give me a sci-fi movie from the 90s"

**Flow**: 
`orchestrate` → `find_movies` → `write_recommendation` → `evaluate` → `respond`

### 2. `rag` Route
**Purpose**: Questions about the system itself

**Triggers**:
- Questions about how the app works
- Inquiries about capabilities
- Data source questions
- Privacy or limitation questions

**Examples**:
- "How does this app work?"
- "Where do you get movie data?"
- "What features does this assistant have?"
- "Are my messages stored?"

**Flow**: 
`orchestrate` → `rag_retrieve` → `rag_respond` → `respond`

### 3. `hybrid` Route
**Purpose**: Requests needing both movie data AND system knowledge

**Triggers**:
- Movie requests with contextual questions
- Recommendations requiring explanation of methodology
- Comparative or educational movie queries

**Examples**:
- "Recommend movies for date night and explain why they're romantic"
- "Find me a thriller and tell me what makes a good thriller"
- "What horror movies work for Halloween and why?"

**Flow**: 
`orchestrate` → `find_movies` → `rag_retrieve` → `write_recommendation` → `evaluate` → `respond`

### 4. `clarification` Route
**Purpose**: Handle ambiguous or unclear requests

**Triggers**:
- Message too vague to classify
- Ambiguous intent
- Incomplete requests

**Examples**:
- "help"
- "hi"
- "?"

**Flow**: 
`orchestrate` → `END` (response already set with clarification question)

## Route Selection Logic

The InputOrchestratorAgent uses these heuristics:

1. **Default to `movies`** for straightforward recommendation requests
2. **Choose `rag`** when questions focus on the system, not content
3. **Choose `hybrid`** only when BOTH movie data AND explanation are needed
4. **Choose `clarification`** when intent cannot be determined

## Constraint Extraction

For `movies` and `hybrid` routes, constraints are extracted:

### Genre Normalization
| User Says | Normalized Genre |
|-----------|-----------------|
| "scary", "horror" | horror |
| "funny", "comedy" | comedy |
| "sci-fi", "science fiction" | sci-fi |
| "romantic", "romance" | romance |
| "suspense", "thriller" | thriller |

### Runtime Parsing
| User Says | Constraint |
|-----------|-----------|
| "under 2 hours" | max_runtime_minutes: 120 |
| "short movie", "quick watch" | max_runtime_minutes: 90 |
| "long movie", "over 2 hours" | min_runtime_minutes: 120 |

## RAG Query Generation

For `rag` and `hybrid` routes, a RAG query is generated:
- Transforms user question into retrieval-friendly format
- Focuses on the knowledge aspect of the question
- Used to search internal documentation

## Post-Routing Behavior

### Movies/Hybrid Path
1. MovieFinderAgent retrieves candidates from TMDB
2. RecommendationWriterAgent selects best match and writes text
3. EvaluatorAgent validates quality
4. Retry loop if evaluation fails (up to 3 times)

### RAG Path
1. Retriever searches knowledge base documents
2. Relevant chunks are added to context
3. RAGAssistantAgent generates grounded answer
4. Response formatted and returned

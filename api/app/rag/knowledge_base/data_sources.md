# Data Sources

## Primary Movie Data: TMDB

### What is TMDB?
The Movie Database (TMDB) is a community-built movie and TV database. It provides:
- Comprehensive movie metadata
- User-contributed ratings and reviews
- Regularly updated content

### Data Retrieved from TMDB

For each movie, the system retrieves:

| Field | Description |
|-------|-------------|
| `id` | Unique TMDB identifier |
| `title` | Official movie title |
| `year` | Release year |
| `genres` | List of genre names (Action, Comedy, etc.) |
| `runtime_minutes` | Duration in minutes |
| `overview` | Official plot summary |
| `rating` | Average user rating (0-10 scale) |
| `poster_url` | URL to movie poster image |

### Genre Discovery

Movies are discovered through TMDB's genre-based discovery API:
- User's genre preferences map to TMDB genre IDs
- The system queries for popular movies in those genres
- Results are sorted by popularity and rating

### Data Freshness
- TMDB data is queried in real-time for each request
- No local caching of movie data
- Information is as current as TMDB's database

## LLM Provider: Azure OpenAI

### Model Usage
- GPT-4 models power all natural language processing
- Different temperature settings for different tasks:
  - `0.0` for classification (deterministic)
  - `0.3` for recommendation writing (slight creativity)
  - `0.7` for general responses (conversational)

### What LLMs Provide
- Intent classification (movies vs. system questions)
- Constraint extraction from natural language
- Recommendation text generation
- Quality evaluation

### What LLMs Don't Provide
- Movie data (comes from TMDB)
- User memory (stateless)
- Real-time information

## Internal Knowledge Base

### RAG Documents
For system questions, the assistant uses internal documentation:
- System architecture explanations
- Feature descriptions
- Known limitations
- Routing logic details

### Document Retrieval
- Simple semantic search over markdown documents
- Relevant chunks are retrieved based on query similarity
- Answers are grounded in retrieved documentation

## Data Flow Summary

```
User Message
    ↓
InputOrchestratorAgent (LLM)
    ↓
┌─────────────────────────────────────┐
│  Route: movies/hybrid               │
│  → MovieFinderAgent queries TMDB    │
│  → RecommendationWriterAgent (LLM)  │
│  → EvaluatorAgent (LLM)             │
└─────────────────────────────────────┘
┌─────────────────────────────────────┐
│  Route: rag                         │
│  → RAG retrieval from knowledge base│
│  → RAGAssistantAgent (LLM)          │
└─────────────────────────────────────┘
    ↓
Response to User
```

# Movie Night Assistant - System Overview

## What is the Movie Night Assistant?

The Movie Night Assistant is an AI-powered chatbot that helps users discover movies to watch. It combines a conversational interface with real movie data from The Movie Database (TMDB) to provide personalized recommendations.

## Core Architecture

The system is built using:
- **FastAPI**: A modern Python web framework for the REST API
- **LangChain**: For LLM orchestration and structured outputs
- **LangGraph**: For stateful workflow management with retry loops
- **Azure OpenAI**: GPT-4 models for natural language understanding and generation
- **ChromaDB**: Vector database storing embedded knowledge base documents for semantic search
- **TMDB API**: Real-time movie data including titles, genres, ratings, and descriptions

## Request Processing Flow

1. **Guardrail Check**: Before any workflow runs, the GuardrailService inspects the message:
   - Blocks messages that exceed the maximum length limit
   - Blocks messages matching known prompt injection patterns
   - Uses an LLM classifier to block prompt injection attempts and off-topic messages

2. **Input Classification**: When a message passes guardrails, the InputOrchestratorAgent classifies it as:
   - `movies`: Pure movie recommendation requests
   - `rag`: Questions about the system itself
   - `hybrid`: Requests needing both movie data and system knowledge
   - `clarification`: Ambiguous messages that require follow-up before routing

3. **Constraint and Signal Extraction**: For movie-related requests, the system extracts:
   - Genre preferences (comedy, horror, action, etc.)
   - Runtime constraints (minimum/maximum duration)
   - Rich search signals: actors, directors, year or year range, keywords, mood, setting, language

4. **Movie Retrieval**: The MovieFinderAgent queries TMDB to find matching movies based on extracted constraints and search signals.

5. **Recommendation Writing**: The RecommendationWriterAgent crafts a personalized response explaining why a specific movie matches the user's preferences.

6. **Quality Evaluation**: The EvaluatorAgent validates that recommendations are grounded in facts and satisfy user constraints.

7. **Retry Loop**: If a recommendation fails evaluation, the system automatically tries alternative movies up to 3 times.

## Key Components

- **GuardrailService**: Pre-workflow safety layer (length, injection, off-topic)
- **InputOrchestratorAgent**: Classifies user intent, extracts constraints, and generates rich search signals
- **MovieFinderAgent**: Retrieves candidate movies from TMDB (or in-memory catalog)
- **RecommendationWriterAgent**: Generates natural language recommendations
- **EvaluatorAgent**: Validates recommendation quality
- **RAGAssistantAgent**: Answers questions about the system using ChromaDB-backed documentation

## API Endpoints

- `POST /chat`: Main conversation endpoint accepting user messages
- `GET /health`: Health check endpoint for monitoring

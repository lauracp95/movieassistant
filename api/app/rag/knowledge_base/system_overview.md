# Movie Night Assistant - System Overview

## What is the Movie Night Assistant?

The Movie Night Assistant is an AI-powered chatbot that helps users discover movies to watch. It combines a conversational interface with real movie data from The Movie Database (TMDB) to provide personalized recommendations.

## Core Architecture

The system is built using:
- **FastAPI**: A modern Python web framework for the REST API
- **LangChain**: For LLM orchestration and structured outputs
- **LangGraph**: For stateful workflow management with retry loops
- **Azure OpenAI**: GPT-4 models for natural language understanding and generation
- **TMDB API**: Real-time movie data including titles, genres, ratings, and descriptions

## Request Processing Flow

1. **Input Classification**: When a user sends a message, the InputOrchestratorAgent classifies it as:
   - `movies`: Pure movie recommendation requests
   - `rag`: Questions about the system itself
   - `hybrid`: Requests needing both movie data and system knowledge

2. **Constraint Extraction**: For movie-related requests, the system extracts:
   - Genre preferences (comedy, horror, action, etc.)
   - Runtime constraints (minimum/maximum duration)

3. **Movie Retrieval**: The MovieFinderAgent queries TMDB to find matching movies based on extracted constraints.

4. **Recommendation Writing**: The RecommendationWriterAgent crafts a personalized response explaining why a specific movie matches the user's preferences.

5. **Quality Evaluation**: The EvaluatorAgent validates that recommendations are grounded in facts and satisfy user constraints.

6. **Retry Loop**: If a recommendation fails evaluation, the system automatically tries alternative movies up to 3 times.

## Key Components

- **InputOrchestratorAgent**: Classifies user intent and extracts constraints
- **MovieFinderAgent**: Retrieves candidate movies from TMDB
- **RecommendationWriterAgent**: Generates natural language recommendations
- **EvaluatorAgent**: Validates recommendation quality
- **RAGAssistantAgent**: Answers questions about the system using internal documentation

## API Endpoints

- `POST /chat`: Main conversation endpoint accepting user messages
- `GET /health`: Health check endpoint for monitoring

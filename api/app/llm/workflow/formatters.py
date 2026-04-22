"""Response formatting utilities for the Movie Night Assistant workflow.

This module contains functions that format workflow data into user-facing
responses. It is deliberately separated from workflow orchestration to
keep formatting logic isolated and testable.
"""

from app.schemas.domain import MovieResult
from app.schemas.orchestrator import Constraints


def format_candidate_list_response(
    candidates: list[MovieResult],
    constraints: Constraints,
) -> str:
    """Format candidate movies into a simple list response.

    This is a fallback formatter used when no draft recommendation is available.
    The primary path uses RecommendationWriterAgent for richer, grounded prose.

    Args:
        candidates: List of MovieResult objects.
        constraints: User constraints for context.

    Returns:
        Formatted response string.
    """
    if not candidates:
        return "I couldn't find any movies matching your criteria."

    lines = ["Here are some movie recommendations for you:\n"]

    for i, movie in enumerate(candidates[:5], 1):
        year_str = f" ({movie.year})" if movie.year else ""
        genres_str = ", ".join(movie.genres[:3]) if movie.genres else "Unknown genre"
        rating_str = f" - Rating: {movie.rating:.1f}/10" if movie.rating else ""
        runtime_str = f" - {movie.runtime_minutes} min" if movie.runtime_minutes else ""

        lines.append(f"{i}. **{movie.title}**{year_str}")
        lines.append(f"   {genres_str}{runtime_str}{rating_str}")

        if movie.overview:
            overview = (
                movie.overview[:150] + "..."
                if len(movie.overview) > 150
                else movie.overview
            )
            lines.append(f"   {overview}")
        lines.append("")

    return "\n".join(lines)


NO_MOVIES_FOUND_MESSAGE = (
    "I couldn't find any movies matching your criteria. "
    "Try broadening your search or specifying different preferences."
)

RETRY_EXHAUSTED_FALLBACK_MESSAGE = (
    "I tried a few options but couldn't land on a recommendation that met the "
    "quality bar for your request. Could you rephrase or loosen your "
    "preferences a little?"
)

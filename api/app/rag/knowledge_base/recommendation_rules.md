# Recommendation Rules and Quality Standards

## Grounding Principles

All movie recommendations must be **strictly grounded** in actual movie data. The system follows these rules:

### What Grounding Means

1. **Only factual claims**: The recommendation text may only mention facts that exist in the movie's TMDB data (title, year, genres, runtime, rating, overview).

2. **No invention**: The system must NOT invent:
   - Cast or crew information (unless provided by TMDB)
   - Awards or accolades
   - Plot details beyond the official overview
   - Comparative statements about rankings
   - Release dates or box office information

3. **Missing data handling**: If a piece of information is not available (e.g., no runtime), the recommendation simply omits it rather than guessing.

## Constraint Satisfaction

Recommendations must satisfy user-specified constraints:

### Genre Constraints
- If a user asks for "horror", the recommended movie must have "Horror" as one of its genres
- Multiple genres work as OR conditions (asking for "comedy or action" accepts either)

### Runtime Constraints
- `max_runtime_minutes`: Movie runtime must be less than or equal to this value
- `min_runtime_minutes`: Movie runtime must be greater than or equal to this value
- If a movie lacks runtime data and a runtime constraint exists, the movie is considered non-compliant

## Quality Standards

### Tone and Style
- Recommendations should be conversational and warm
- 2-4 sentences explaining why the movie fits
- Reference the user's stated preferences naturally
- Avoid marketing language ("you'll love it!", "10/10 must-watch")

### What Gets Rejected
The EvaluatorAgent will fail a recommendation if:
- Runtime violates constraints
- The movie title was previously rejected in the same session
- Fabricated information appears in the text
- The recommendation discusses a different movie than the one selected
- The text is off-topic or irrelevant to the request

## Retry Behavior

When a recommendation fails evaluation:
1. The failed movie title is added to a rejection list
2. The system selects a different movie from candidates
3. A new recommendation is generated
4. This continues up to 3 times (MAX_RETRIES)
5. After 3 failures, a graceful fallback message is shown

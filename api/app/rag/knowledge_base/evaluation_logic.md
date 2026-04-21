# Evaluation Logic

## Purpose of Evaluation

The EvaluatorAgent acts as a quality gate before recommendations reach users. Its job is to ensure recommendations are:
1. Factually accurate (grounded in TMDB data)
2. Constraint-compliant (matching user preferences)
3. Relevant and well-written

## Evaluation Criteria

### Hard Constraints (Automatic Failure)

These violations cause immediate rejection:

1. **Runtime Violations**
   - Movie exceeds `max_runtime_minutes` when set
   - Movie is shorter than `min_runtime_minutes` when set
   - These are checked deterministically before LLM evaluation

2. **Rejected Title**
   - The recommended movie was already rejected in this session
   - Prevents recommending the same movie twice after failure

### Soft Constraints (LLM Judgment)

The LLM evaluates:

1. **Grounding Violations**
   - Does the text claim facts not in the movie data?
   - Are cast, awards, or plot details invented?
   - Is the rating or runtime stated correctly?

2. **Movie Identity**
   - Does the recommendation talk about the correct movie?
   - Are details accidentally mixed with other films?

3. **Relevance**
   - Does the recommendation address the user's request?
   - Is the tone appropriate?

## Scoring System

### Score Range
- `0.0` to `1.0` where higher is better
- Scores below `0.7` (PASS_THRESHOLD) result in failure

### Pass/Fail Decision
- `passed: true` only when all criteria are satisfied
- `passed: false` triggers a retry with a different movie

## Evaluation Output

The evaluator produces:

```json
{
  "passed": true/false,
  "score": 0.0-1.0,
  "feedback": "Brief explanation of the judgment",
  "constraint_violations": ["list", "of", "violations"],
  "improvement_suggestions": ["actionable", "hints"]
}
```

## Retry Flow

When evaluation fails:

1. `retry_count` is incremented
2. The failed movie title is added to `rejected_titles`
3. `draft_recommendation` is cleared
4. Workflow loops back to `write_recommendation` node
5. Writer selects a different candidate and generates new text

## Retry Limits

- Maximum retries: 3 (configurable via `MAX_RETRIES`)
- After exhausting retries, user receives a fallback message
- Fallback asks user to rephrase or loosen constraints

## Deterministic vs LLM Checks

For efficiency and reliability:
- Runtime and title checks are **deterministic** (no LLM needed)
- Grounding and quality checks use **LLM judgment**
- Deterministic failures short-circuit LLM evaluation when possible

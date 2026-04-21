# Known Limitations

## Memory and Context

### No Persistent Memory
- The system is **stateless between requests**
- Each conversation starts fresh with no memory of previous interactions
- Users cannot say "like the movie you recommended yesterday"
- No user profiles or watch history are maintained

### No Multi-Turn Context
- Each message is processed independently
- The system cannot reference earlier messages in the same conversation
- Clarification follow-ups work within a single request cycle only

## Data Limitations

### TMDB Data Scope
- Movie information comes exclusively from TMDB
- Some movies may have incomplete data (missing runtime, limited overview)
- Very new releases may not be in the database yet
- Regional availability information is not provided

### Genre Limitations
- Only supports TMDB's standard genre categories
- Custom or niche genres may not be recognized
- Genre matching is based on TMDB's classification, which may differ from user expectations

### No Streaming Availability
- The system does not know where movies are available to stream
- Cannot filter by streaming service (Netflix, HBO, etc.)
- Users need to check availability separately

## Recommendation Limitations

### Single Movie Focus
- Each recommendation focuses on one movie at a time
- The system selects the best match rather than providing ranked lists
- Comparison between multiple options is limited

### No Personal Taste Learning
- Cannot learn user preferences over time
- Every interaction uses only the constraints provided in that message
- No collaborative filtering or "users like you also watched"

## Technical Limitations

### Response Time
- LLM calls add latency (typically 2-5 seconds)
- Multiple retries can extend response time
- TMDB API calls add additional latency

### Retry Exhaustion
- After 3 failed attempts, the system returns a fallback message
- Very restrictive constraints may exhaust all candidates
- Obscure genre combinations may have limited matches

## What the System Cannot Do

- Provide streaming links or purchase options
- Remember previous conversations
- Learn from user feedback
- Search by actor, director, or crew
- Filter by release year ranges
- Provide movie trailers or clips
- Compare multiple movies in detail
- Explain detailed plot points beyond TMDB overview

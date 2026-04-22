import os

import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", os.environ.get("API_URL", "http://localhost:8000"))

st.set_page_config(
    page_title="Movie Night Assistant",
    page_icon="🎬",
    layout="centered",
)

st.title("🎬 Movie Night Assistant")
st.caption("Your friendly helper for planning the perfect movie night")

with st.sidebar:
    st.header("Backend Status")
    if st.button("Check Health", use_container_width=True):
        try:
            r = requests.get(f"{BACKEND_URL}/health", timeout=5)
            if r.status_code == 200:
                st.success("Backend is healthy")
            else:
                st.error(f"Status: {r.status_code}")
        except requests.RequestException as e:
            st.error(f"Cannot reach backend: {e}")

    st.divider()

    st.subheader("Debug Options")
    show_debug = st.checkbox("Show debug info", value=False)
    show_raw_json = st.checkbox("Show raw JSON", value=False, disabled=not show_debug)

    st.divider()
    st.caption(f"Backend: {BACKEND_URL}")


def render_debug_panel(debug_data: dict) -> None:
    """Render formatted debug information."""
    route = debug_data.get("route")
    constraints = debug_data.get("constraints", {})
    debug = debug_data.get("debug", {})

    if route:
        route_colors = {
            "movies": "🎬",
            "rag": "📚",
            "hybrid": "🔀",
        }
        route_icon = route_colors.get(route, "❓")
        st.markdown(f"**Route:** {route_icon} `{route}`")

    if constraints:
        genres = constraints.get("genres", [])
        max_runtime = constraints.get("max_runtime_minutes")
        min_runtime = constraints.get("min_runtime_minutes")

        if genres or max_runtime or min_runtime:
            st.markdown("**Extracted Constraints:**")
            if genres:
                st.markdown(f"- Genres: {', '.join(genres)}")
            if max_runtime:
                st.markdown(f"- Max runtime: {max_runtime} min")
            if min_runtime:
                st.markdown(f"- Min runtime: {min_runtime} min")

    if debug:
        if debug.get("rag_query"):
            st.markdown(f"**RAG Query:** `{debug['rag_query']}`")

        if debug.get("retrieved_contexts"):
            st.markdown(f"**Retrieved Contexts:** ({len(debug['retrieved_contexts'])} documents)")
            for i, ctx in enumerate(debug["retrieved_contexts"], 1):
                with st.expander(f"{i}. {ctx.get('title', 'Unknown')} (score: {ctx.get('relevance_score', 'N/A')})"):
                    st.caption(f"Source: {ctx.get('source', 'unknown')}")
                    st.text(ctx.get("content", ""))

        if debug.get("selected_movie"):
            movie = debug["selected_movie"]
            st.markdown("**Selected Movie:**")
            title = movie.get("title", "Unknown")
            year = movie.get("year")
            rating = movie.get("rating")
            runtime = movie.get("runtime_minutes")
            genres = movie.get("genres", [])

            movie_info = f"- **{title}**"
            if year:
                movie_info += f" ({year})"
            st.markdown(movie_info)
            if genres:
                st.markdown(f"  - Genres: {', '.join(genres)}")
            if runtime:
                st.markdown(f"  - Runtime: {runtime} min")
            if rating:
                st.markdown(f"  - Rating: {rating}/10")

        if debug.get("evaluation"):
            eval_data = debug["evaluation"]
            passed = eval_data.get("passed", False)
            score = eval_data.get("score", 0)
            feedback = eval_data.get("feedback", "")
            violations = eval_data.get("constraint_violations", [])

            status_icon = "✅" if passed else "❌"
            st.markdown(f"**Evaluation:** {status_icon} {'Passed' if passed else 'Failed'} (score: {score:.2f})")
            if feedback:
                st.caption(feedback)
            if violations:
                st.markdown("  - Violations: " + ", ".join(violations))

        retry_count = debug.get("retry_count", 0)
        rejected = debug.get("rejected_titles", [])
        if retry_count > 0 or rejected:
            st.markdown(f"**Retries:** {retry_count}")
            if rejected:
                st.markdown(f"**Rejected titles:** {', '.join(rejected)}")


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if show_debug and msg.get("debug"):
            with st.expander("Debug Info", expanded=False):
                if show_raw_json:
                    st.json(msg["debug"])
                else:
                    render_debug_panel(msg["debug"])

if prompt := st.chat_input("What would you like to watch tonight?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                r = requests.post(
                    f"{BACKEND_URL}/chat",
                    json={"message": prompt},
                    timeout=60,
                )

                if r.status_code == 200:
                    data = r.json()
                    reply = data.get("reply", "No response received")
                    route = data.get("route")
                    constraints = data.get("extracted_constraints")
                    debug = data.get("debug")

                    st.markdown(reply)

                    debug_info = {
                        "route": route,
                        "constraints": constraints,
                        "debug": debug,
                    }

                    if show_debug:
                        with st.expander("Debug Info", expanded=True):
                            if show_raw_json:
                                st.json(debug_info)
                            else:
                                render_debug_panel(debug_info)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": reply,
                        "debug": debug_info,
                    })
                elif r.status_code == 422:
                    error_msg = "Invalid message. Please enter some text."
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": f"Error: {error_msg}"})
                else:
                    error_msg = f"Error: {r.status_code} - {r.json().get('detail', 'Unknown error')}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": f"Error: {error_msg}"})
            except requests.RequestException as e:
                error_msg = f"Failed to connect to backend: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": f"Error: {error_msg}"})

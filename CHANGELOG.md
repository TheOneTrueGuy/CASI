# CASI Project Changelog

## [2025-11-20] - Progress Round-Up

### Core Logic & Backend
- **OpenRouter Integration**: 
  - Fully integrated OpenRouter as a first-class backend for both Generator and Critic agents.
  - Implemented API key handling and model selection logic.
  - Set `qwen/qwen3-32b` as the default model for both agents, with fallback to `qwen/qwen3-30b-a3b`.
- **Stable Generator/Critic Loop**:
  - Refined the iterative loop logic in `webCASI.py`.
  - Implemented "Automatic Cycle" mode which orchestrates the turn-taking.
  - Ensured full conversation history is injected into prompts to prevent agent "amnesia."
  - Added logic to switch the Generator's system prompt from "Initial" to "Iteration" mode after the first round.
- **Prompt Engineering**:
  - **Generator**: Updated to a "resilient" persona that does not acquiesce to criticism but vigorously defends/improves the idea (`prompts.json`).
  - **Critic**: Updated to a "mercilessly rigorous" persona to provide high-quality adversarial feedback.

### PythonAnywhere Deployment (Flask)
- **Session Management Overhaul**:
  - Solved "502 Bad Gateway" errors caused by large cookies.
  - Implemented server-side file storage for conversation history (`/tmp/casi_trace_{uuid}.json`) instead of storing it in the session cookie.
  - Only the session UUID is stored in the cookie now.
- **UI/UX Improvements**:
  - Unified the interface into a single page (`views.py` / `casi.html`).
  - **Download Trace**: Added a button to download the full conversation history as a text file.
  - **Form Handling**: Renamed model input fields (`gen_model_id`, `crit_model_id`) to prevent browser autofill from overwriting defaults.
  - **Visual Feedback**: Wired the UI to display the correct system prompts and model IDs.

### Local Interfaces
- **Gradio App (`gradio_app.py`)**:
  - Completely refactored to wrap the shared `webCASI.py` backend logic.
  - Features: Manual/Automatic modes, OpenRouter support, Download Trace, and API key inputs.
  - Ready for Hugging Face Spaces deployment.
- **Streamlit App (`app.py`)**:
  - Aligned with shared backend changes and OpenRouter defaults.

### Cleanup
- **Legacy Code**: Removed old Flask-AppBuilder views that were causing startup crashes (`KeyError`).
- **Documentation**: Updated `README.md` with instructions for Hugging Face deployment and OpenRouter usage.

# CASI: Plans and Progress
**Date:** November 22, 2025

## Current Status
The remote deployment is stable. "Run Generator", "Run Critic", and "Automatic Cycle" (with resume/step functionality) are operational. State management has been moved to server-side files to prevent 502 errors. The Generator now outputs plain text instead of JSON.

---

## User Feedback & Observations

### 1. Interface & Onboarding
*   **Issue:** The interface can be overwhelming for new users.
*   **Goal:** Make the interface more intuitive and self-guiding.
*   **Suggestions:**
    *   Add page elements, resources, or graphics to provide hints.
    *   Implement a "Walkthrough" or "Tour" mode for first-time users.
    *   Add tooltips to technical terms (e.g., "Temperature", "System Prompt").

### 2. Dynamic Prompts & Meta-Feedback
*   **Issue:** System prompts remain static throughout the cycle (except for the basic switch to "Iteration Mode").
*   **Issue:** Agents (Generator/Critic) do not provide feedback *to each other* on how to perform better (e.g., "Critic, your feedback was too vague").
*   **Goal:** Implement adaptive prompting and a meta-feedback loop.
*   **Complexity:** High. Requires managing prompt state or injecting "instructions for the partner" into the context.

---

## Proposed Improvements (AI Suggestions)

### 1. UI/UX Enhancements
*   **Visual State Indicator:** A dynamic diagram or progress bar showing exactly where the process is (e.g., "Generator is thinking..." -> "Critic is reviewing...").
*   **Preset Scenarios:** A dropdown to pre-fill prompts for common tasks (e.g., "Creative Writing", "Code Debugging", "Business Strategy") so users don't have to write system prompts from scratch.
*   **Collapsible History:** Automatically collapse older iterations in the "Session History" to keep the view clean, with a "Expand All" toggle.
*   **Real-time Streaming:** If supported by the backend/hosting, stream the text generation to the UI instead of waiting for the full response (reduces perceived latency).

### 4. Development Strategy
*   **Local Sandbox:** Use the local Gradio and Streamlit interfaces (`gradio_app.py`, `streamlit_app.py`) to prototype and test advanced features like Meta-Feedback and Web Search.
*   **Local Innovation -> Remote Stability:** Only deploy features to the remote server once they have been refined and stabilized locally. This avoids timeout issues and keeps the live demo reliable.

### 2. Advanced Agent Logic
*   **Web Search Integration:** The backend (`webCASI.py`) has `agentic_step` capability (Search/Plan/Synthesize), but it is not currently exposed in the UI. Adding a "Allow Web Search" checkbox would significantly boost capability.
*   **Context Summarization:** For long cycles, the history text becomes massive. Implement an automatic summarization step (e.g., every 3 rounds) to compress the history while retaining key decisions.
*   **Meta-Reviewer Agent (Meta-Feedback):** Introduce a third role (or a periodic step) where an agent analyzes the *quality of the interaction* and updates the system prompts dynamically for the next round.
    *   *Example:* "The Generator is ignoring safety constraints. Critic, please emphasize safety in the next turn."
    *   *Implementation Plan:* Prototype this in the local Gradio interface first.

### 3. Technical Stability
*   **Async Task Queue:** Move the LLM calls to a background task queue (like Celery or Redis Queue) instead of holding the web request open. This is the robust solution to the "3-4 minute wait" and timeout issues.
*   **Database Storage:** Move from file-based state (`/tmp/casi_state_...`) to a proper SQLite or PostgreSQL database model for robust history tracking and user sessions.

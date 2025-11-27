# CASI: Plans and Progress
**Date:** November 25, 2025

## Current Status
The remote deployment is stable. "Run Generator", "Run Critic", and "Automatic Cycle" are operational.
*   **Default Backend:** Google (Gemini 2.5 Flash) - free tier for testing.
*   **Session Management:** 
    *   ✅ "Start Fresh" button clears all fields and history for new ideas.
    *   ✅ Page refresh automatically clears session (no more "Resuming at Round 14" confusion).
*   **UX Improvements:**
    *   ✅ Hover tips on all buttons and inputs.
    *   ✅ Brief descriptions under Generator/Critic headers.
*   **Logic Fixes:** Generator prompt switches to "Iteration Mode" after first critique. Download Trace button works immediately.
*   **Dependencies:** `google-generativeai` installed on server.

---

## User Feedback & Observations

### 1. Interface & Onboarding
*   **Issue:** The interface can be overwhelming for new users.
*   **Goal:** Make the interface more intuitive and self-guiding.
*   **Suggestions:**
    *   **Visual Guidance:** Add arrows or pointers directing the user from "Generator" -> "Critic" -> "Next Step" to visualize the flow.
    *   **Hover Tips:** Add tooltips to buttons and labels explaining their function (e.g., "This runs the Generator using the selected backend").
    *   **Tour Mode:** Implement a "Walkthrough" for first-time users.

### 2. Dynamic Prompts & Meta-Feedback
*   **Issue:** System prompts remain static throughout the cycle (except for the basic switch to "Iteration Mode").
*   **Issue:** Agents (Generator/Critic) do not provide feedback *to each other* on how to perform better (e.g., "Critic, your feedback was too vague").
*   **Goal:** Implement adaptive prompting and a meta-feedback loop.
*   **Complexity:** High. Requires managing prompt state or injecting "instructions for the partner" into the context.

---

## Proposed Improvements (AI Suggestions)

### 1. UI/UX Enhancements
*   ✅ ~~Hover Tips~~ - Implemented Nov 25.
*   **Visual State Indicator:** A dynamic diagram or progress bar showing exactly where the process is (e.g., "Generator is thinking..." -> "Critic is reviewing...").
*   **Preset Scenarios:** A dropdown to pre-fill prompts for common tasks. Each preset includes a full prompt set:
    *   `generator_initial`, `generator_iteration`, `critic_initial`, `critic_iteration`
    *   Example presets: "Code Review", "Essay Refinement", "Brainstorming", "Business Strategy"
    *   Store in `prompts.json` as nested objects.
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

### 5. Usage Limits & Access Control (Post-Testing Phase)
*   **User-Provided Keys:** Re-introduce optional input fields for Google and Groq keys so power users can bypass free-tier limits and use their own quotas.
*   **Rate Limiter:** Implement a rate limiting mechanism (e.g., Flask-Limiter) for the default free-tier keys to prevent abuse and exhaustion of the quota.

---

## Priority Roadmap

| Priority | Feature | Effort | Impact | Status |
|----------|---------|--------|--------|--------|
| 1 | Hover Tips | Low | High (UX) | ✅ Done |
| 2 | Start Fresh / Session Clear | Low | High (UX) | ✅ Done |
| 3 | Input Validation (empty check) | Low | Medium | ✅ Done |
| 4 | Web Search Checkbox | Medium | High | ✅ Verified & Patched Backend |
| 5 | Preset Scenarios Dropdown | Medium | High (UX) | ✅ Done |
| 6 | Visual Flow Arrows | Medium | High (UX) | Pending |
| 7 | Rate Limiter | Medium | Required | Pre-public |
| 8 | Async Task Queue | High | High (stability) | Future |
| 9 | Meta-Feedback Agent | High | Experimental | Future |
| 10 | Orchestrator Agent (Admin/Judge) | High | High (Control) | Future Idea |


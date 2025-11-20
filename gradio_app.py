import gradio as gr
import os
import tempfile
from typing import List, Dict, Any

import webCASI as casi

# Simple Gradio wrapper around webCASI's generator/critic/run_automatic_cycle.
# This is an initial scaffold; we will expand it with full controls and HF-ready behavior.


def run_generator(backend, model, generator_prompt, generator_input, critic_output,
                  openai_key, anthropic_key, openrouter_key,
                  history: List[Dict[str, Any]]):
    """Wrapper that calls casi.generator with appropriate api_key routing.

    Returns both the generator output (for the Generator Output box) and the same
    text for the Critic Input box, so the critic can immediately work from the
    latest generator response.
    """
    api_key = None
    if backend == "openai":
        api_key = openai_key or None
    elif backend == "anthropic":
        api_key = anthropic_key or None
    elif backend == "openrouter":
        api_key = openrouter_key or None

    # Fallback to default model from config if none provided
    if not model:
        model = getattr(casi.config, f"{backend}_model", None)

    output, _suggestions, _trace = casi.generator(
        backend=backend,
        model=model,
        prompt=generator_prompt,
        user_input=generator_input,
        critic_feedback=critic_output,
        api_key=api_key,
    )
    # Update unified history with this manual generator step
    new_history = list(history or [])
    new_history.append({
        "step_type": "generator",
        "mode": "manual",
        "backend": backend,
        "model": model,
        "prompt": generator_prompt,
        "input": generator_input,
        "previous_critic_feedback": critic_output,
        "output": output,
    })

    # Return output twice: once for the Generator Output box and once to
    # populate the Critic Input box, plus updated history.
    return output, output, new_history


def run_critic(backend, model, critic_prompt, critic_input,
               openai_key, anthropic_key, openrouter_key,
               current_generator_prompt,
               history: List[Dict[str, Any]]):
    """Wrapper that calls casi.critic with appropriate api_key routing.

    Returns three values:
    - critic_output: shown in the Critic Output box
    - next_generator_input: set to the critic_output so the Generator can
      address the criticisms on the next turn
    - next_generator_prompt: usually switched from the initial generator
      prompt to the iteration prompt after the first critique
    """
    api_key = None
    if backend == "openai":
        api_key = openai_key or None
    elif backend == "anthropic":
        api_key = anthropic_key or None
    elif backend == "openrouter":
        api_key = openrouter_key or None

    if not model:
        model = getattr(casi.config, f"{backend}_model", None)

    output, _suggestions, _trace = casi.critic(
        backend=backend,
        model=model,
        prompt=critic_prompt,
        generator_output=critic_input,
        api_key=api_key,
    )

    # Decide what the next Generator system prompt should be: once we have
    # at least one critique, move from the "initial" prompt to the
    # "iteration" prompt so the Generator focuses on addressing criticism.
    gen_initial = casi.config.prompts.get("generator_initial")
    gen_iteration = casi.config.prompts.get("generator_iteration", current_generator_prompt)

    if current_generator_prompt == gen_initial and gen_iteration:
        next_generator_prompt = gen_iteration
    else:
        next_generator_prompt = current_generator_prompt

    # Update unified history with this manual critic step
    new_history = list(history or [])
    new_history.append({
        "step_type": "critic",
        "mode": "manual",
        "backend": backend,
        "model": model,
        "prompt": critic_prompt,
        "input": critic_input,
        "output": output,
    })

    # Critic output becomes the next generator input.
    return output, output, next_generator_prompt, new_history


def run_cycle(backend, model, generator_prompt, initial_input, critic_prompt,
              max_iterations, openai_key, anthropic_key, openrouter_key,
              history: List[Dict[str, Any]]):
    """Run the full automatic Generator/Critic cycle via casi.run_automatic_cycle.

    Returns final generator output, final critic output, and a simple
    plain-text summary of the iteration history.
    """

    # Select API key based on backend (both agents use the same backend here).
    api_key = None
    if backend == "openai":
        api_key = openai_key or None
    elif backend == "anthropic":
        api_key = anthropic_key or None
    elif backend == "openrouter":
        api_key = openrouter_key or None

    # Resolve model if not explicitly provided
    if not model:
        model = getattr(casi.config, f"{backend}_model", None)

    results = casi.run_automatic_cycle(
        max_iterations=max_iterations,
        initial_input=initial_input,
        gen_backend=backend,
        gen_model=model,
        gen_prompt=generator_prompt,
        gen_api_key=api_key,
        crit_backend=backend,
        crit_model=model,
        crit_prompt=critic_prompt,
        crit_api_key=api_key,
    )

    cycle_history: List[Dict[str, Any]] = results.get("history", [])

    # Extend unified history with automatic cycle steps (generator + critic per iteration)
    new_history = list(history or [])
    for step in cycle_history:
        it = step.get("iteration")
        new_history.append({
            "step_type": "generator",
            "mode": "automatic",
            "iteration": it,
            "backend": backend,
            "model": model,
            "prompt": generator_prompt,
            "input": step.get("generator_input", ""),
            "output": step.get("generator_output", ""),
        })
        new_history.append({
            "step_type": "critic",
            "mode": "automatic",
            "iteration": it,
            "backend": backend,
            "model": model,
            "prompt": critic_prompt,
            "input": step.get("generator_output", ""),
            "output": step.get("critic_output", ""),
        })

    # Build a compact text summary for display in Gradio from cycle_history
    lines = []
    for step in cycle_history:
        it = step.get("iteration")
        lines.append(f"=== Iteration {it} ===")
        lines.append("Generator output:\n" + step.get("generator_output", ""))
        lines.append("")
        lines.append("Critic output:\n" + step.get("critic_output", ""))
        lines.append("\n" + "-" * 40 + "\n")

    history_text = "\n".join(lines) if lines else "(No history recorded)"

    return (
        results.get("final_generator_output", ""),
        results.get("final_critic_output", ""),
        history_text,
        new_history,
    )


def prepare_trace_file(history: List[Dict[str, Any]]):
    """Create a temporary text file containing the full conversation history.

    This includes both manual Generator/Critic steps and any automatic
    cycle steps, in the order they were recorded.
    """
    if not history:
        text = "No CASI history available yet. Run some steps or a full cycle first."
    else:
        lines: List[str] = []
        for idx, step in enumerate(history, start=1):
            role = step.get("step_type", "unknown").capitalize()
            mode = step.get("mode", "?")
            it = step.get("iteration")
            header = f"=== Step {idx} ({role}, {mode}"
            if it is not None:
                header += f", iteration {it}"
            header += ") ==="
            lines.append(header)
            if "input" in step and step["input"]:
                lines.append("Input:\n" + step["input"])
                lines.append("")
            lines.append("Output:\n" + step.get("output", ""))
            lines.append("\n" + "=" * 40 + "\n")
        text = "\n".join(lines)

    fd, path = tempfile.mkstemp(suffix="_casi_trace.txt")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    return path


with gr.Blocks() as demo:
    gr.Markdown("""# CASI (Gradio)

Gradio wrapper around the CASI Generator/Critic logic from `webCASI.py`.
You can select a backend (including OpenRouter) and optionally provide per-session API keys and model IDs.
""")

    with gr.Row():
        with gr.Column():
            backend = gr.Radio(
                choices=["openai", "anthropic", "google", "ollama", "openrouter"],
                label="Backend",
                value="openai",
            )
            model = gr.Textbox(label="Model ID (optional)")

            openai_key = gr.Textbox(label="OpenAI API Key", type="password")
            anthropic_key = gr.Textbox(label="Anthropic API Key", type="password")
            openrouter_key = gr.Textbox(label="OpenRouter API Key", type="password")

            generator_prompt = gr.Textbox(
                label="Generator System Prompt",
                lines=3,
                value=casi.config.prompts.get("generator_initial", "Formalize and expand this idea."),
            )
            generator_input = gr.Textbox(label="Generator Input", lines=4)
            generator_output = gr.Textbox(label="Generator Output", lines=8)
            gen_button = gr.Button("Run Generator")

        with gr.Column():
            critic_prompt = gr.Textbox(
                label="Critic System Prompt",
                lines=3,
                value=casi.config.prompts.get("critic_initial", "Analyze and critique this idea."),
            )
            critic_input = gr.Textbox(label="Critic Input (Generator Output)", lines=8)
            critic_output = gr.Textbox(label="Critic Output", lines=8)
            crit_button = gr.Button("Run Critic")

    # Hold the full conversation history (manual + automatic) for download
    history_state = gr.State([])

    # Automatic cycle controls and outputs
    with gr.Row():
        with gr.Column():
            max_iterations = gr.Slider(
                minimum=1,
                maximum=20,
                value=5,
                step=1,
                label="Max Iterations (Automatic Cycle)",
            )
            cycle_button = gr.Button("Run Full Automatic Cycle")
        with gr.Column():
            cycle_gen_final = gr.Textbox(
                label="Final Generator Output (Cycle)",
                lines=6,
            )
            cycle_crit_final = gr.Textbox(
                label="Final Critic Output (Cycle)",
                lines=6,
            )
        with gr.Column():
            cycle_history_text = gr.Textbox(
                label="Cycle History Summary",
                lines=12,
            )

    # Download trace of the last automatic cycle
    with gr.Row():
        with gr.Column():
            download_trace_button = gr.Button("Download Trace (Text)")
        with gr.Column():
            trace_file = gr.File(label="Trace File")

    gen_button.click(
        fn=run_generator,
        inputs=[backend, model, generator_prompt, generator_input, critic_output,
                openai_key, anthropic_key, openrouter_key, history_state],
        outputs=[generator_output, critic_input, history_state],
    )

    crit_button.click(
        fn=run_critic,
        inputs=[backend, model, critic_prompt, critic_input,
                openai_key, anthropic_key, openrouter_key,
                generator_prompt, history_state],
        outputs=[critic_output, generator_input, generator_prompt, history_state],
    )

    cycle_button.click(
        fn=run_cycle,
        inputs=[backend, model, generator_prompt, generator_input, critic_prompt,
                max_iterations, openai_key, anthropic_key, openrouter_key, history_state],
        outputs=[cycle_gen_final, cycle_crit_final, cycle_history_text, history_state],
    )

    download_trace_button.click(
        fn=prepare_trace_file,
        inputs=[history_state],
        outputs=[trace_file],
    )


if __name__ == "__main__":
    demo.launch()

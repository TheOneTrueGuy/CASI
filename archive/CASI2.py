import gradio as gr
import openai
import ollama
import os
import json
import ast
from dotenv import load_dotenv
from typing import Tuple, List, Dict, Any, Literal
import time
from pathlib import Path

# Load environment variables and configuration
load_dotenv()

class Config:
    def __init__(self):
        self.config_file = Path("config.json")
        self.config = self.load_config()

        # API Configuration
        self.openai_api_base = os.getenv('OPENAI_API_BASE', self.config['api']['openai']['api_base'])
        self.openai_api_key = os.getenv('OPENAI_API_KEY', self.config['api']['openai']['api_key'])
        self.openai_model = os.getenv('OPENAI_MODEL', self.config['ui']['openai_model'])

        # Ollama config
        self.ollama_host = os.getenv('OLLAMA_HOST', self.config['api']['ollama']['host'])
        self.ollama_model = os.getenv('OLLAMA_MODEL', self.config['ui']['ollama_model'])

        # General config
        self.max_retries = self.config['generation']['retry']['max_attempts']
        self.retry_delay = self.config['generation']['retry']['delay_seconds']
        self.max_tokens = self.config['generation']['max_tokens']
        self.temperature = self.config['generation']['temperature']

        # UI config
        self.ui = self.config['ui']

        # Load prompt templates
        self.prompts = self.load_prompt_templates()

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file or return defaults"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading config.json: {e}")
            print("Using default configuration")

        # Return default configuration if file loading fails
        return {
            "ui": {
                "backend": "openai",
                "openai_model": "local_model",
                "ollama_model": "llama2",
                "auto_run": False,
                "max_iterations": 5,
                "generator": {
                    "prompt": "Formalize and expand this idea.",
                    "input": "",
                    "lines": {"prompt": 3, "input": 2, "output": 5}
                },
                "critic": {
                    "prompt": "Analyze and critique this idea.",
                    "input": "",
                    "lines": {"prompt": 3, "input": 2, "output": 5}
                }
            },
            "api": {
                "openai": {
                    "api_base": "http://localhost:1234/v1/chat/completions",
                    "api_key": "dummy_key"
                },
                "ollama": {
                    "host": "http://localhost:11434"
                }
            },
            "generation": {
                "max_tokens": 2000,
                "temperature": 0.7,
                "retry": {
                    "max_attempts": 3,
                    "delay_seconds": 1
                }
            }
        }

    @staticmethod
    def load_prompt_templates() -> Dict[str, str]:
        default_prompts = {
            "generator_initial": "Formalize and expand this idea. Focus on clarity, creativity, and feasibility.",
            "generator_iteration": "Using your brilliant imagination and knowledge, answer these criticisms step-by-step with new ideas that correct or fulfill each criticism. Ensure your response addresses clarity, creativity, and feasibility.",
            "critic_initial": "You are a constructive critic. Analyze and critique this idea focusing on: 1) Clarity, 2) Creativity, 3) Feasibility. Provide specific suggestions for improvement.",
            "critic_iteration": "Analyze the revised idea, focusing on new aspects introduced by the generator. Rate the improvements and provide further critique on clarity, creativity, and feasibility."
        }

        prompt_file = Path("prompts.json")
        if prompt_file.exists():
            try:
                with open(prompt_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return default_prompts
        return default_prompts

config = Config()

# Set up OpenAI configuration
openai.api_base = config.openai_api_base
openai.api_key = config.openai_api_key

# Set up Ollama client
ollama_client = ollama.Client(host=config.ollama_host)

# Add iteration tracking
iteration_count = 0
max_iterations = config.ui['max_iterations']

def safe_parse_json(raw_response: str) -> Tuple[Dict[str, Any], bool]:
    """Safely parse JSON response with schema validation."""
    try:
        response = json.loads(raw_response)
        # Validate required fields
        if not isinstance(response, dict):
            return {"response": raw_response, "suggestions": []}, False
        if "response" not in response:
            response["response"] = raw_response
        if "suggestions" not in response:
            response["suggestions"] = []
        return response, True
    except json.JSONDecodeError:
        return {"response": raw_response, "suggestions": []}, False

def generate_response(backend: Literal["openai", "ollama"], model: str, prompt: str, input_text: str, critique: str = None) -> str:
    """Generate response with retry logic and error handling for both OpenAI and Ollama backends."""
    if not model:
        return "Error: No model specified"

    messages = [{"role": "system", "content": prompt}]

    if input_text and input_text.strip():
        messages.append({"role": "user", "content": input_text})
    if critique and critique.strip():
        messages.append({"role": "assistant", "content": critique})

    for attempt in range(config.max_retries):
        try:
            if backend == "openai":
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature
                )
                return response.choices[0].message.content
            else:  # ollama
                response = ollama_client.chat(
                    model=model,
                    messages=messages,
                    stream=False,
                    options={
                        "temperature": config.temperature,
                        "num_predict": config.max_tokens
                    }
                )
                return response.message.content

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == config.max_retries - 1:
                return f"Error: Failed to generate response after {config.max_retries} attempts. Error: {str(e)}"
            time.sleep(config.retry_delay * (attempt + 1))

    return "Error: Maximum retries exceeded"

def generator(backend: str, model: str, prompt: str, user_input: str, critic_feedback: str) -> Tuple[str, List[str]]:
    """Generate response with enhanced JSON formatting and specific focus areas."""
    json_prompt = f"{prompt}. Please respond in JSON format with keys for 'response' and 'suggestions'. After generating the output, consider clarity, creativity, and feasibility in your response."

    full_prompt = f"""
    {json_prompt}
    User input: {user_input}
    Critique: {critic_feedback}
    Current iteration: {iteration_count + 1}/{max_iterations}
    """

    raw_response = generate_response(backend, model, full_prompt, "", None)
    response_dict, is_valid_json = safe_parse_json(raw_response)
    return response_dict['response'], response_dict['suggestions']

def critic(backend: str, model: str, prompt: str, critic_input: str, suggestions: List[str] = None) -> Tuple[str, List[str]]:
    """Enhanced critic function with scoring and specific feedback areas."""
    suggestions = suggestions or []
    suggestions_str = ", ".join(suggestions) if suggestions else "No suggestions provided."

    scoring_prompt = """
    Please provide:
    1. Scores (1-10) for:
       - Clarity: How well-explained and understandable is the idea?
       - Creativity: How innovative and original is the solution?
       - Feasibility: How practical and implementable is the proposal?
    2. At least three specific areas for improvement
    3. Concrete suggestions for each area

    Format your response as plain text, with clear sections for scores and suggestions.
    """

    full_prompt = f"""
    {prompt}

    {scoring_prompt}

    Previous suggestions to consider: {suggestions_str}
    Current iteration: {iteration_count + 1}/{max_iterations}

    Critic Input:
    {critic_input}
    """

    response = generate_response(backend, model, full_prompt, "", None)

    # Extract suggestions from the response text using simple parsing
    # Look for lines that start with "- " or "* " as potential suggestions
    new_suggestions = [
        line.strip('- *').strip()
        for line in response.split('\n')
        if line.strip().startswith(('- ', '* '))
    ]

    return response, new_suggestions

def get_ollama_models() -> List[str]:
    """Get list of available Ollama models using ollama_client.list().
    This function tries to extract the model names from the returned list.
    If an error occurs or list is empty, returns the default configured Ollama model."""
    try:
        models = ollama_client.list()
        if isinstance(models, list) and len(models) > 0:
            # Assuming each model dict has a key 'model' based on the ollama-python spec
            return [model.get('model', config.ollama_model) for model in models]
        else:
            print("No models found from Ollama client list method.")
            return [config.ollama_model]
    except Exception as e:
        print(f"Error fetching Ollama models: {e}")
        return [config.ollama_model]

def get_current_model(backend: str, openai_model: str, ollama_model: str) -> str:
    """Return the model name based on the selected backend."""
    if backend.lower() == "openai":
        return openai_model
    elif backend.lower() == "ollama":
        return ollama_model
    else:
        return openai_model

# Gradio interface setup
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            # Model selection
            backend = gr.Radio(
                choices=["openai", "ollama"],
                label="Backend",
                value=config.ui['backend'],
                interactive=True
            )

            # Model selection - both always visible
            with gr.Group():
                openai_model = gr.Textbox(
                    label="OpenAI Model Name",
                    value=config.openai_model,
                    interactive=True
                )
                ollama_model = gr.Dropdown(
                    label="Ollama Model",
                    choices=get_ollama_models(),
                    value=config.ollama_model,
                    interactive=True
                )

            # Auto-run toggle
            auto_run = gr.Checkbox(
                label="Auto Run",
                value=config.ui['auto_run'],
                info="Automatically run iterations between Generator and Critic"
            )

            # Generator components
            gen_prompt = gr.Textbox(label="Generator System Prompt", lines=config.ui['generator']['lines']['prompt'], value=config.prompts["generator_initial"])
            gen_input = gr.Textbox(label="Generator Input", lines=config.ui['generator']['lines']['input'])
            gen_output = gr.Textbox(label="Generator Output", lines=config.ui['generator']['lines']['output'])
            gen_submit = gr.Button("Submit Generator")

        with gr.Column():
            # Critic components
            crit_prompt = gr.Textbox(label="Critic System Prompt", lines=config.ui['critic']['lines']['prompt'], value=config.prompts["critic_initial"])
            crit_input = gr.Textbox(label="Critic Input", lines=config.ui['critic']['lines']['input'])
            crit_output = gr.Textbox(label="Critic Output", lines=config.ui['critic']['lines']['output'])
            crit_submit = gr.Button("Submit Critic")

    # Add reset button
    reset_button = gr.Button("Reset Conversation")

    # Iteration counter display
    iteration_display = gr.Textbox(label="Iteration Progress", value="0/5", interactive=False)

    # Use State to store suggestions and auto-run state
    suggestions_state = gr.State([])

    def refresh_ollama_models(backend_value):
        """Refresh Ollama models list when switching to Ollama backend"""
        if backend_value == "ollama":
            return gr.update(choices=get_ollama_models())
        return gr.update()

    def on_backend_change(backend_value):
        """Update model name when backend changes"""
        if backend_value == "openai":
            return config.openai_model
        else:
            return config.ollama_model

    def reset_conversation():
        """Reset the conversation state"""
        global iteration_count
        iteration_count = 0
        return [
            "",  # gen_output
            config.prompts["generator_initial"],  # gen_prompt
            [],  # suggestions_state
            "0/5",  # iteration_display
            "",  # crit_input
            "",  # crit_output
            config.prompts["critic_initial"],  # crit_prompt
        ]

    def on_gen_submit(backend: str, openai_model: str, ollama_model: str, gen_prompt: str, gen_input: str, crit_output: str, auto_run: bool, suggestions_state: List[str]) -> List[Any]:
        """Handle generator submission with iteration tracking."""
        global iteration_count

        # Reset iteration count if this is a new conversation (only if no input and no prior critic output)
        if not gen_input.strip() and not crit_output.strip():
            iteration_count = 0

        model = get_current_model(backend, openai_model, ollama_model)

        # Include critic's suggestions in the generator prompt
        critic_suggestions_str = ", ".join(suggestions_state) if suggestions_state else "No prior suggestions."
        generator_prompt_with_suggestions = f"""
        {gen_prompt}

        Critic's Suggestions from previous iteration: {critic_suggestions_str}
        """

        output, suggestions = generator(backend, model, generator_prompt_with_suggestions, gen_input, crit_output)

        iteration_count += 1
        progress = f"{iteration_count}/{max_iterations}"

        # If auto-run is enabled and we haven't reached max iterations,
        # trigger the critic automatically
        if auto_run and iteration_count < max_iterations:
            try:
                crit_results = on_crit_submit(backend, openai_model, ollama_model, config.prompts["critic_iteration"],
                                            output, output, []) # Pass generator output as crit_input, empty suggestions
                return [
                    output,  # gen_output
                    config.prompts["generator_iteration"],  # gen_prompt
                    crit_results[2],  # suggestions_state - Use critic's suggestions from crit_results
                    progress,  # iteration_display
                    output,  # crit_input - Pass generator output to critic input in auto-run
                    crit_results[0],  # crit_output
                    config.prompts["critic_iteration"],  # crit_prompt - Corrected prompt update for auto-run
                ]
            except Exception as e:
                print(f"Auto-run error: {e}")
                # Fall back to manual mode if auto-run fails
                pass

        return [
            output,  # gen_output
            config.prompts["generator_iteration"],  # gen_prompt
            suggestions,  # suggestions_state -  Keep generator's suggestions in state for now (could be changed to critic's later)
            progress,  # iteration_display
            output,  # crit_input - Pass generator output to critic input for manual run as well
            crit_output,  # crit_output (unchanged)
            config.prompts["critic_iteration"],  # crit_prompt - Always use iteration prompt after first gen run
        ]

    def on_crit_submit(backend: str, openai_model: str, ollama_model: str, crit_prompt: str, crit_input: str, gen_output: str, suggestions: list) -> List[Any]:
        """Handle critic submission with iteration limit check."""
        model = get_current_model(backend, openai_model, ollama_model)

        # Combine gen_output and crit_input for the critic's input
        full_critic_input = f"""
        Generator Output:
        {gen_output}

        ---

        Critic Additional Instructions/Context:
        {crit_input}
        """

        output, new_suggestions = critic(backend, model, crit_prompt, full_critic_input, suggestions) # Pass combined input to critic

        # Check if maximum iterations reached
        if iteration_count >= max_iterations:
            return [
                f"FINAL EVALUATION (Iteration Limit Reached):\n\n{output}",  # crit_output
                "Maximum iterations reached. This is the final evaluation.",  # crit_prompt
                new_suggestions,  # suggestions_state - Pass new suggestions to state
                f"{iteration_count}/{max_iterations} (Complete)"  # iteration_display
            ]

        return [
            output,  # crit_output
            config.prompts["critic_iteration"],  # crit_prompt
            new_suggestions,  # suggestions_state - Pass new suggestions to state
            f"{iteration_count}/{max_iterations}"  # iteration_display
        ]

    # Event handlers for the buttons
    backend.change(
        fn=refresh_ollama_models,
        inputs=[backend],
        outputs=[ollama_model]
    )

    backend.change(
        fn=on_backend_change,
        inputs=[backend],
        outputs=[openai_model]
    )

    gen_submit.click(
        fn=on_gen_submit,
        inputs=[
            backend, openai_model, ollama_model,
            gen_prompt, gen_input,
            crit_output, auto_run, suggestions_state
        ],
        outputs=[
            gen_output, gen_prompt, suggestions_state,
            iteration_display, crit_input, crit_output,
            crit_prompt
        ]
    )

    crit_submit.click(
        fn=on_crit_submit,
        inputs=[
            backend, openai_model, ollama_model,
            crit_prompt, crit_input,
            gen_output, suggestions_state
        ],
        outputs=[
            crit_output, crit_prompt,
            suggestions_state, iteration_display
        ]
    )

    reset_button.click(
        fn=reset_conversation,
        inputs=[],
        outputs=[
            gen_output, gen_prompt, suggestions_state,
            iteration_display, crit_input, crit_output,
            crit_prompt
        ]
    )

if __name__ == "__main__":
    # Initialize UI with configuration
    demo.launch(
        share=False,  # Set to True if you want to share the interface
        server_name="0.0.0.0",  # Listen on all network interfaces
        server_port=7860  # Default Gradio port
    )

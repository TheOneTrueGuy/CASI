import gradio as gr
import openai
import os
import json
import ast
from dotenv import load_dotenv
from typing import Tuple, List, Dict, Any
import time
from pathlib import Path

# Load environment variables
load_dotenv()

# Configuration
class Config:
    def __init__(self):
        self.api_base = os.getenv('OPENAI_API_BASE', "http://localhost:1234/v1/chat/completions")
        self.api_key = os.getenv('OPENAI_API_KEY', "dummy_key")
        self.model = os.getenv('OPENAI_MODEL', "local_model")
        self.max_retries = 3
        self.retry_delay = 1  # seconds
        
        # Load prompt templates
        self.prompts = self.load_prompt_templates()
    
    @staticmethod
    def load_prompt_templates() -> Dict[str, str]:
        default_prompts = {
            "generator_initial": "Formalize and expand this idea",
            "generator_iteration": "Using your brilliant imagination and knowledge, answer these criticisms step-by-step with new ideas that correct or fulfill each criticism.",
            "critic_initial": "You are a constructive critic, analyze and critique this idea",
            "critic_iteration": "Analyze the revised idea, focusing on new aspects introduced by the generator and provide further critique."
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
openai.api_base = config.api_base
openai.api_key = config.api_key

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

def generate_response(model: str, prompt: str, input_text: str, critique: str = None) -> str:
    """Generate response with retry logic and error handling."""
    messages = [{"role": "system", "content": prompt}]
    
    if input_text and input_text.strip():
        messages.append({"role": "user", "content": input_text})
    if critique and critique.strip():
        messages.append({"role": "assistant", "content": critique})
    
    for attempt in range(config.max_retries):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                max_tokens=150,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt == config.max_retries - 1:
                return f"Error: Failed to generate response after {config.max_retries} attempts. Error: {str(e)}"
            time.sleep(config.retry_delay)
    
    return "Error: Maximum retries exceeded"

def generator(prompt: str, user_input: str, critic_feedback: str) -> Tuple[str, List[str]]:
    """Generate response with JSON formatting."""
    json_prompt = f"{prompt}. Please respond in JSON format with keys for 'response' and 'suggestions'."
    
    full_prompt = f"""
    {json_prompt}
    User input: {user_input}
    Critique: {critic_feedback}
    """
    
    raw_response = generate_response(config.model, full_prompt, "")
    response_dict, is_valid_json = safe_parse_json(raw_response)
    return response_dict['response'], response_dict['suggestions']

def critic(prompt: str, generator_output: str, suggestions: List[str] = None) -> Tuple[str, List[str]]:
    """Critic function with proper suggestion handling."""
    suggestions = suggestions or []
    suggestions_str = ", ".join(suggestions) if suggestions else "No suggestions provided."
    json_prompt = f"{prompt}. Incorporate these suggestions in your analysis: {suggestions_str}"
    
    full_prompt = f"""
    {json_prompt}
    Generator's Output: {generator_output}
    """
    
    raw_response = generate_response(config.model, full_prompt, "")
    response_dict, is_valid_json = safe_parse_json(raw_response)
    return response_dict['response'], response_dict['suggestions']

# Gradio interface setup
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            model_name = gr.Textbox(label="Model Name", value=config.model)
            gen_prompt = gr.Textbox(label="Generator System Prompt", lines=3, value=config.prompts["generator_initial"])
            gen_input = gr.Textbox(label="Generator Input", lines=2)
            gen_output = gr.Textbox(label="Generator Output", lines=5)
            gen_submit = gr.Button("Submit Generator")
        
        with gr.Column():
            crit_prompt = gr.Textbox(label="Critic System Prompt", lines=3, value=config.prompts["critic_initial"])
            crit_input = gr.Textbox(label="Critic Input (Optional)", lines=2)
            crit_output = gr.Textbox(label="Critic Output", lines=5)
            crit_submit = gr.Button("Submit Critic")
    
    # Use State to store suggestions instead of textboxes
    suggestions_state = gr.State([])

    def on_gen_submit(model: str, gen_prompt: str, gen_input: str, crit_output: str) -> Dict[str, Any]:
        """Handle generator submission."""
        output, suggestions = generator(gen_prompt, gen_input, crit_output)
        return {
            gen_output: output,
            gen_prompt: config.prompts["generator_iteration"],
            suggestions_state: suggestions
        }

    def on_crit_submit(model: str, crit_prompt: str, crit_input: str, gen_output: str, suggestions: list) -> Dict[str, Any]:
        """Handle critic submission."""
        output, new_suggestions = critic(crit_prompt, gen_output, suggestions)
        return {
            crit_output: output,
            crit_prompt: config.prompts["critic_iteration"],
            suggestions_state: new_suggestions
        }

    # Event handlers for the buttons
    gen_submit.click(
        fn=on_gen_submit,
        inputs=[model_name, gen_prompt, gen_input, crit_output],
        outputs=[gen_output, gen_prompt, suggestions_state]
    )
    crit_submit.click(
        fn=on_crit_submit,
        inputs=[model_name, crit_prompt, crit_input, gen_output, suggestions_state],
        outputs=[crit_output, crit_prompt, suggestions_state]
    )

if __name__ == "__main__":
    demo.launch()
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
openai.api_base = config.api_base
openai.api_key = config.api_key

# Add iteration tracking
iteration_count = 0
max_iterations = 5  # Maximum number of improvement cycles

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
    """Generate response with enhanced JSON formatting and specific focus areas."""
    json_prompt = f"{prompt}. Please respond in JSON format with keys for 'response' and 'suggestions'. After generating the output, consider clarity, creativity, and feasibility in your response."
    
    full_prompt = f"""
    {json_prompt}
    User input: {user_input}
    Critique: {critic_feedback}
    Current iteration: {iteration_count + 1}/{max_iterations}
    """
    
    raw_response = generate_response(config.model, full_prompt, "")
    response_dict, is_valid_json = safe_parse_json(raw_response)
    return response_dict['response'], response_dict['suggestions']

def critic(prompt: str, generator_output: str, suggestions: List[str] = None) -> Tuple[str, List[str]]:
    """Enhanced critic function with scoring and specific feedback areas."""
    suggestions = suggestions or []
    suggestions_str = ", ".join(suggestions) if suggestions else "No suggestions provided."
    
    # Enhanced scoring prompt
    scoring_prompt = """
    Please provide:
    1. Scores (1-10) for:
       - Clarity: How well-explained and understandable is the idea?
       - Creativity: How innovative and original is the solution?
       - Feasibility: How practical and implementable is the proposal?
    2. At least three specific areas for improvement
    3. Concrete suggestions for each area
    """
    
    json_prompt = f"""{prompt}
    
    {scoring_prompt}
    
    Previous suggestions to consider: {suggestions_str}
    Current iteration: {iteration_count + 1}/{max_iterations}
    """
    
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
            
    # Iteration counter display
    iteration_display = gr.Textbox(label="Iteration Progress", value="0/5", interactive=False)
    
    # Use State to store suggestions
    suggestions_state = gr.State([])

    def on_gen_submit(model: str, gen_prompt: str, gen_input: str, crit_output: str) -> Dict[str, Any]:
        """Handle generator submission with iteration tracking."""
        global iteration_count
        output, suggestions = generator(gen_prompt, gen_input, crit_output)
        
        iteration_count += 1
        progress = f"{iteration_count}/{max_iterations}"
        
        return {
            gen_output: output,
            gen_prompt: config.prompts["generator_iteration"],
            suggestions_state: suggestions,
            iteration_display: progress
        }

    def on_crit_submit(model: str, crit_prompt: str, crit_input: str, gen_output: str, suggestions: list) -> Dict[str, Any]:
        """Handle critic submission with iteration limit check."""
        output, new_suggestions = critic(crit_prompt, gen_output, suggestions)
        
        # Check if maximum iterations reached
        if iteration_count >= max_iterations:
            return {
                crit_output: f"FINAL EVALUATION (Iteration Limit Reached):\n\n{output}",
                crit_prompt: "Maximum iterations reached. This is the final evaluation.",
                suggestions_state: new_suggestions,
                iteration_display: f"{iteration_count}/{max_iterations} (Complete)"
            }
        
        return {
            crit_output: output,
            crit_prompt: config.prompts["critic_iteration"],
            suggestions_state: new_suggestions,
            iteration_display: f"{iteration_count}/{max_iterations}"
        }

    # Event handlers for the buttons
    gen_submit.click(
        fn=on_gen_submit,
        inputs=[model_name, gen_prompt, gen_input, crit_output],
        outputs=[gen_output, gen_prompt, suggestions_state, iteration_display]
    )
    crit_submit.click(
        fn=on_crit_submit,
        inputs=[model_name, crit_prompt, crit_input, gen_output, suggestions_state],
        outputs=[crit_output, crit_prompt, suggestions_state, iteration_display]
    )

if __name__ == "__main__":
    demo.launch()
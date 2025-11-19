import openai
import os
import json
from dotenv import load_dotenv
from typing import Tuple, List, Dict, Any, Literal
import time
from pathlib import Path

# Import SDKs that will be used, with error handling for missing packages
try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    import ollama
except ImportError:
    ollama = None

# Placeholder for other SDKs
# try:
#     import groq
# except ImportError:
#     groq = None

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for webCASI, focused on API-based models."""
    def __init__(self):
        # API Keys from environment variables
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        self.groq_api_key = os.getenv('GROQ_API_KEY')
        
        # Ollama config
        self.ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')

        # Default model selections
        self.openai_model = os.getenv('OPENAI_MODEL', 'gpt-4o')
        self.anthropic_model = os.getenv('ANTHROPIC_MODEL', 'claude-3-opus-20240229')
        self.google_model = os.getenv('GOOGLE_MODEL', 'gemini-1.5-pro-latest')
        self.groq_model = os.getenv('GROQ_MODEL', 'llama3-8b-8192')
        self.ollama_model = os.getenv('OLLAMA_MODEL', 'llama2')

        # Generation parameters
        self.max_retries = 3
        self.retry_delay = 2
        self.max_tokens = 4000
        self.temperature = 0.7
        
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

# Initialize configuration
config = Config()

# OpenAI client is now configured on-the-fly in generate_response to support user-provided keys.

# Configure Google Gemini client
if config.google_api_key and genai:
    genai.configure(api_key=config.google_api_key)

# Configure Ollama client
if ollama:
    ollama_client = ollama.Client(host=config.ollama_host)
else:
    ollama_client = None

def safe_parse_json(raw_response: str) -> Tuple[Dict[str, Any], bool]:
    """Safely parse a JSON string from a model's response."""
    try:
        # The model might wrap the JSON in markdown, so we find the first '{' and last '}'
        start = raw_response.find('{')
        end = raw_response.rfind('}') + 1
        if start != -1 and end != 0:
            clean_response = raw_response[start:end]
            response = json.loads(clean_response)
            if "response" not in response:
                response["response"] = raw_response # Fallback
            if "suggestions" not in response:
                response["suggestions"] = []
            return response, True
    except (json.JSONDecodeError, IndexError):
        pass
    return {"response": raw_response, "suggestions": []}, False

def generate_response(backend: Literal["openai", "anthropic", "google", "groq", "ollama"], model: str, prompt: str, input_text: str, critique: str = None, api_key: str = None) -> str:
    """Generate response with retry logic, using a user-provided API key if available."""
    attempt = 0
    full_prompt = f"{prompt}\n\nUser Input: {input_text}"
    if critique:
        full_prompt += f"\n\nPrevious Critique: {critique}"

    while attempt < config.max_retries:
        try:
            if backend == "openai":
                # Use user's key if provided, otherwise fall back to server config
                key = api_key if api_key else config.openai_api_key
                if not key: raise ValueError("OpenAI API key not found.")
                client = openai.OpenAI(api_key=key)
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": full_prompt}],
                    max_tokens=config.max_tokens,
                    temperature=config.temperature
                )
                return response.choices[0].message.content

            elif backend == "anthropic":
                if not anthropic: raise ImportError("Anthropic SDK not installed.")
                key = api_key if api_key else config.anthropic_api_key
                if not key: raise ValueError("Anthropic API key not found.")
                client = anthropic.Anthropic(api_key=key)
                response = client.messages.create(
                    model=model,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                    messages=[{"role": "user", "content": full_prompt}]
                )
                return "".join([c.text for c in response.content if hasattr(c, 'text')])

            elif backend == "google":
                # Google's SDK is configured globally, so we don't support user keys for it at this time
                # to avoid thread-safety issues. It will use the server's key.
                if not genai: raise ImportError("Google Generative AI SDK not installed.")
                if not config.google_api_key: raise ValueError("Google API key not configured on server.")
                gemini_model = genai.GenerativeModel(model)
                response = gemini_model.generate_content(full_prompt)
                return response.text
            
            elif backend == "ollama":
                if not ollama_client: raise ImportError("Ollama SDK not installed or client not initialized.")
                response = ollama_client.chat(
                    model=model,
                    messages=[{"role": "user", "content": full_prompt}],
                    stream=False,
                    options={
                        "temperature": config.temperature,
                        "num_predict": config.max_tokens
                    }
                )
                return response['message']['content']

            else:
                raise NotImplementedError(f"Backend '{backend}' is not yet implemented or supported for user keys.")

        except Exception as e:
            print(f"Error on attempt {attempt+1} with {backend}: {e}")
            attempt += 1
            if attempt >= config.max_retries:
                return f"Error: Failed to get response from {backend} after {config.max_retries} attempts. Details: {e}"
            time.sleep(config.retry_delay)
    return "Error: Function failed unexpectedly."

def generator(backend: str, model: str, prompt: str, user_input: str, critic_feedback: str, api_key: str = None) -> Tuple[str, List[str]]:
    """Prepares prompt and calls generate_response for the Generator agent, passing the API key."""
    json_prompt = f"{prompt}. Please respond in JSON format with keys for 'response' and 'suggestions'."
    raw_response = generate_response(backend, model, json_prompt, user_input, critic_feedback, api_key=api_key)
    response_dict, _ = safe_parse_json(raw_response)
    return response_dict.get('response', ''), response_dict.get('suggestions', [])

def critic(backend: str, model: str, prompt: str, generator_output: str, api_key: str = None) -> Tuple[str, List[str]]:
    """Prepares prompt and calls generate_response for the Critic agent, passing the API key."""
    scoring_prompt = """
    Please provide:
    1. Scores (1-10) for: Clarity, Creativity, Feasibility.
    2. At least three specific areas for improvement.
    3. Concrete suggestions for each area.
    Format your response as plain text.
    """
    full_prompt = f"{prompt}\n\n{scoring_prompt}\n\nGenerator's Output to critique: {generator_output}"
    response = generate_response(backend, model, full_prompt, "", None, api_key=api_key)
    
    # Simple parsing for suggestions
    suggestions = [line.strip('-* ') for line in response.split('\n') if line.strip().startswith(('-', '*'))]
    return response, suggestions

def run_automatic_cycle(
    max_iterations: int,
    initial_input: str,
    gen_backend: str, gen_model: str, gen_prompt: str, gen_api_key: str,
    crit_backend: str, crit_model: str, crit_prompt: str, crit_api_key: str
) -> Dict[str, Any]:
    """Runs the full Generator-Critic cycle automatically for a set number of iterations."""
    history = []
    current_input = initial_input
    critic_feedback = ""

    for i in range(max_iterations):
        # --- Generator's Turn ---
        gen_output, _ = generator(
            backend=gen_backend, model=gen_model, prompt=gen_prompt,
            user_input=current_input, critic_feedback=critic_feedback, api_key=gen_api_key
        )

        # --- Critic's Turn ---
        crit_output, _ = critic(
            backend=crit_backend, model=crit_model, prompt=crit_prompt,
            generator_output=gen_output, api_key=crit_api_key
        )

        # Store history for this iteration
        history.append({
            'iteration': i + 1,
            'generator_input': current_input if i == 0 else "(From previous critique)",
            'critic_feedback_input': critic_feedback,
            'generator_output': gen_output,
            'critic_output': crit_output
        })

        # Prepare for the next iteration
        critic_feedback = crit_output
        # The generator's original input is only used on the first turn.
        # On subsequent turns, the generator works from the critic's feedback.
        current_input = ""

    return {
        'final_generator_output': gen_output,
        'final_critic_output': crit_output,
        'history': history
    }

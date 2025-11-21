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

# Search tool import
try:
    from duckduckgo_search import DDGS
    ddgs = DDGS()
except ImportError:
    ddgs = None

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
        self.openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
        
        # Ollama config
        self.ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')

        # Default model selections
        self.openai_model = os.getenv('OPENAI_MODEL', 'gpt-4o')
        self.anthropic_model = os.getenv('ANTHROPIC_MODEL', 'claude-3-opus-20240229')
        self.google_model = os.getenv('GOOGLE_MODEL', 'gemini-1.5-pro-latest')
        self.groq_model = os.getenv('GROQ_MODEL', 'llama3-8b-8192')
        self.ollama_model = os.getenv('OLLAMA_MODEL', 'llama2')
        self.openrouter_model = os.getenv('OPENROUTER_MODEL', 'qwen/qwen3-32b')
        self.openrouter_fallback_model = 'qwen/qwen3-30b-a3b'

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
            "generator_initial": "You are a relentless and creative innovator. Formalize and expand this idea. Never give up on solving a problem; if you encounter a block, pivot and find a new angle. Focus on clarity, creativity, and feasibility.",
            "generator_iteration": "Using your brilliant imagination and knowledge, answer these criticisms step-by-step with new ideas that correct or fulfill each criticism. Maintain a 'never give up' attitude—every critique is just a stepping stone to perfection. Ensure your response addresses clarity, creativity, and feasibility.",
            "critic_initial": "You are a constructive but rigorous critic. Analyze and critique this idea focusing on: 1) Clarity, 2) Creativity, 3) Feasibility. Where possible, verify claims using your knowledge or search tools. Provide specific suggestions for improvement.",
            "critic_iteration": "Analyze the revised idea, focusing on new aspects introduced by the generator. Rate the improvements and provide further critique on clarity, creativity, and feasibility. Cite sources or verified facts if applicable."
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

def search_web(query: str, max_results: int = 3) -> List[Dict[str, str]]:
    """Perform a web search using DuckDuckGo."""
    if not ddgs:
        return [{"title": "Error", "body": "Search module (duckduckgo-search) not installed.", "href": ""}]
    try:
        results = list(ddgs.text(query, max_results=max_results))
        return results
    except Exception as e:
        return [{"title": "Error", "body": f"Search failed: {str(e)}", "href": ""}]

def format_search_results(results: List[Dict[str, str]]) -> str:
    """Format search results for inclusion in a prompt."""
    formatted = "Search Results:\n"
    for r in results:
        formatted += f"- [{r.get('title', 'No Title')}]({r.get('href', '#')}): {r.get('body', '')}\n"
    return formatted

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

def generate_response(backend: Literal["openai", "anthropic", "google", "groq", "ollama", "openrouter"], model: str, prompt: str, input_text: str, critique: str = None, api_key: str = None) -> str:
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

            elif backend == "openrouter":
                # OpenRouter uses OpenAI SDK with custom base_url
                key = api_key if api_key else config.openrouter_api_key
                if not key: raise ValueError("OpenRouter API key not found.")
                
                client = openai.OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=key,
                )
                
                # OpenRouter recommends adding these headers
                extra_headers = {
                    "HTTP-Referer": "https://github.com/TheOneTrueGuy/CASI",
                    "X-Title": "CASI"
                }
                
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": full_prompt}],
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                    extra_headers=extra_headers
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

def agentic_step(backend: str, model: str, role: str, prompt: str, context: str, api_key: str = None) -> Tuple[str, Dict[str, Any]]:
    """
    Executes an agentic step: Plan -> Search -> Synthesize.
    
    1. PLAN: Ask the model what it needs to search for.
    2. SEARCH: Execute the search queries.
    3. SYNTHESIZE: Provide the original prompt + search results to the model for the final output.
    
    Returns:
        Tuple[str, Dict]: (Augmented Prompt, Trace Data)
    """
    trace = {
        "role": role,
        "plan_prompt": "",
        "plan_response": "",
        "search_queries": [],
        "search_results": [],
        "augmented_prompt_snippet": ""
    }
    
    # Step 1: Planning
    plan_prompt = f"""
    You are an expert {role}. You are about to perform the following task:
    
    {prompt}
    
    Context:
    {context}
    
    Determine if you need external information to perform this task to the highest standard.
    If yes, provide up to 3 specific search queries.
    If no, strictly respond with "NO_SEARCH_NEEDED".
    
    Output Format:
    Just the search queries, one per line.
    """
    trace["plan_prompt"] = plan_prompt
    
    plan_response = generate_response(backend, model, plan_prompt, "", None, api_key=api_key)
    trace["plan_response"] = plan_response
    
    search_context = ""
    if "NO_SEARCH_NEEDED" not in plan_response:
        queries = [line.strip() for line in plan_response.split('\n') if line.strip()]
        trace["search_queries"] = queries
        
        all_results = []
        for q in queries[:3]: # Limit to 3 queries
             results = search_web(q, max_results=2)
             all_results.extend(results)
             trace["search_results"].append({"query": q, "results": results})
        
        if all_results:
            search_context = format_search_results(all_results)
            
    # Step 2: Final Generation with Context
    final_prompt = f"{prompt}\n\n"
    if search_context:
        augmentation = f"Thinking Process:\nI have researched the following information to help with my task:\n{search_context}\n\n"
        final_prompt += augmentation
        trace["augmented_prompt_snippet"] = augmentation
        
    return final_prompt, trace

def generator(backend: str, model: str, prompt: str, user_input: str, critic_feedback: str, api_key: str = None, use_search: bool = False) -> Tuple[str, List[str], Dict[str, Any]]:
    """Prepares prompt and calls generate_response for the Generator agent, passing the API key."""
    
    trace_data = {}
    final_prompt = prompt
    if use_search:
        # Run the agentic planning step to augment the prompt with search results
        context = f"User Input: {user_input}\nCritique: {critic_feedback}"
        final_prompt, trace_data = agentic_step(backend, model, "Generator", prompt, context, api_key=api_key)
    
    json_prompt = f"{final_prompt}. Please respond in JSON format with keys for 'response' and 'suggestions'."
    raw_response = generate_response(backend, model, json_prompt, user_input, critic_feedback, api_key=api_key)
    response_dict, _ = safe_parse_json(raw_response)
    return response_dict.get('response', ''), response_dict.get('suggestions', []), trace_data

def critic(backend: str, model: str, prompt: str, generator_output: str, api_key: str = None, use_search: bool = False) -> Tuple[str, List[str], Dict[str, Any]]:
    """Prepares prompt and calls generate_response for the Critic agent, passing the API key."""
    
    trace_data = {}
    final_prompt = prompt
    if use_search:
        # Run the agentic planning step to augment the prompt with search results
        context = f"Generator's Output: {generator_output}"
        final_prompt, trace_data = agentic_step(backend, model, "Critic", prompt, context, api_key=api_key)
    
    scoring_prompt = """
    Please provide:
    1. Scores (1-10) for: Clarity, Creativity, Feasibility.
    2. At least three specific areas for improvement.
    3. Concrete suggestions for each area.
    Format your response as plain text.
    """
    full_prompt = f"{final_prompt}\n\n{scoring_prompt}\n\nGenerator's Output to critique: {generator_output}"
    response = generate_response(backend, model, full_prompt, "", None, api_key=api_key)
    
    # Simple parsing for suggestions
    suggestions = [line.strip('-* ') for line in response.split('\n') if line.strip().startswith(('-', '*'))]
    return response, suggestions, trace_data

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
    
    # Determine if we should switch to iteration prompts after round 1
    # We only switch if the user started with the default initial prompts.
    gen_prompt_iter = config.prompts.get("generator_iteration")
    crit_prompt_iter = config.prompts.get("critic_iteration")
    
    use_gen_iter = (gen_prompt == config.prompts.get("generator_initial"))
    use_crit_iter = (crit_prompt == config.prompts.get("critic_initial"))

    for i in range(max_iterations):
        # Determine prompts for this iteration
        current_gen_prompt = gen_prompt
        current_crit_prompt = crit_prompt
        
        if i > 0:
            if use_gen_iter and gen_prompt_iter:
                current_gen_prompt = gen_prompt_iter
            if use_crit_iter and crit_prompt_iter:
                current_crit_prompt = crit_prompt_iter

        # Prepare Generator Input with Context
        if i == 0:
            gen_input_text = current_input
        else:
            # In subsequent rounds, provide full context to the generator
            history_text = format_history_as_text(history)
            gen_input_text = f"ORIGINAL GOAL: {initial_input}\n\nPREVIOUS HISTORY:\n{history_text}\n\nLATEST CRITIQUE:\n{critic_feedback}"

        # --- Generator's Turn ---
        gen_output, _, gen_trace = generator(
            backend=gen_backend, model=gen_model, prompt=current_gen_prompt,
            user_input=gen_input_text, critic_feedback="" if i > 0 else critic_feedback, api_key=gen_api_key
        )
        
        # Prepare Critic Input with Context
        if i == 0:
            crit_input_text = gen_output
        else:
            # In subsequent rounds, provide full context to the critic
            crit_input_text = f"ORIGINAL GOAL: {initial_input}\n\nPREVIOUS HISTORY:\n{history_text}\n\nNEW DRAFT TO CRITIQUE:\n{gen_output}"

        # --- Critic's Turn ---
        crit_output, _, crit_trace = critic(
            backend=crit_backend, model=crit_model, prompt=current_crit_prompt,
            generator_output=crit_input_text, api_key=crit_api_key
        )

        # Store history for this iteration
        history.append({
            'iteration': i + 1,
            'generator_input': current_input if i == 0 else "(From previous critique)",
            'critic_feedback_input': critic_feedback,
            'generator_output': gen_output,
            'critic_output': crit_output,
            'generator_trace': gen_trace,
            'critic_trace': crit_trace
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

def format_history_as_text(history: List[Dict[str, Any]]) -> str:
    """Formats the conversation history into a readable text log."""
    output = []
    for step in history:
        output.append(f"=== Iteration {step['iteration']} ===")
        if step['iteration'] == 1:
            output.append(f"Original Input:\n{step.get('generator_input', '')}\n")
        
        output.append(f"--- Generator Output ---\n{step.get('generator_output', '')}\n")
        output.append(f"--- Critic Feedback ---\n{step.get('critic_output', '')}\n")
        output.append("-" * 40 + "\n")
    return "".join(output)

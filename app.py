import streamlit as st
import json
import xml.dom.minidom
from pathlib import Path
from typing import Any, Dict, List
import sys
import importlib.util
import datetime
import os

# --- Import agent logic from webCASI.py ---
CASI_PATH = Path(__file__).parent / "webCASI.py"
spec = importlib.util.spec_from_file_location("casi_module", CASI_PATH)
casi = importlib.util.module_from_spec(spec)
sys.modules["casi_module"] = casi
spec.loader.exec_module(casi)

# --- Data Storage and Format Handling ---
DATA_FILE = Path("exchanges.json")

# Helper functions for serialization

def serialize_exchange(exchange: Dict[str, Any], fmt: str) -> str:
    if fmt == "JSON":
        return json.dumps(exchange, indent=2)
    elif fmt == "XML":
        xml_str = dict_to_xml("exchange", exchange)
        return xml.dom.minidom.parseString(xml_str).toprettyxml()
    else:  # Plain text
        return f"Generator: {exchange['generator']}\nCritic: {exchange['critic']}\nComment: {exchange.get('comment','')}"

def dict_to_xml(tag: str, d: Dict[str, Any]) -> str:
    parts = [f'<{tag}>']
    for k, v in d.items():
        if isinstance(v, dict):
            parts.append(dict_to_xml(k, v))
        elif isinstance(v, list):
            for item in v:
                parts.append(dict_to_xml(k[:-1] if k.endswith('s') else k, item) if isinstance(item, dict) else f'<{k}>{item}</{k}>')
        else:
            parts.append(f'<{k}>{v}</{k}>')
    parts.append(f'</{tag}>')
    return ''.join(parts)

def load_exchanges() -> List[Dict[str, Any]]:
    if DATA_FILE.exists():
        with open(DATA_FILE, 'r') as f:
            return json.load(f)
    return []

def save_exchanges(exchanges: List[Dict[str, Any]]):
    with open(DATA_FILE, 'w') as f:
        json.dump(exchanges, f, indent=2)

# --- Thread Data Structure ---

def now_filename():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".txt"

THREAD_FILE = Path(now_filename())

# Each exchange is a dict with generator/critic state, prompts, outputs, modes, flags, timestamp
# thread = [ { 'generator': {...}, 'critic': {...}, 'timestamp': ... } ]
def load_thread():
    if THREAD_FILE.exists():
        with open(THREAD_FILE, 'r') as f:
            return json.load(f)
    return []

def save_thread(thread):
    with open(THREAD_FILE, 'w') as f:
        json.dump(thread, f, indent=2)

def get_backend_from_service(service_name: str) -> str:
    mapping = {
        "Local (Ollama)": "ollama",
        "OpenAI": "openai",
        "Google Gemini": "google",
        "Anthropic Claude": "anthropic",
        "Grok/Groq": "groq",
        "OpenRouter": "openrouter"
    }
    return mapping.get(service_name, "openai")

# --- Streamlit UI ---
st.set_page_config(page_title="CASI: Cyclical Adversarial Stepwise Improvement", layout="wide")
st.title("CASI: Cyclical Adversarial Stepwise Improvement")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("Controls")
    
    # Navigation
    col_nav1, col_nav2 = st.columns(2)
    with col_nav1:
        if st.button("Back"):
            st.session_state.thread_idx = max(0, st.session_state.thread_idx - 1)
            st.experimental_rerun()
    with col_nav2:
        if st.button("Forward"):
            st.session_state.thread_idx = min(len(load_thread()) - 1, st.session_state.thread_idx + 1)
            st.experimental_rerun()

    fmt = st.radio("Format", ["JSON", "XML", "Text"], key="fmt")
    max_rounds = st.number_input("Max Rounds (Automatic Mode)", min_value=1, max_value=100, value=10, step=1, key="max_rounds")
    
    st.markdown("---")
    st.subheader("Agentic Capabilities")
    use_search_gen = st.checkbox("Enable Web Search for Generator", value=False, help="Allows the Generator to research before creating content.")
    use_search_crit = st.checkbox("Enable Web Search for Critic", value=False, help="Allows the Critic to verify facts and find citations.")

    st.markdown("---")
    with st.expander("API Configuration", expanded=False):
        st.caption("Enter API keys here or set them in your .env file.")
        
        openai_key = st.text_input("OpenAI API Key", value=os.getenv("OPENAI_API_KEY", ""), type="password", key="openai_key_input")
        anthropic_key = st.text_input("Anthropic API Key", value=os.getenv("ANTHROPIC_API_KEY", ""), type="password", key="anthropic_key_input")
        google_key = st.text_input("Google API Key", value=os.getenv("GOOGLE_API_KEY", ""), type="password", key="google_key_input")
        openrouter_key = st.text_input("OpenRouter API Key", value=os.getenv("OPENROUTER_API_KEY", ""), type="password", key="openrouter_key_input")
        
        # Store keys in a dictionary for easy access
        api_keys = {
            "openai": openai_key,
            "anthropic": anthropic_key,
            "google": google_key,
            "openrouter": openrouter_key,
            "ollama": None # No key needed
        }
    
    st.markdown("---")
    st.subheader("Export")
    
    def generate_export_data(thread_data, format_type):
        if format_type == "JSON":
            return json.dumps(thread_data, indent=2)
        elif format_type == "XML":
            # Wrap list in root element
            return dict_to_xml("thread", {"exchanges": thread_data})
        else: # Text
            out = []
            for i, step in enumerate(thread_data):
                out.append(f"=== Step {i+1} ===")
                out.append(f"Timestamp: {step.get('timestamp')}")
                
                gen = step.get('generator', {})
                out.append(f"\n--- GENERATOR ({gen.get('service', 'unknown')} - {gen.get('model', 'unknown')}) ---")
                out.append(f"Input: {gen.get('input')}")
                
                if 'generator_trace' in step and step['generator_trace']:
                    trace = step['generator_trace']
                    out.append(f"\n[Thinking Process]: {trace.get('plan_response')}")
                    for q in trace.get('search_results', []):
                        out.append(f"  - Search: {q['query']}")
                
                out.append(f"\nOutput:\n{gen.get('output')}")
                
                crit = step.get('critic', {})
                if crit.get('output'):
                    out.append(f"\n--- CRITIC ({crit.get('service', 'unknown')} - {crit.get('model', 'unknown')}) ---")
                    
                    if 'critic_trace' in step and step['critic_trace']:
                        trace = step['critic_trace']
                        out.append(f"\n[Thinking Process]: {trace.get('plan_response')}")
                        for q in trace.get('search_results', []):
                            out.append(f"  - Search: {q['query']}")
                            
                    out.append(f"\nCritique:\n{crit.get('output')}")
                out.append("\n" + "="*20 + "\n")
            return "\n".join(out)

    thread_data = load_thread()
    if thread_data:
        export_format = st.selectbox("Export Format", ["Text", "JSON", "XML"])
        export_str = generate_export_data(thread_data, export_format)
        file_ext = export_format.lower() if export_format != "Text" else "txt"
        
        st.download_button(
            label=f"Download Conversation ({export_format})",
            data=export_str,
            file_name=f"casi_session_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.{file_ext}",
            mime=f"text/{file_ext}"
        )

def get_api_key_for_backend(backend: str) -> str:
    return api_keys.get(backend)

thread = load_thread()
if 'thread_idx' not in st.session_state:
    st.session_state.thread_idx = len(thread) - 1 if thread else -1

current_state = thread[st.session_state.thread_idx] if (0 <= st.session_state.thread_idx < len(thread)) else None

def serialize_state(state, fmt):
    if fmt == "JSON":
        return json.dumps(state, indent=2)
    return str(state) # Fallback

# --- Trace Visualizer ---
def render_trace(trace_data, label):
    if not trace_data or "plan_response" not in trace_data:
        return
    
    with st.expander(f"🧠 {label} Thinking Process", expanded=False):
        st.markdown("### 1. Planning")
        st.markdown(f"**Intent:** {trace_data.get('plan_response', 'No plan generated')}")
        
        queries = trace_data.get('search_queries', [])
        if queries:
            st.markdown("### 2. Research")
            for item in trace_data.get('search_results', []):
                st.markdown(f"**Query:** `{item['query']}`")
                for res in item['results']:
                    st.markdown(f"- [{res.get('title')}]({res.get('href')})")
        else:
            st.info("No external search was deemed necessary.")

# --- Dual Column UI ---
col_gen, col_crit = st.columns(2)

# Common service options
service_options = {
    "Local (Ollama)": ["llama2", "mistral", "custom-local-model"],
    "OpenAI": ["gpt-3.5-turbo", "gpt-4", "gpt-4o"],
    "Google Gemini": ["gemini-1.5-pro-latest", "gemini-pro"],
    "Anthropic Claude": ["claude-3-opus-20240229", "claude-3-sonnet-20240229"],
    "OpenRouter": ["qwen/qwen3-32b", "anthropic/claude-3-opus", "openai/gpt-4-turbo", "google/gemini-pro-1.5", "meta-llama/llama-3-70b-instruct"],
    "Grok/Groq": ["llama3-8b-8192", "mixtral-8x7b-32768"],
}

with col_gen:
    st.subheader("Generator Agent")
    gen_mode = st.radio("Mode", ["Manual", "Automatic"], key="gen_mode")
    gen_resend = st.checkbox("Resend last round if no response", key="gen_resend")
    
    gen_service = st.selectbox("Generator Service/API", list(service_options.keys()), key="gen_service")
    
    # Allow custom model input for OpenRouter or others
    if gen_service == "OpenRouter":
        gen_model = st.text_input("Generator Model (OpenRouter ID)", value=casi.config.openrouter_model, key="gen_model_custom")
    else:
        gen_model = st.selectbox("Generator Model", service_options[gen_service], key="gen_model")
    
    default_gen_prompt = casi.config.prompts["generator_initial"]
    gen_prompt = st.text_area("Generator Prompt", value=(current_state['generator']['prompt'] if current_state else default_gen_prompt), disabled=(gen_mode == "Automatic"))
    gen_input = st.text_area("Generator Input", value=(current_state['generator']['input'] if current_state else ""), disabled=(gen_mode == "Automatic"))
    gen_output = st.text_area("Generator Output", value=(current_state['generator']['output'] if current_state else ""), disabled=True)
    
    # Render trace if available
    if current_state and 'generator_trace' in current_state:
        render_trace(current_state['generator_trace'], "Generator")

    if st.button("Run Generator"):
        backend = get_backend_from_service(gen_service)
        api_key = get_api_key_for_backend(backend)
        
        # Fallback to default model if empty
        if not gen_model:
            gen_model = getattr(casi.config, f"{backend}_model", None)

        
        # If resend and no previous output, use last input
        if gen_resend and not gen_output.strip() and thread:
            prev_input = thread[st.session_state.thread_idx]['generator']['input']
            gen_input = prev_input
            
        with st.spinner("Generator is thinking..."):
            # webCASI returns (response, suggestions, trace_data)
            gen_result = casi.generator(backend, gen_model, gen_prompt, gen_input, critic_feedback="", api_key=api_key, use_search=use_search_gen)
            gen_out = gen_result[0]
            gen_trace = gen_result[2]
        
        new_state = {
            'generator': {'prompt': gen_prompt, 'input': gen_input, 'output': gen_out, 'mode': gen_mode, 'resend': gen_resend, 'service': gen_service, 'model': gen_model},
            'critic': current_state['critic'] if current_state else {'prompt': casi.config.prompts["critic_initial"], 'input': '', 'output': '', 'mode': 'Manual', 'resend': False, 'service': 'Local (Ollama)', 'model': 'llama2'},
            'timestamp': datetime.datetime.now().isoformat(),
            'generator_trace': gen_trace
        }
        thread = thread[:st.session_state.thread_idx+1] + [new_state]
        save_thread(thread)
        st.session_state.thread_idx += 1
        st.experimental_rerun()
    
    st.code(serialize_state({'service': gen_service, 'model': gen_model, 'prompt': gen_prompt, 'input': gen_input, 'output': gen_output}, fmt), language=fmt.lower() if fmt != "Text" else "text")

with col_crit:
    st.subheader("Critic Agent")
    crit_mode = st.radio("Mode", ["Manual", "Automatic"], key="crit_mode")
    crit_resend = st.checkbox("Resend last round if no response", key="crit_resend")
    
    crit_service = st.selectbox("Critic Service/API", list(service_options.keys()), key="crit_service")
    
    if crit_service == "OpenRouter":
        crit_model = st.text_input("Critic Model (OpenRouter ID)", value=casi.config.openrouter_model, key="crit_model_custom")
    else:
        crit_model = st.selectbox("Critic Model", service_options[crit_service], key="crit_model")
    
    default_crit_prompt = casi.config.prompts["critic_initial"]
    crit_prompt = st.text_area("Critic Prompt", value=(current_state['critic']['prompt'] if current_state else default_crit_prompt), disabled=(crit_mode == "Automatic"))
    crit_input = st.text_area("Critic Input", value=(current_state['critic']['input'] if current_state else (current_state['generator']['output'] if current_state else "")), disabled=(crit_mode == "Automatic"))
    crit_output = st.text_area("Critic Output", value=(current_state['critic']['output'] if current_state else ""), disabled=True)
    
    # Render trace if available
    if current_state and 'critic_trace' in current_state:
        render_trace(current_state['critic_trace'], "Critic")

    if st.button("Run Critic"):
        backend = get_backend_from_service(crit_service)
        api_key = get_api_key_for_backend(backend)
        
        # Fallback to default model if empty
        if not crit_model:
            crit_model = getattr(casi.config, f"{backend}_model", None)

        
        # If resend and no previous output, use last input
        if crit_resend and not crit_output.strip() and thread:
            prev_input = thread[st.session_state.thread_idx]['critic']['input']
            crit_input = prev_input
            
        with st.spinner("Critic is thinking..."):
            # webCASI returns (response, suggestions, trace_data)
            crit_result = casi.critic(backend, crit_model, crit_prompt, crit_input, api_key=api_key, use_search=use_search_crit)
            crit_out = crit_result[0]
            crit_trace = crit_result[2]
        
        new_state = {
            'generator': current_state['generator'] if current_state else {'prompt': casi.config.prompts["generator_initial"], 'input': '', 'output': '', 'mode': 'Manual', 'resend': False, 'service': 'Local (Ollama)', 'model': 'llama2'},
            'critic': {'prompt': crit_prompt, 'input': crit_input, 'output': crit_out, 'mode': crit_mode, 'resend': crit_resend, 'service': crit_service, 'model': crit_model},
            'timestamp': datetime.datetime.now().isoformat(),
            'critic_trace': crit_trace
        }
        thread = thread[:st.session_state.thread_idx+1] + [new_state]
        save_thread(thread)
        st.session_state.thread_idx += 1
        st.experimental_rerun()
        
    st.code(serialize_state({'service': crit_service, 'model': crit_model, 'prompt': crit_prompt, 'input': crit_input, 'output': crit_output}, fmt), language=fmt.lower() if fmt != "Text" else "text")

# --- Strict Alternation in Automatic Mode ---
if (current_state
    and current_state.get('generator', {}).get('mode') == "Automatic"
    and current_state.get('critic', {}).get('mode') == "Automatic"
    and st.session_state.thread_idx == len(thread) - 1):

    # Automatic mode with max rounds enforcement
    if 'auto_halted' not in st.session_state:
        st.session_state.auto_halted = False

    # A round is complete if the critic has provided output.
    is_round_complete = bool(current_state.get('critic', {}).get('output'))
    round_count = len(thread)

    if round_count >= max_rounds and is_round_complete:
        st.session_state.auto_halted = True
        st.info(f"Max rounds ({max_rounds}) reached. Automatic mode halted.")
    else:
        # Critic's turn: Generator has run, but Critic has not.
        if current_state.get('generator', {}).get('output') and not is_round_complete:
            st.info(f"Running Critic (Round {round_count}/{max_rounds})...")
            
            # Use service/model from the current state's critic settings
            crit_service_name = current_state['critic']['service']
            crit_model_name = current_state['critic']['model']
            backend = get_backend_from_service(crit_service_name)
            api_key = get_api_key_for_backend(backend)
            
            crit_prompt = current_state['critic']['prompt']
            crit_input = current_state['generator']['output']
            
            with st.spinner("Critic is thinking..."):
                crit_result = casi.critic(backend, crit_model_name, crit_prompt, crit_input, api_key=api_key, use_search=use_search_crit)
                crit_out = crit_result[0]
                crit_trace = crit_result[2]
            
            # UPDATE the current state with the critic's output
            thread[-1]['critic']['output'] = crit_out
            thread[-1]['timestamp'] = datetime.datetime.now().isoformat()
            thread[-1]['critic_trace'] = crit_trace
            
            save_thread(thread)
            st.experimental_rerun()

        # Generator's turn: The previous round is complete.
        elif is_round_complete:
            st.info(f"Running Generator (Round {round_count + 1}/{max_rounds})...")
            
            # Use service/model from the current state's generator settings
            gen_service_name = current_state['generator']['service']
            gen_model_name = current_state['generator']['model']
            backend = get_backend_from_service(gen_service_name)
            api_key = get_api_key_for_backend(backend)

            gen_prompt = casi.config.prompts['generator_iteration']
            gen_input = current_state['generator']['output'] # Base for next iteration
            critic_feedback = current_state['critic']['output'] # Feedback for refinement

            with st.spinner("Generator is thinking..."):
                gen_result = casi.generator(backend, gen_model_name, gen_prompt, gen_input, critic_feedback, api_key=api_key, use_search=use_search_gen)
                gen_out = gen_result[0]
                gen_trace = gen_result[2]

            # Create a NEW state for the new round
            # Preserve the service/model settings from the previous round
            new_state = {
                'generator': {
                    'prompt': gen_prompt, 
                    'input': gen_input, 
                    'output': gen_out, 
                    'mode': 'Automatic', 
                    'resend': False,
                    'service': gen_service_name,
                    'model': gen_model_name
                },
                'critic': {
                    'prompt': casi.config.prompts['critic_iteration'], 
                    'input': '', 
                    'output': '', 
                    'mode': 'Automatic', 
                    'resend': False,
                    'service': current_state['critic']['service'],
                    'model': current_state['critic']['model']
                },
                'timestamp': datetime.datetime.now().isoformat(),
                'generator_trace': gen_trace
            }
            
            thread.append(new_state)
            save_thread(thread)
            st.session_state.thread_idx += 1
            st.experimental_rerun()

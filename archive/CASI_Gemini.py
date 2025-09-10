#Cyclical Adversarail Stepwise Improvement
import gradio as gr
import google.generativeai as genai
import os, json
import re
from datetime import datetime

# Initialize logging
log_filename = datetime.now().strftime("%Y%m%d_%H%M%S_CASI.txt")

def log_interaction(interaction_type, prompt, input_text, output):
    with open(log_filename, 'a', encoding='utf-8') as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"\n{'='*50}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Type: {interaction_type}\n")
        f.write(f"Prompt: {prompt}\n")
        f.write(f"Input: {input_text}\n")
        f.write(f"Output: {output}\n")
        f.write(f"{'='*50}\n")

# from google.colab import userdata
api_key ="AIzaSyBPv2RhmsMT0HPcMyfibd2ELTuCAiCH1j0" #userdata.get("GEMINI_API_KEY")


# Configure the Gemini API client
genai.configure(api_key=api_key)

# Create a generative model object
model = genai.GenerativeModel("gemini-1.5-flash")  # Replace with your model name if different

def generate_response(prompt, input_text, critique=None):
    # Construct the prompt content
    full_content = prompt
    if input_text:
        full_content += f"\n\nUser input: {input_text}"
    if critique:
        full_content += f"\n\nPrevious critique: {critique}"

    # Add explicit instruction for JSON formatting
    full_content += "\n\nPlease format your response as JSON with 'response' and 'suggestions' fields, but make the response field contain natural language without JSON artifacts."

    # Send request to generate content
    response = model.generate_content(full_content)
    predicted_output = response.text

    try:
        # Try to parse as JSON first
        json_response = json.loads(predicted_output)
        return json_response['response'], json_response.get('suggestions', [])
    except json.JSONDecodeError:
        # If it's not valid JSON, try to extract content between curly braces
        json_match = re.search(r'\{.*\}', predicted_output, re.DOTALL)
        if json_match:
            try:
                json_response = json.loads(json_match.group())
                return json_response['response'], json_response.get('suggestions', [])
            except (json.JSONDecodeError, KeyError):
                pass
        
        # If all parsing fails, return the raw output and empty suggestions
        return predicted_output.strip(), []

def generator(prompt, user_input, critic_feedback):
    # Modify the prompt to be more explicit about JSON formatting
    json_prompt = f"{prompt}\n\nPlease provide your response in JSON format with two fields:\n1. 'response': Your main response in natural language\n2. 'suggestions': A list of specific suggestions for improvement"

    full_prompt = f"""
    {json_prompt}
    User input: {user_input}
    Critique: {critic_feedback}
    """

    raw_response, suggestions = generate_response(full_prompt, "")
    log_interaction("Generator", json_prompt, user_input, raw_response)
    return raw_response, suggestions

def critic(prompt, generator_output, suggestions=None):
  # If suggestions are provided, include them in the prompt
  suggestions_str = ", ".join(suggestions) if suggestions else "No suggestions provided."
  json_prompt = f"{prompt}. Incorporate these suggestions in your analysis: {suggestions_str}"

  full_prompt = f"""
  {json_prompt}
  Generator's Output: {generator_output}
  """

  raw_response, suggestions = generate_response(full_prompt, "")
  log_interaction("Critic", json_prompt, generator_output, raw_response)
  return raw_response, suggestions

with gr.Blocks() as demo:
    # Initialize state
    genseg = gr.State(False)
    
    with gr.Row():  # Two-column layout
        with gr.Column():
            gen_prompt = gr.Textbox(label="Generator System Prompt", lines=3, value="Formalize and expand this idea")
            gen_input = gr.Textbox(label="Generator Input", lines=2)
            gen_output = gr.Textbox(label="Generator Output", lines=5)
            gen_suggestions = gr.Textbox(label="Generator Suggestions", lines=2)
            gen_submit = gr.Button("Submit Generator")

        with gr.Column():
            crit_prompt = gr.Textbox(label="Critic System Prompt", lines=3, value="You are a constructive critic, analyze and critique this idea")
            crit_input = gr.Textbox(label="Critic Input", lines=2)
            crit_output = gr.Textbox(label="Critic Output", lines=5)
            crit_suggestions = gr.Textbox(label="Critic Suggestions", lines=2)
            crit_submit = gr.Button("Submit Critic")
    
    # Add history display
    with gr.Row():
        history_display = gr.TextArea(label="Interaction History", lines=10, interactive=False)

    def on_gen_submit(gen_prompt, gen_input, crit_output, genseg_value, history):
        if genseg_value:
            gen_prompt = "Using your brilliant imagination and knowledge, answer these criticisms step-by-step with new ideas that correct or fulfill each criticism. Also suggest improvements for future critiques."
        
        output, suggestions = generator(gen_prompt, gen_input, crit_output)
        critic_input = f"{output}\nSuggestions: {suggestions}"
        
        # Update history
        new_history = f"{history}\n\nGenerator:\nPrompt: {gen_prompt}\nInput: {gen_input}\nOutput: {output}"
        
        return output, gen_prompt, str(suggestions), critic_input, True, new_history

    def on_crit_submit(crit_prompt, crit_input, gen_output, gen_suggestions, history):
        output, suggestions = critic(crit_prompt, gen_output, eval(gen_suggestions))
        new_crit_prompt = "Analyze the revised idea, focusing on new aspects introduced by the generator and provide further critique."
        
        # Update history
        new_history = f"{history}\n\nCritic:\nPrompt: {crit_prompt}\nInput: {crit_input}\nOutput: {output}"
        
        return output, new_crit_prompt, str(suggestions), new_history

    # Event handlers with updated parameters
    gen_submit.click(
        fn=on_gen_submit,
        inputs=[gen_prompt, gen_input, crit_output, genseg, history_display],
        outputs=[gen_output, gen_prompt, gen_suggestions, crit_input, genseg, history_display]
    )
    crit_submit.click(
        fn=on_crit_submit,
        inputs=[crit_prompt, crit_input, gen_output, gen_suggestions, history_display],
        outputs=[crit_output, crit_prompt, crit_suggestions, history_display]
    )

# Launch the interface
demo.launch(debug=True)
import gradio as gr
import openai
import os
#from dotenv import load_dotenv

# Load environment variables if you're using them
#load_dotenv()

# Set the API base to your local LM Studio server
openai.api_base = "http://localhost:1234/v1"
# Ensure you have a valid API key, even if it's not used by LM Studio, some libraries might still check for it
#openai.api_key = os.getenv("OPENAI_API_KEY") or "dummy_key"  # Replace with a real key or use a dummy if not needed
openai.api_key = "dummy_key" 

# Function to create a response from the LLM
def generate_response(model, prompt, input_text, critique=None):
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": input_text}
    ]
    if critique:
        messages.append({"role": "assistant", "content": critique})
    
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=150,
        temperature=0.7
    )
    return response.choices[0].message.content

# Generator function
def generator(prompt, user_input, critic_feedback):
    return generate_response("local_model", prompt, user_input, critic_feedback)

# Critic function
def critic(prompt, generator_output):
    return generate_response("local_model", prompt, generator_output)

# Gradio interface setup
with gr.Blocks() as demo:
    with gr.Row():  # Two-column layout
        with gr.Column():
            gen_prompt = gr.Textbox(label="Generator System Prompt", lines=3, value="Formalize and expand this idea")
            gen_input = gr.Textbox(label="Generator Input", lines=2)
            gen_output = gr.Textbox(label="Generator Output", lines=5)
            gen_submit = gr.Button("Submit Generator")
        
        with gr.Column():
            crit_prompt = gr.Textbox(label="Critic System Prompt", lines=3)
            crit_input = gr.Textbox(label="Critic Input (Optional)", lines=2)
            crit_output = gr.Textbox(label="Critic Output", lines=5)
            crit_submit = gr.Button("Submit Critic")

    def on_gen_submit(gen_prompt, gen_input, crit_output):
        output = generator(gen_prompt, gen_input, crit_output)
        return {gen_output: output}

    def on_crit_submit(crit_prompt, crit_input, gen_output):
        output = critic(crit_prompt, gen_output)
        return {crit_output: output}

    # Event handlers for the buttons
    gen_submit.click(
        fn=on_gen_submit, 
        inputs=[gen_prompt, gen_input, crit_output],
        outputs=[gen_output]
    )
    crit_submit.click(
        fn=on_crit_submit,
        inputs=[crit_prompt, crit_input, gen_output],
        outputs=[crit_output]
    )

# Launch the interface
demo.launch(debug=True)
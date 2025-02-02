# CASI (Cyclical Adversarial Stepwise Improvement)

A Python-based creative AI system that implements a dual-agent approach for idea generation and refinement. CASI uses a Generator and Critic in a feedback loop to iteratively improve ideas.

## Features

- **Generator Agent**: Creates and expands ideas based on user input
- **Critic Agent**: Provides constructive feedback on generated content
- **Iterative Improvement**: Implements a feedback loop between Generator and Critic
- **Configurable Prompts**: Customizable system prompts via `prompts.json`
- **Modern UI**: Built with Gradio for an intuitive user interface

## Requirements

- Python 3.6+
- Gradio
- OpenAI API compatible service (can be used with local LM Studio)

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install gradio openai python-dotenv
   ```
3. Configure your environment variables (optional):
   - `OPENAI_API_BASE`: API endpoint (defaults to local LM Studio)
   - `OPENAI_API_KEY`: Your API key
   - `OPENAI_MODEL`: Model to use

## Usage

1. Run the application:
   ```bash
   python CASI.py
   ```
2. Access the web interface (typically at http://localhost:7860)
3. Enter your initial idea in the Generator input
4. Use the Generator and Critic alternately to refine your idea

## Customization

Edit `prompts.json` to customize the system prompts for both Generator and Critic agents.

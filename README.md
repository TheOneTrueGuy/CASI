# CASI: Cyclical Adversarial Stepwise Improvement

To create a stable loop of adversarial critique that allows an Agent to improve its code/output quality without human intervention.


CASI is a Python application that uses a dual-agent system—a **Generator** and a **Critic**—to iteratively refine ideas, text, or other content. The process is cyclical: the Generator creates, the Critic provides feedback, and the Generator uses that feedback to improve its next output.

This project has been consolidated to use a **Streamlit** web interface as its primary UI. The original Gradio interface and other development files have been moved to the `archive/` directory for historical reference.

---

### Step 1: Install Required Dependencies

1.  **Python Installation**:
    Make sure you have Python 3.8+ installed. You can download it from [python.org](https://python.org).

2.  **Create a Virtual Environment** (Recommended):
    It's best practice to create a virtual environment to manage project dependencies in isolation.

    ```bash
    # Create the virtual environment
    python -m venv venv
    ```

    Activate the virtual environment:
    -   **On Windows**:
        ```powershell
        .\venv\Scripts\Activate.ps1
        ```
    -   **On macOS/Linux**:
        ```bash
        source venv/bin/activate
        ```

3.  **Install Required Packages**:
    With your virtual environment activated, install the necessary packages from the `requirements.txt` file.

    ```bash
    pip install -r requirements.txt
    ```

### Step 2: Set Up Environment Variables

1.  **Create a `.env` File**:
    In the root project directory, create a file named `.env`. This file will securely store your API keys and other configuration variables.

2.  **Add Environment Variables**:
    Open the `.env` file and add the following lines, replacing the placeholder values with your actual credentials. You only need to provide keys for the services you intend to use.

    ```env
    # For OpenAI
    OPENAI_API_KEY="your_openai_api_key_here"
    OPENAI_MODEL="gpt-4-turbo"

    # For Local LLMs via Ollama
    OLLAMA_HOST="http://localhost:11434"
    OLLAMA_MODEL="llama3"

    # For Anthropic
    ANTHROPIC_API_KEY="your_anthropic_api_key_here"
    
    # For Google Gemini
    GOOGLE_API_KEY="your_google_api_key_here"

    # For OpenRouter
    OPENROUTER_API_KEY="your_openrouter_api_key_here"
    ```

### Step 3: Run the Application

With your virtual environment activated and dependencies installed, run the Streamlit application with the following command:

```bash
streamlit run app.py
```

This will launch the CASI application in a new tab in your web browser.

### Features & Usage

#### API Configuration
You can configure your API keys in two ways:
1.  **`.env` File**: Set them persistently as described in Step 2.
2.  **UI Sidebar**: Expand the **"API Configuration"** section in the sidebar to enter or override keys temporarily for your current session.

#### Using OpenRouter
CASI supports **OpenRouter**, allowing access to a wide range of models.
1.  Select **"OpenRouter"** as the service for either the Generator or Critic.
2.  A text input field will appear for the **Model**.
3.  Enter the full OpenRouter model ID (e.g., `anthropic/claude-3-opus`, `meta-llama/llama-3-70b-instruct`, `google/gemini-pro-1.5`).

#### Automatic Loop
1.  Set the **Mode** to "Automatic" for both agents.
2.  Set the **Max Rounds** in the sidebar.
3.  Run the Generator once to start the process. The system will then automatically alternate between Critic and Generator until the max rounds are reached.

### Step 4: Troubleshooting

-   **`ModuleNotFoundError`**: Ensure your virtual environment is activated and you have run `pip install -r requirements.txt`.
-   **API Errors**: Double-check that your `.env` file is correctly formatted, located in the project root, and contains valid API keys for the selected backend. Alternatively, check the keys entered in the sidebar.
-   **`streamlit` command not found**: Verify that your virtual environment is active, as Streamlit was installed there.

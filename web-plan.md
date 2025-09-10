# Plan: Build a Flask-based GUI for CASI

This document outlines the plan to create a web-based user interface for the `CASI.py` module using the Flask web framework. The goal is to build an application that is compatible with WSGI-based hosting platforms like PythonAnywhere.

---

### Step 1: Project Structure

We will create the following files and directories:

```
CASI/
|-- web_app.py            # The main Flask application file.
|-- CASI.py               # The existing backend logic (no changes needed).
|-- requirements.txt      # Will need to add 'Flask'.
|-- templates/
|   |-- index.html        # The HTML template for the user interface.
|-- static/
|   |-- style.css         # (Optional) For custom styling.
|-- .env
|-- archive/
|-- ... (other project files)
```

### Step 2: Flask Application Setup (`web_app.py`)

The initial setup will involve:

1.  **Importing Libraries**: Import `Flask`, `render_template`, `request`, `session`, `os`, and our `CASI as casi` module.
2.  **Initializing Flask**: Create the Flask app instance: `app = Flask(__name__)`.
3.  **Setting a Secret Key**: A secret key is required for Flask's session management, which we will use to track the conversation state. `app.secret_key = os.urandom(24)`.

### Step 3: HTML Template (`templates/index.html`)

We will design the user interface using standard HTML and Jinja2 templating:

1.  **Layout**: Create a two-column layout for the Generator and Critic agents.
2.  **Form**: Wrap all user inputs (text areas, buttons, dropdowns) in a single `<form method="POST">`.
3.  **Dynamic Content**: Use Jinja2 template variables (e.g., `{{ generator_output }}`) to display the data passed from our Flask app.
4.  **Controls**: Add buttons for "Run Generator," "Run Critic," and navigation controls like "Back" and "Forward."

### Step 4: Flask Route and Core Logic (`web_app.py`)

We will create a primary route to handle all user interactions:

```python
@app.route('/', methods=['GET', 'POST'])
def index():
    # Logic will go here
    pass
```

This function will:

-   **On a `GET` request** (first page load): Initialize the conversation state in the Flask `session` and render the `index.html` template.
-   **On a `POST` request** (when a form button is clicked):
    1.  Read the user's input from the `request.form` object.
    2.  Identify which button was pressed to determine the action (e.g., run Generator).
    3.  Call the appropriate function from the `casi` module (e.g., `casi.generator(...)`).
    4.  Update the conversation history stored in the `session` object.
    5.  Re-render the `index.html` template, passing the updated data back to it.

### Step 5: State Management with Flask Sessions

Since Flask is stateless, we must manage the conversation history manually. We will use Flask's `session` object for this:

-   The entire conversation `thread` (a list of state dictionaries) will be stored in `session['thread']`.
-   An index, `session['thread_idx']`, will keep track of the current turn being viewed.
-   This ensures the conversation state persists between user requests.

### Step 6: Future Development

- **User-Provided API Keys**: Implement a feature allowing users to enter their own API keys for LLM services. This provides a 'bring-your-own-key' model, reducing operational costs.
- **Paywalled Service**: As a next step, build a paywalled system where users can purchase credits or a subscription to use the site's centrally-managed API keys. This will be integrated with the existing Stripe payment manager.

### Step 7: Running the Application

We will add the standard boilerplate to make the app runnable:

```python
if __name__ == '__main__':
    app.run(debug=True)
```

This plan provides a clear roadmap for building a robust, deployable web interface for the CASI project.

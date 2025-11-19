# CASI
To create a stable loop of adversarial critique that allows an Agent to improve its code/output quality without human intervention.


Cyclical Adversarial Stepwise Improvement
To ensure that your CASI.py application runs successfully on the first try, you'll need to set up the environment variables and install any necessary dependencies. Here’s a step-by-step guide to help you through the process:
Step 1: Install Required Dependencies

    Python Installation:
    Make sure you have Python installed on your system. You can download it from python.org.

2. Create a Virtual Environment (Optional but Recommended):

    It's a good practice to create a virtual environment for your project to manage dependencies. You can create one using the following commands:
         python -m venv venv
    - Activate the virtual environment:
    On Windows:
           venv\Scripts\activate
    - On macOS/Linux:
           source venv/bin/activate

3. Install Required Packages:

    Install the necessary packages using pip. You can create a requirements.txt file with the following content:
         gradio
         openai
         python-dotenv
    - Then, run the following command to install the packages:
         pip install -r requirements.txt

Step 2: Set Up Environment Variables

    Create a .env File:
    In the same directory as your CASI.py file, create a file named .env. This file will store your environment variables.

2. Add Environment Variables:

    Open the .env file and add the following lines, replacing the placeholder values with your actual OpenAI API credentials:
         OPENAI_API_BASE=http://localhost:1234/v1/chat/completions
         OPENAI_API_KEY=your_openai_api_key_here
         OPENAI_MODEL=your_model_name_here
    - Make sure to replace your_openai_api_key_here with your actual OpenAI API key and your_model_name_here with the model you intend to use (e.g., gpt-3.5-turbo).

Step 3: Run the Application

    Run the Application:
    With your virtual environment activated and the dependencies installed, you can run the application using the following command:
         python CASI.py
    Access the Gradio Interface:
    After running the application, you should see output in the terminal indicating that the Gradio interface is launching. It will typically provide a local URL (e.g., http://127.0.0.1:7860) where you can access the application in your web browser.

Step 4: Troubleshooting

    If you encounter any issues:
    Double-check that all dependencies are installed correctly.
    Ensure that your .env file is correctly formatted and located in the same directory as your CASI.py file.
    Verify that your OpenAI API key is valid and that you have access to the specified model.

Conclusion
By following these steps, you should be able to set up the environment variables and dependencies needed to run your CASI.py application successfully.

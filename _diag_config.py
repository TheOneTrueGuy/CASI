import os
import sys
import dotenv

# Add app directory to path so we can import webCASI
sys.path.append('/home/Guyzer/Firsty/app')

# Load dotenv just like webCASI does
dotenv.load_dotenv('/home/Guyzer/Firsty/.env')

print(f"Raw OPENROUTER_MODEL from os.environ: {os.environ.get('OPENROUTER_MODEL')}")

try:
    from webCASI import config
    print(f"Config object openrouter_model: {config.openrouter_model}")
except ImportError as e:
    print(f"Could not import webCASI: {e}")
except Exception as e:
    print(f"Error accessing config: {e}")

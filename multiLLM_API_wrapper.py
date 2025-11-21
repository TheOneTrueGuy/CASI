import requests
import os
from typing import List, Dict, Optional

class LLMFallbackWrapper:
    def __init__(self, providers: List[Dict[str, str]]):
        """
        providers: List of dicts, e.g., [
            {'name': 'groq', 'api_key': os.getenv('GROQ_API_KEY'), 'model': 'llama3-8b-8192', 'endpoint': 'https://api.groq.com/openai/v1/chat/completions'},
            {'name': 'openrouter', 'api_key': os.getenv('OPENROUTER_API_KEY'), 'model': 'meta-llama/llama-3-8b-instruct:free', 'endpoint': 'https://openrouter.ai/api/v1/chat/completions'}
        ]
        """
        self.providers = providers
        self.current_index = 0

    def call(self, prompt: str, system_prompt: Optional[str] = None, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        messages = [{'role': 'system', 'content': system_prompt}] if system_prompt else []
        messages.append({'role': 'user', 'content': prompt})
        
        for _ in range(len(self.providers)):  # Try each once
            provider = self.providers[self.current_index]
            try:
                headers = {
                    'Authorization': f'Bearer {provider["api_key"]}',
                    'Content-Type': 'application/json'
                }
                data = {
                    'model': provider['model'],
                    'messages': messages,
                    'max_tokens': max_tokens,
                    'temperature': temperature
                }
                response = requests.post(provider['endpoint'], headers=headers, json=data, timeout=30)
                response.raise_for_status()
                return response.json()['choices'][0]['message']['content']
            except Exception as e:
                print(f"Error with {provider['name']}: {e}")
                self.current_index = (self.current_index + 1) % len(self.providers)
        
        raise Exception("All LLM providers failed—check keys/rates.")

# Example setup (add your real keys via env)
providers = [
    {'name': 'groq', 'api_key': os.getenv('GROQ_API_KEY'), 'model': 'llama3-8b-8192', 'endpoint': 'https://api.groq.com/openai/v1/chat/completions'},
    {'name': 'openrouter', 'api_key': os.getenv('OPENROUTER_API_KEY'), 'model': 'meta-llama/llama-3-8b-instruct:free', 'endpoint': 'https://openrouter.ai/api/v1/chat/completions'},
    {'name': 'deepseek', 'api_key': os.getenv('DEEPSEEK_API_KEY'), 'model': 'deepseek-chat', 'endpoint': 'https://api.deepseek.com/v1/chat/completions'}  # Add more
]
wrapper = LLMFallbackWrapper(providers)

# Usage in 1nventory (e.g., for object description tool)
desc = wrapper.call("Describe this object from the frame: [image data]", system_prompt="You are an inventory AI.")
print(desc)
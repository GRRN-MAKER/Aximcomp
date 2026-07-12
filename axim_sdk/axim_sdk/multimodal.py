import logging
import ollama
import os
import json
import urllib.request
import base64
import ssl
import certifi

try:
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import get_message_profile
    from mlx_vlm.utils import load_config
    MLX_VLM_AVAILABLE = True
except ImportError:
    MLX_VLM_AVAILABLE = False

logger = logging.getLogger(__name__)

def _get_mistral_api_key():
    # Try getting from environment first
    key = os.environ.get("MISTRAL_API_KEY")
    return key

class AximMultimodalEngine:
    """
    Axim custom wrapper for local AI models via Ollama, native MLX-VLM, and Mistral API.
    Provides a unified API to handle Text and Images natively on any hardware (Intel or Apple Silicon).
    """
    def __init__(self, model_name="magnus"):
        self.model_name = model_name
        self.is_mlx = "mlx" in model_name.lower() or "qwen2-vl" in model_name.lower()
        self.is_mistral_api = "mistral" in model_name.lower()
        
        if self.is_mlx:
            logger.info(f"Initializing Axim Engine natively with MLX model: {model_name}...")
            if not MLX_VLM_AVAILABLE:
                raise ImportError("mlx-vlm is not installed. Please run 'pip install mlx-vlm'")
            # Load MLX model and processor
            self.model, self.processor = load(self.model_name)
            self.config = load_config(self.model_name)
        elif self.is_mistral_api:
            logger.info(f"Initializing Axim Engine connected directly to Mistral API...")
            self.mistral_api_key = _get_mistral_api_key()
            if not self.mistral_api_key:
                raise ValueError("Mistral API key not found. Please set MISTRAL_API_KEY.")
        else:
            logger.info(f"Initializing Axim Engine connected to Magnus Cloud API: {model_name}...")
            self.magnus_url = os.environ.get("MAGNUS_API_URL", "https://grrnmaker-magnus-api.hf.space/v1/chat/completions")

    def ask(self, prompt: str = None, image_paths: list = None, messages: list = None, **kwargs):
        """
        Process a prompt with optional images.
        
        Args:
            prompt (str): The text prompt.
            image_paths (list): List of paths to image files.
            messages (list): Optional full chat history.
            **kwargs: Extra generation parameters (temperature, num_predict, etc.)
        """
        images = image_paths if image_paths else []

        if self.is_mlx:
            return self._ask_mlx(prompt, images, **kwargs)
        elif self.is_mistral_api:
            return self._ask_mistral_api(prompt, images, **kwargs)
        else:
            return self._ask_ollama(prompt, images, messages=messages, **kwargs)

    def _ask_mistral_api(self, prompt: str, images: list, **kwargs):
        """
        Direct native HTTP call to Mistral API without wrappers.
        """
        url = "https://api.mistral.ai/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.mistral_api_key}"
        }

        # If there are images, we should use pixtral model
        target_model = "pixtral-12b-2409" if images else "mistral-large-latest"

        content = []
        content.append({"type": "text", "text": prompt})

        if images:
            for img_path in images:
                with open(img_path, "rb") as img_file:
                    base64_image = base64.b64encode(img_file.read()).decode('utf-8')
                    # basic mime type guessing based on extension
                    mime_type = "image/jpeg"
                    if img_path.lower().endswith(".png"): mime_type = "image/png"
                    elif img_path.lower().endswith(".webp"): mime_type = "image/webp"
                    
                    content.append({
                        "type": "image_url",
                        "image_url": f"data:{mime_type};base64,{base64_image}"
                    })

        payload = {
            "model": target_model,
            "messages": [{"role": "user", "content": content}],
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("num_predict", 512)
        }

        req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers=headers, method="POST")
        try:
            # Fix macOS Python SSL verify issues by explicitly pointing to certifi
            context = ssl.create_default_context(cafile=certifi.where())
            with urllib.request.urlopen(req, context=context) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8')
            logger.error(f"Mistral API Error: {e.code} - {error_body}")
            raise Exception(f"Mistral API Error: {e.code} - {error_body}")

    def _ask_mlx(self, prompt: str, images: list, **kwargs):
        """
        Direct native MLX generation without wrappers.
        """
        # Determine the maximum number of tokens to generate
        max_tokens = kwargs.get("num_predict", 512)
        temperature = kwargs.get("temperature", 0.7)
        
        # Apply the chat template/processor based on mlx-vlm's standard method
        if hasattr(self.processor, "apply_chat_template"):
            # For Qwen2-VL, we need to structure it with vision inputs
            messages = [{"role": "user", "content": prompt}]
            if images:
                # Some processors expect specific formatting for images inline
                content = []
                for img in images:
                    content.append({"type": "image", "image": img}) # mlx-vlm expects image path or url
                content.append({"type": "text", "text": prompt})
                messages = [{"role": "user", "content": content}]
                
            formatted_prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True
            )
        else:
            # Fallback for models that do not use apply_chat_template
            formatted_prompt = prompt
            
        response = generate(
            self.model,
            self.processor,
            prompt=formatted_prompt,
            image_processor=self.processor.image_processor if hasattr(self.processor, "image_processor") else None,
            image=images,
            max_tokens=max_tokens,
            temperature=temperature,
            verbose=False
        )
        return response

    def _ask_ollama(self, prompt: str, images: list, messages: list = None, **kwargs):
        """
        Direct generation via Hugging Face Space endpoint instead of local Ollama daemon.
        """
        if messages is None:
            # Construct the message format expected by the API
            messages = [{
                'role': 'user',
                'content': prompt,
            }]
            
            # If images are provided, attach them to the message payload
            if images:
                messages[0]['images'] = images

        # Merge user kwargs with safe defaults for strict instruction following
        options = {
            "num_predict": kwargs.get("num_predict", 128),
            "temperature": kwargs.get("temperature", 0.1),
            "top_p": kwargs.get("top_p", 0.9),
            "repetition_penalty": kwargs.get("repetition_penalty", 1.1),
            "do_sample": kwargs.get("do_sample", False)
        }

        # Generate response using the Magnus Hugging Face endpoint
        url = self.magnus_url
        headers = {
            "Content-Type": "application/json"
        }
        payload = {
            "model": "magnus", # Use a generic name, server usually defaults to the only loaded model
            "messages": messages,
            "max_tokens": options["num_predict"],
            "temperature": options["temperature"]
        }
        
        req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers=headers, method="POST")
        try:
            context = ssl.create_default_context(cafile=certifi.where())
            with urllib.request.urlopen(req, context=context) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result['choices'][0]['message']['content']
        except Exception as e:
            logger.error(f"Magnus API Error: {e}")
            raise Exception(f"Connection error to Magnus HF endpoint. Underlying error: {e}")


from typing import List, Dict, Any, Optional
import os
import json
from .tools.bash import BashTool
from .tools.predictor import PredictionTool
from .shell import Shell
from .memory import AximMemory
from .turboquant import TurboQuant
from dotenv import load_dotenv
from .multimodal import AximMultimodalEngine

try:
    from mistralai.client.sdk import Mistral
except ImportError:
    Mistral = None

load_dotenv()

class Agent:
    """
    The Axim Agent orchestrator.
    Manages state, tools, and interaction logic.
    """
    
    def __init__(self, name: str = "Axim", shell: Optional[Shell] = None):
        self.name = name
        self.shell = shell or Shell()
        
        # Load API key explicitly as fallback if needed
        api_key = os.environ.get("MISTRAL_API_KEY")
        # Initialize client with explicitly increased timeout (in ms) for long prediction tasks
        self.client = Mistral(api_key=api_key, timeout_ms=600000) if Mistral and api_key else None
        
        # We now use the direct AximMultimodalEngine for OSS instead of openai wrappers
        self.oss_engine = AximMultimodalEngine(model_name=os.environ.get("OSS_MODEL_NAME", "magnus"))

        self.tools = {
            "bash": BashTool(self.shell),
            "predict": PredictionTool(self._chat_complete),
        }
        
        self.history: List[Dict[str, Any]] = []
        self.model = "mistral-large-latest"
        
        # Initialize Axim Memory (Local Vector Store replacing third-party wrappers)
        self.axim_memory = AximMemory(self.client)
        
        # Initialize TurboQuant Compressor (Mistral-embed returns 1024-dim vectors)
        # Using 128 bit target dimension for Extreme Compression
        self.compressor = TurboQuant(original_dim=1024, target_dim=128)
        self.compressed_memory: List[Dict[str, Any]] = []
        
        # User-selected active provider ("mistral" or "oss")
        # Default to "oss" (Magnus) as requested
        self.active_provider = "oss"
        
        # Base system prompt reused per request to avoid ordering issues
        self.base_messages = [
            {"role": "system", "content": "You are Axim, a high-performance AI assistant. Do not explain unless explicitly asked. Keep responses minimal, professional, and concise. You have access to a bash terminal. NEVER tell the user to run a command manually; ALWAYS use your `bash` tool to run it yourself. If you need to install packages, use `pip install --break-system-packages <pkg>`. Do not decline requests for visual or system modifications; write scripts to perform them."}
        ]

    def _chat_complete(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None, tool_choice: Optional[str] = None) -> Any:
        """
        Wrapper to handle API calls with explicit routing between Mistral and GPT-OSS (OpenAI local fallback).
        """
        if self.active_provider == "mistral":
            try:
                if not self.client:
                    raise Exception("Mistral client is not initialized.")
                    
                kwargs = {
                    "model": self.model,
                    "messages": messages,
                }
                if tools:
                    kwargs["tools"] = tools
                    if tool_choice:
                        kwargs["tool_choice"] = tool_choice
                        
                return self.client.chat.complete(**kwargs)
                
            except Exception as e:
                print(f"\n\033[38;5;160m[Fallback Triggered]\033[0m Mistral failed ({e}). Falling back to Magnus Cloud API.")
                self.active_provider = "oss"
                return self._chat_complete(messages, tools, tool_choice)

        elif self.active_provider == "oss":
            try:
                # Add <|think|> to the first message if needed for reasoning models
                if "magnus" in self.oss_engine.model_name.lower():
                    if len(messages) > 0 and not messages[0]['content'].startswith("<|think|>"):
                        messages[0]['content'] = "<|think|>\n" + messages[0]['content']

                response = self.oss_engine.ask(prompt=None, messages=messages)
                
                # Wrap the raw text string into a mock object so the CLI can read it
                class MockMessage:
                    def __init__(self, c): self.content = c; self.tool_calls = None
                class MockChoice:
                    def __init__(self, m): self.message = m
                class MockResponse:
                    def __init__(self, choices): self.choices = choices
                
                return MockResponse([MockChoice(MockMessage(response))])

            except Exception as e:
                raise Exception(f"Connection error to Magnus Cloud API. Underlying error: {e}")

        else:
            raise Exception("Invalid active provider.")

    def handle_request(self, request: str) -> Dict[str, Any]:
        """
        Process a user request and determine the next action.
        """
        req_lower = request.lower().strip()
        
        # Handle explicit model switching commands
        # Accept more flexible phrasing for switching to OSS/local models
        if req_lower in ["switch to oss", "switch to gemma", "switch oss", "use oss", "use gemma", "switch to magnus"]:
            if self.oss_engine:
                self.active_provider = "oss"
                return {"status": "success", "output": "Switched to Magnus Cloud API natively.", "cwd": self.tools["bash"].cwd, "command": None}
            else:
                return {"status": "error", "message": "Failed to switch: Magnus API client not configured properly."}
        
        if req_lower in ["switch to mistral", "use mistral", "switch mistral"]:
            if not self.client:
                return {
                    "status": "error",
                    "message": "Mistral API Key not found. Please provide your Mistral API Key to switch to Mistral model.",
                    "require_key": "mistral"
                }
            self.active_provider = "mistral"
            return {"status": "success", "output": "Switched to Mistral model.", "cwd": self.tools["bash"].cwd, "command": None}
        
        # Fallback if no AI client is set up
        if not self.client and not self.oss_engine:
            if request.startswith("run "):
                command = request[4:]
                result = self.tools["bash"].execute(command)
                self.history.append({
                    "type": "command_result",
                    "content": result,
                    "request": request
                })
                return result
            
            return {
                "status": "error",
                "message": "AI engines are not configured properly. Prefix commands with 'run ' to execute directly."
            }

        # Build a fresh message list per request
        messages = list(self.base_messages)
        
        # Inject recent conversational history directly to maintain flow (last 10 messages)
        recent_history = self.history[-10:] if len(self.history) > 10 else self.history
        for msg in recent_history:
            # Reconstruct the messages based on what we saved in history
            if msg["type"] == "user":
                messages.append({"role": "user", "content": msg["content"]})
            elif msg["type"] == "assistant":
                messages.append({"role": "assistant", "content": msg["content"]})
        
        request_context = request
        try:
            # Multi-Layer Memory Execution: AximMemory (Long-term) + TurboQuant (Compressed Cache)
            # Only trigger this heavy remote operation when the active provider is Mistral
            if self.client and self.active_provider == "mistral":
                # 1. Fetch from AximMemory (Mem0 alternative)
                relevant_memories = self.axim_memory.search(request, limit=3, threshold=0.4)
                
                # 2. Fetch from TurboQuant compressed cache
                embed_response = self.client.embeddings.create(model="mistral-embed", inputs=[request])
                raw_vector = embed_response.data[0].embedding
                compressed_req = self.compressor.compress(raw_vector)
                
                tq_relevant_past = []
                historical_memory = self.compressed_memory[:-5] if len(self.compressed_memory) > 5 else []
                
                for past_mem in historical_memory:
                    sim = self.compressor.compute_similarity(compressed_req["qjl_bits"], past_mem["compressed"]["qjl_bits"])
                    if sim > 0.4:
                        tq_relevant_past.append(past_mem["text"])
                        
                combined_context = []
                if relevant_memories:
                    combined_context.append("### Persistent Axim Memory:\n" + "\n".join([mem["text"] for mem in relevant_memories]))
                if tq_relevant_past:
                    combined_context.append("### Compressed TurboQuant Cache:\n" + "\n".join(tq_relevant_past))
                    
                if combined_context:
                    request_context = "\n\n".join(combined_context) + f"\n\nCurrent request: {request}"
                
                # Save the new user request to both memory layers
                self.axim_memory.add(request)
                self.compressed_memory.append({
                    "text": request,
                    "compressed": compressed_req
                })
        except Exception as mem_err:
            pass # Fallback to normal behavior if memory fails
            
        messages.append({"role": "user", "content": request_context})
        
        # Save this user request to history
        self.history.append({"type": "user", "content": request})

        tools = [{
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Execute a bash command in a stateful shell.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The bash command to execute."
                        }
                    },
                    "required": ["command"]
                }
            }
        }, {
            "type": "function",
            "function": {
                "name": "predict",
                "description": "Run a multi-agent swarm intelligence simulation to predict outcomes.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "scenario": {
                            "type": "string",
                            "description": "The scenario or topic to predict and analyze."
                        }
                    },
                    "required": ["scenario"]
                }
            }
        }]

        try:
            response = self._chat_complete(
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )

            response_message = response.choices[0].message
            messages.append(response_message)

            if response_message.tool_calls:
                # The model can return multiple tool calls in a single response.
                # We need to process all of them and append a tool response for each.
                for tool_call in response_message.tool_calls:
                    if tool_call.function.name == "bash":
                        args = json.loads(tool_call.function.arguments)
                        command = args["command"]
                        
                        print(f"\n\033[38;5;28m[Running Command]\033[0m {command}")
                        
                        result = self.tools["bash"].execute(command)
                        
                        # Handle result which is a dict in bash tool
                        out_text = str(result.get("output", result)) if isinstance(result, dict) else str(result)
                        out_preview = out_text.strip()
                        if len(out_preview) > 300:
                            out_preview = out_preview[:300] + "\n...(Output truncated for preview)"
                        print(f"\033[90m{out_preview}\033[0m")
                        
                        tool_result_content = json.dumps(result)
                        
                        messages.append({
                            "role": "tool",
                            "name": "bash",
                            "content": tool_result_content,
                            "tool_call_id": tool_call.id
                        })
                    elif tool_call.function.name == "predict":
                        args = json.loads(tool_call.function.arguments)
                        scenario = args["scenario"]
                        
                        print(f"\n\033[38;5;28m[Running Prediction]\033[0m Scenario: {scenario[:50]}...")
                        
                        result = self.tools["predict"].execute(scenario)
                        tool_result_content = json.dumps(result)
                        
                        messages.append({
                            "role": "tool",
                            "name": "predict",
                            "content": tool_result_content,
                            "tool_call_id": tool_call.id
                        })
                
                # After resolving all tool calls, get the final response from the model
                final_response = self._chat_complete(
                    messages=messages
                )
                final_message = final_response.choices[0].message
                messages.append(final_message)
                
                # Save assistant response to history
                self.history.append({"type": "assistant", "content": final_message.content})
                
                return {
                    "status": "success",
                    "output": final_message.content,
                    "cwd": self.tools["bash"].cwd,
                    "command": "Multiple commands executed" if len(response_message.tool_calls) > 1 else json.loads(response_message.tool_calls[0].function.arguments).get("command", ""),
                    "tool_output": "Check individual commands"
                }
            
            # Save standard assistant response to history
            self.history.append({"type": "assistant", "content": response_message.content})
            
            return {
                "status": "success",
                "output": response_message.content,
                "cwd": self.tools["bash"].cwd,
                "command": None
            }

        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

    def set_keys(self, mistral_key: Optional[str] = None, hf_key: Optional[str] = None, magnus_url: Optional[str] = None):
        """
        Dynamically sets API keys for the agent and its engines.
        """
        if magnus_url and self.oss_engine:
            self.oss_engine.magnus_url = magnus_url

        if mistral_key:
            os.environ["MISTRAL_API_KEY"] = mistral_key
            if Mistral:
                self.client = Mistral(api_key=mistral_key, timeout_ms=600000)
                # Re-initialize tools that depend on chat_complete
                self.tools["predict"] = PredictionTool(self._chat_complete)
                # Re-initialize memory
                self.axim_memory = AximMemory(self.client)
            
            if self.oss_engine and hasattr(self.oss_engine, 'is_mistral_api') and self.oss_engine.is_mistral_api:
                self.oss_engine.mistral_api_key = mistral_key

        if hf_key:
            os.environ["HUGGINGFACE_TOKEN"] = hf_key
            # Magnus engine uses HF token if needed
            if self.oss_engine:
                # If we add token support to Magnus engine, set it here
                pass

    def save_state(self, path: str):
        """
        Saves the agent's current state to a file.
        """
        state = {
            "name": self.name,
            "cwd": self.tools["bash"].cwd,
            "history": self.history
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=4)

    def load_state(self, path: str):
        """
        Loads the agent's state from a file.
        """
        if not os.path.exists(path):
            return
        with open(path, "r") as f:
            state = json.load(f)
            self.name = state.get("name", self.name)
            self.tools["bash"].cwd = state.get("cwd", os.getcwd())
            self.history = state.get("history", [])

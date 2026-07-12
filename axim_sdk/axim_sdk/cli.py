import sys
import time
from axim_sdk.agent import Agent
from axim_sdk.shell import Shell
from rich.console import Console
from rich.markdown import Markdown

def main():
    agent = Agent()
    console = Console()
    
    deep_green = "\033[38;5;28m"
    red = "\033[38;5;160m"
    reset = "\033[0m"
    dim = "\033[2m"
    
    print(f"{deep_green}╭───────────────────────────────────────────────────╮{reset}")
    print(f"{deep_green}│{reset}      {deep_green}GRRN{reset} - A Fintech by {deep_green}HERNS INC{reset}           {deep_green}│{reset}")
    print(f"{deep_green}│{reset}                                                   {deep_green}│{reset}")
    print(f"{deep_green}│{reset}                    {deep_green}Axim Engine{reset}                    {deep_green}│{reset}")
    print(f"{deep_green}│{reset}         {dim}Axim uses AI. Check for mistakes.{reset}         {deep_green}│{reset}")
    print(f"{deep_green}╰───────────────────────────────────────────────────╯{reset}")
    
    print(f"{dim}Model: {agent.model} (Active Engine: {agent.active_provider.upper()}){reset}")
    
    while True:
        try:
            cwd_display = agent.tools['bash'].cwd
            print(f"\n{dim}workspace (/directory){reset}")
            print(f"{cwd_display}\n")
            
            user_input = input(f"{deep_green}>{reset} Type your message or @path/to/file: ")
            if not user_input.strip():
                continue
            if user_input.lower() in ["exit", "quit"]:
                break
                
            with console.status(f"[bold green]Axim is thinking...", spinner="dots"):
                result = agent.handle_request(user_input)

            if result["status"] == "success":
                output = result['output']
                
                # Check for Thinking tags from Magnus / reasoning models
                if "<|channel>thought" in output:
                    parts = output.split("<channel|>", 1)
                    thought = parts[0].replace("<|channel>thought", "").strip()
                    final_ans = parts[1].strip() if len(parts) > 1 else ""
                    
                    print(f"\n{dim}Agent's Thought Process:{reset}")
                    print(f"{dim}{thought}{reset}")
                    print(f"\n{deep_green}Axim:{reset}")
                    console.print(Markdown(final_ans))
                else:
                    print(f"\n{deep_green}Axim:{reset}")
                    console.print(Markdown(output))
            elif result["status"] == "denied":
                print(f"\n{deep_green}Warning:{reset} {result['message']}")
            else:
                print(f"\n{red}Error:{reset} {result['message']}")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n{red}An unexpected error occurred:{reset} {e}")

if __name__ == "__main__":
    main()

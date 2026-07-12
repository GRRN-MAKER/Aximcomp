from axim_sdk.agent import Agent

def main():
    agent = Agent()
    print("Initial Provider:", agent.active_provider)
    
    # Force switch to Gemma
    res = agent.handle_request("switch to gemma")
    print(res['output'])
    
    print("\nTesting Axim Engine via Gemma 4:")
    res = agent.handle_request("Say 'Axim Agent Gemma Core activated!'")
    print(res['output'])

if __name__ == "__main__":
    main()

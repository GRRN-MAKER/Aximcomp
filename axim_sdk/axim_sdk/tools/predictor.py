import json
from typing import Dict, Any, List
import time
from axim_sdk.shell import Shell

class PredictionTool:
    """
    A multi-agent swarm intelligence prediction engine inspired by Axim Prediction.
    It spins up different AI personas, runs a simulated timeline of events where 
    agents react to each other, builds a knowledge graph of facts, and generates 
    a comprehensive analytical report.
    """
    def __init__(self, chat_fn):
        self.name = "predict"
        self.description = "Run a multi-agent swarm intelligence simulation to predict outcomes of a scenario or analyze public opinion. Usage: {\"scenario\": \"scenario description\"}"
        self.chat_fn = chat_fn
        
    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        if not self.chat_fn:
            return "Error: Chat function not initialized."
        try:
            response = self.chat_fn(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error during LLM call: {e}"

    def execute(self, scenario: str) -> Dict[str, Any]:
        if not self.chat_fn:
            return {"status": "error", "message": "LLM client is required for prediction simulations."}
            
        print(f"\n\033[96m[Axim Prediction Engine]\033[0m Initializing swarm intelligence simulation for: '{scenario[:50]}...'")
        
        # 1. Graph Building (Entity Extraction)
        print("\033[90m  -> Step 1: Building Entity Graph...\033[0m")
        entity_prompt = "Identify the core entities (people, companies, countries, concepts) involved in this scenario. Return as a comma-separated list."
        entities = self._call_llm("You are an Entity Extraction AI.", f"Scenario: {scenario}\n\n{entity_prompt}")
        
        # 2. Environment Setup (Personas)
        print("\033[90m  -> Step 2: Spinning up Agent Personas...\033[0m")
        personas = [
            {"role": "Financial/Market Analyst", "focus": "Economic impact, market trends, and financial risks."},
            {"role": "Sociologist/Public Opinion Expert", "focus": "Public reaction, social behavior, and cultural shifts."},
            {"role": "Policy/Strategy Maker", "focus": "Regulatory changes, political implications, and governance."},
            {"role": "Tech/Industry Visionary", "focus": "Technological disruption, innovation, and digital transformation."}
        ]
        
        # 3. Simulation (Multi-Round Debate / Timeline Evolution)
        print("\033[90m  -> Step 3: Running Simulation Timeline...\033[0m")
        timeline = []
        current_state = scenario
        
        for round_num in range(1, 3):  # Simulate 2 time steps (e.g. Short-term, Long-term)
            print(f"\033[90m     [Round {round_num}] Simulating agent interactions...\033[0m")
            round_events = []
            
            for p in personas:
                prompt = f"Current State: {current_state}\n\nBased on your focus ({p['focus']}), what is the immediate next action, reaction, or consequence that occurs? Describe it in 1-2 sentences as a factual event."
                event = self._call_llm(f"You are a {p['role']}.", prompt)
                round_events.append(f"[{p['role']}] {event}")
            
            # Aggregate events to form the new state
            timeline.append(f"Round {round_num} Events:\n" + "\n".join(round_events))
            current_state += "\n" + "\n".join(round_events)
            
        # 4. Report Generation (ReportAgent)
        print("\033[90m  -> Step 4: Generating Consensus Report...\033[0m")
        
        report_sys_prompt = """You are the Central Report Agent (inspired by Axim Prediction).
Synthesize the simulated timeline and agent predictions into a professional 'Prediction Report'.
Format your response using Markdown with the following structure:
# Axim Prediction Report
## Executive Summary
## Key Entities & Dynamics
## Simulated Timeline (Short-term vs Long-term)
## Most Likely Outcome & Risks
"""
        report_user_prompt = f"Original Scenario: {scenario}\n\nIdentified Entities: {entities}\n\nSimulation Timeline:\n{chr(10).join(timeline)}\n\nGenerate the comprehensive report."
        
        final_report = self._call_llm(report_sys_prompt, report_user_prompt)
        
        print("\033[92m[Axim Prediction Engine]\033[0m Simulation Complete!")
        
        return {
            "status": "success",
            "output": final_report,
            "agent_logs": timeline
        }

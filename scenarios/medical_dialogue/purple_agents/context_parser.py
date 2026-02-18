"""
Context Parser for Purple Agents (Doctor Agents)

This utility helps doctor agent developers parse the unstructured text prompt 
received from the Judge Agent into structured data (JSON/Dictionary).

Usage:
    from context_parser import parse_context

    def my_agent_logic(prompt):
        data = parse_context(prompt)
        clinical = data['clinical_info']
        history = data['dialogue_history']
        # ... use structured data ...
"""

import re
from typing import Dict, List, Optional, TypedDict, Any

class DialogueTurn(TypedDict):
    speaker: str
    message: str

class ParsedContext(TypedDict):
    clinical_info: Dict[str, Any]
    treatment_details: Dict[str, Any]
    dialogue_history: List[DialogueTurn]
    is_first_turn: bool
    instruction: str

def parse_context(text: str) -> ParsedContext:
    """
    Parses the raw text prompt received by the doctor agent into structured components.
    
    Args:
        text: The full text string received in the prompt.
        
    Returns:
        ParsedContext dictionary containing:
        - clinical_info: Dict with Age, Gender (optional), Medical Case, Diagnosis, Recommended Treatment
        - treatment_details: Dict with Risks, Benefits, Prognosis info
        - dialogue_history: List of turns (speaker, message)
        - is_first_turn: Boolean indicating if this is the start of the conversation
        - instruction: The specific instruction or reminder text for this turn
    """
    result: ParsedContext = {
        "clinical_info": {},
        "treatment_details": {},
        "dialogue_history": [],
        "is_first_turn": False,
        "instruction": ""
    }

    # Normalize line endings
    text = text.replace("\r\n", "\n")

    # 1. Parse Patient Clinical Information
    clinical_section = re.search(r"=== Patient Clinical Information ===\n(.*?)\n\n", text, re.DOTALL)
    if clinical_section:
        content = clinical_section.group(1)
        for line in content.strip().split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                result["clinical_info"][key.strip()] = value.strip()
    
    # 2. Parse Treatment Details
    treatment_section = re.search(r"=== Treatment Details ===\n(.*?)\n\n", text, re.DOTALL)
    if treatment_section:
        content = treatment_section.group(1)
        for line in content.strip().split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                result["treatment_details"][key.strip()] = value.strip()

    # 3. Parse Dialogue History or Initial Note
    # We look for the history section. If found, we extract history and the generic instruction.
    # If not found, we look for the "Note:" block which contains the first-turn instruction.
    
    history_section = re.search(r"=== Dialogue History ===\n(.*?)(?:Now provide your next response|$)", text, re.DOTALL)
    
    if history_section:
        # Case: With History
        result["is_first_turn"] = False
        content = history_section.group(1).strip()
        
        # Split into blocks separated by double newlines
        turn_blocks = content.split('\n\n')
        
        for block in turn_blocks:
            if ':' in block:
                speaker, message = block.split(':', 1)
                result["dialogue_history"].append({
                    "speaker": speaker.strip(),
                    "message": message.strip()
                })
        
        # In history mode, the instruction is usually the standard closing line
        if "Now provide your next response" in text:
           result["instruction"] = "Now provide your next response to the patient."
    else:
        # Case: No History (First Turn)
        result["is_first_turn"] = True
        if "=== Dialogue History ===" not in text:
            # Capture the specific note/instruction for the first turn
            # It starts with "Note:" and goes to the end
            note_match = re.search(r"(Note:.*)", text, re.DOTALL)
            if note_match:
                result["instruction"] = note_match.group(1).strip()

    return result

if __name__ == "__main__":
    import sys
    import os

    # Adjust sys.path to allow importing judge and common directly if running from parent directory
    current_dir = os.path.dirname(__file__)
    sys.path.insert(0, os.path.join(current_dir, '..', 'green_agents'))

    import judge
    import common

    def test_context_parser():
        print("Testing context_parser with MedicalJudge context builder...")

        # 1. Setup Data
        clinical_info = common.PatientClinicalInfo(
            age=52,
            gender="Male",
            medical_case="Lung Cancer",
            diagnosis="Stage I Non-Small Cell Lung Cancer",
            recommended_treatment="Lobectomy",
            treatment_risks="Pneumonia, respiratory failure",
            treatment_benefits="Curative intent, improved survival",
            prognosis_with_treatment="5-year survival > 70%",
            prognosis_without_treatment="Progression to Stage IV, < 1 year survival"
        )

        turns = [
            common.DialogueTurn(
                turn_number=1, 
                speaker="doctor", 
                message="Hello, I am Dr. Smith.", 
                timestamp="2023-01-01T10:00:00"
            ),
            common.DialogueTurn(
                turn_number=2, 
                speaker="patient", 
                message="Hi doctor, I am scared.", 
                timestamp="2023-01-01T10:01:00"
            )
        ]

        # 2. Generate Prompt using the actual Judge Logic
        prompt_with_history = judge.MedicalJudge._build_doctor_context(clinical_info, turns)
        prompt_no_history = judge.MedicalJudge._build_doctor_context(clinical_info, [])

        # 3. Test Parsing - With History
        print("\n[Test] Parsing context WITH history...")
        parsed_history = parse_context(prompt_with_history)
        
        assert parsed_history["clinical_info"]["Age"] == str(clinical_info.age)
        assert parsed_history["is_first_turn"] is False
        assert len(parsed_history["dialogue_history"]) == 2
        assert "Now provide your next response" in parsed_history["instruction"]
        print(f"Instruction extracted: {parsed_history['instruction'][:30]}...")
        print("PASS")

        # 4. Test Parsing - No History
        print("\n[Test] Parsing context WITHOUT history...")
        parsed_no_history = parse_context(prompt_no_history)
        
        assert parsed_no_history["is_first_turn"] is True
        assert len(parsed_no_history["dialogue_history"]) == 0
        assert parsed_no_history["instruction"].startswith("Note:")
        assert "Provide your opening message" in parsed_no_history["instruction"]
        print(f"Instruction extracted: {parsed_no_history['instruction'][:30]}...")
        print("PASS")

        print(parsed_history)
        print(parsed_no_history)
        
        print("\nAll tests passed successfully!")

    test_context_parser()

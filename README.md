# OSCE-Project

## Generative Adversarial Agents System for evaluating Objective Structured Clinical Examination capabilities

A **Generative Adversarial Agents (GAA)** system that evaluates medical dialogue agents through realistic doctor-patient consultations in standardized clinical examinations. The system uses adversarial patient agents with diverse personalities to rigorously assess doctor agents' clinical communication abilities.

### Key Features

- 🏥 **Medical Dialogue Evaluation** - Evaluates doctor agents' communication and persuasion abilities
- 🧠 **64 Patient Personas** - 16 MBTI personality types × 2 medical conditions × 2 genders
- 📊 **Multi-Dimensional Scoring** - Real-time evaluation of empathy, persuasion, and patient safety
- 🔬 **Information Asymmetry** - Doctor receives only clinical data; patient personality and symptoms remain hidden
- ✅ **Reproducible** - Built on [AgentBeats](https://agentbeats.dev) platform using A2A protocol

## Quickstart

1. Clone the repo

```bash
git clone https://github.com/MadGAA-Lab/OSCE-Project.git
cd OSCE-Project
```

2. Install dependencies

```bash
uv sync
```

3. Set environment variables

```bash
cp sample.env .env
```

Add your API credentials to the `.env` file (supports OpenAI, Anthropic, Google Gemini, etc.)

4. Run evaluation

```bash
uv run agentbeats-run scenarios/medical_dialogue/scenario.toml
```

**Note:** Use `--show-logs` to see agent outputs during the assessment, and `--serve-only` to start agents without running the assessment.

After running, you should see dialogue rounds and evaluation scores:

![Sample output](assets/sample_output.png)

## Project Structure

```
src/agentbeats/              # Core A2A infrastructure
  ├─ green_executor.py       # Base green agent executor
  ├─ models.py               # Pydantic models for agent IO
  ├─ client.py               # A2A messaging helpers
  └─ run_scenario.py         # Scenario runner

scenarios/medical_dialogue/  # Medical dialogue evaluation
  ├─ green_agents/
  │  ├─ judge.py             # Orchestrates doctor-patient dialogue
  │  ├─ patient_agent.py     # Simulates patient with personality
  │  ├─ patient_constructor.py # Generates patient personas (MBTI)
  │  ├─ per_round_scoring.py # Evaluates empathy, persuasion, safety
  │  └─ report_generator.py  # Creates performance reports
  ├─ purple_agents/
  │  └─ doctor_agent.py      # Doctor agent being evaluated
  ├─ prompts/                # MBTI traits & medical cases
  └─ scenario.toml           # Evaluation configuration
```

## Medical Dialogue Evaluation

### Patient Personas

- **16 MBTI Types**: INTJ, INTP, ENTJ, ENTP, INFJ, INFP, ENFJ, ENFP, ISTJ, ISFJ, ESTJ, ESFJ, ISTP, ISFP, ESTP, ESFP
- **2 Medical Cases**: Pneumothorax, Lung Cancer
- **2 Genders**: Male, Female (optional)

### Round-Based Evaluation Process

1. **Doctor** sends response to patient
2. **Patient** generates personality-driven response
3. **Judge** evaluates the round:
   - Empathy Score (0-10)
   - Persuasion Score (0-10)
   - Safety Score (0-10)
4. **Stop Conditions**: Patient left / accepted treatment / max rounds reached

### Information Asymmetry Design

**Doctor receives:**

- Age, gender (if specified)
- Diagnosis and recommended treatment
- Treatment risks, benefits, and prognosis

**Doctor does NOT receive:**

- Patient symptoms (must discover through dialogue)
- Patient personality traits (MBTI)
- Patient concerns and fears
- Patient behavioral patterns

This mirrors real medical practice where doctors must discover patient information through conversation.

## System Architecture

The following diagram shows how the agent evaluation system works:

```mermaid
graph TB
    subgraph "AgentBeats Platform"
        Runner[Scenario Runner]
    end

    subgraph "Green Agents - Evaluation System"
        Judge[Judge Agent<br/>Central Orchestrator]

        subgraph "Patient Simulation"
            PersonaMgr[Persona Manager<br/>64 Personas]
            PatientConst[Patient Constructor<br/>Generate Personas]
            PatientAgent[Patient Agent<br/>MBTI-driven Behavior]
        end

        subgraph "Evaluation Components"
            PerRoundScore[Per-Round Scoring<br/>LLM-as-Judge]
            StopDetector[Stop Detector<br/>Termination Logic]
            ReportGen[Report Generator<br/>Final Analysis]
        end

        Criteria[(Criteria CSV<br/>30 Evaluation Criteria)]
    end

    subgraph "Purple Agent - Under Evaluation"
        Doctor[Doctor Agent<br/>Being Tested]
    end

    %% Initialization Flow
    Runner -->|1. Start Evaluation| Judge
    Judge -->|2. Get Persona| PersonaMgr
    PersonaMgr -->|3. Load Templates| PatientConst
    PatientConst -->|4. Generate Background| PatientAgent
    PatientConst -->|5. Clinical Info| Judge

    %% Round-based Dialogue Loop
    Judge -->|6. Clinical Context| Doctor
    Doctor -->|7. Doctor Response| Judge
    Judge -->|8. Doctor Message| PatientAgent
    PatientAgent -->|9. Patient Response| Judge

    %% Evaluation Flow
    Judge -->|10. Evaluate Round| PerRoundScore
    Criteria -->|Evaluation Criteria| PerRoundScore
    PerRoundScore -->|11. Scores<br/>Empathy/Persuasion/Safety| Judge

    Judge -->|12. Check Stop| StopDetector
    StopDetector -->|13. Continue/Stop| Judge

    %% Final Report
    Judge -->|14. Generate Report| ReportGen
    ReportGen -->|15. Final Analysis| Runner

    %% Styling
    classDef green fill:#90EE90,stroke:#228B22,stroke-width:2px
    classDef purple fill:#DDA0DD,stroke:#8B008B,stroke-width:2px
    classDef data fill:#87CEEB,stroke:#4682B4,stroke-width:2px
    classDef eval fill:#FFD700,stroke:#FF8C00,stroke-width:2px

    class Judge,PersonaMgr,PatientConst,PatientAgent green
    class Doctor purple
    class Criteria data
    class PerRoundScore,StopDetector,ReportGen eval
```

### Evaluation Flow

The system follows a sophisticated multi-round evaluation process:

#### Phase 1: Initialization

1. **Scenario Runner** starts evaluation with persona configuration
2. **Judge Agent** receives evaluation request with persona IDs and max rounds
3. **Persona Manager** selects personas (e.g., INTJ_M_PNEUMO)
4. **Patient Constructor** generates:
   - Full patient background (age, symptoms, personality traits, concerns)
   - Clinical info subset (diagnosis, treatment details) → sent to Doctor
   - Character description (MBTI-driven behavior) → for Patient Agent
   - Roleplay examples → for context priming

#### Phase 2: Round-Based Dialogue Loop

For each round (max 10 rounds):

5. **Judge** sends clinical context to **Doctor Agent**:
   - Patient demographics (age, gender)
   - Diagnosis and recommended treatment
   - Risks, benefits, prognosis
   - Previous dialogue history
   - ⚠️ **NOT included**: Patient personality, symptoms, concerns

6. **Doctor Agent** generates response attempting to:
   - Show empathy and build trust
   - Address patient concerns
   - Persuade patient to accept treatment
   - Ensure safety and informed consent

7. **Patient Agent** generates personality-driven response:
   - Uses MBTI personality traits (hidden from Doctor)
   - Responds naturally with concerns and emotions
   - May resist, question, or gradually accept treatment

8. **Per-Round Scoring Engine** evaluates the round:
   - Uses 30 criteria from `judge_criteria.csv`
   - Categories: Empathy (1-10), Persuasion (11-20), Safety (21-30)
   - LLM judges each criterion as met/not_met/not_relevant
   - Calculates scores: Empathy, Persuasion, Safety (0-10 each)

9. **Stop Detector** checks termination conditions:
   - Patient explicitly left/refused treatment
   - Patient accepted treatment
   - Max rounds reached
   - Uses LLM to detect patient commitment/refusal signals

10. Loop continues or stops based on stop condition

#### Phase 3: Final Report Generation

11. **Report Generator** creates comprehensive analysis:
    - Aggregate scores across all rounds (weighted 30/40/30)
    - Qualitative analysis: strengths, weaknesses, key moments
    - Improvement recommendations
    - Alternative approaches
    - Overall evaluation summary

12. Results returned to **Scenario Runner** for multi-persona aggregation

### Information Asymmetry Design

The system creates realistic doctor-patient dynamics through information asymmetry:

| Information                | Doctor Has | Patient Has | Judge Has |
| -------------------------- | ---------- | ----------- | --------- |
| Patient Personality (MBTI) | ❌         | ✅          | ✅        |
| Patient Symptoms           | ❌         | ✅          | ✅        |
| Patient Concerns/Fears     | ❌         | ✅          | ✅        |
| Medical Diagnosis          | ✅         | ✅          | ✅        |
| Treatment Details          | ✅         | ✅          | ✅        |
| Dialogue History           | ✅         | ✅          | ✅        |
| Evaluation Scores          | ❌         | ❌          | ✅        |

This mirrors real medical consultations where doctors must discover patient information through conversation.

## System Components

### Green Agents (Evaluation System)

- **Judge** - Central orchestrator managing the entire evaluation lifecycle
- **Persona Manager** - Manages 64 patient personas (16 MBTI × 2 cases × 2 genders)
- **Patient Constructor** - Generates complete patient backgrounds from templates using LLM
- **Patient Agent** - Simulates patients with MBTI-driven personality-consistent behaviors
- **Per-Round Scoring** - LLM-as-judge evaluation using 30 criteria across 3 categories
- **Stop Detector** - LLM-based classification to detect dialogue termination conditions
- **Report Generator** - Creates comprehensive performance analysis with qualitative insights

### Purple Agents (Evaluated)

- **Doctor Agent** - The AI agent being evaluated (example implementation provided in `purple_agents/doctor_agent.py`)

## Configuration

Edit `scenarios/medical_dialogue/scenario.toml` to customize evaluation:

```toml
[config]
# Evaluate specific personas
persona_ids = ["INTJ_M_PNEUMO"]  # Single persona with gender
persona_ids = ["INTJ_PNEUMO"]    # Single persona, random gender
persona_ids = ["INTJ_M_PNEUMO", "ESFP_F_LUNG"]  # Multiple specific personas
persona_ids = ["all"]            # All 64 personas with gender
persona_ids = ["random"]         # Random persona each run

# Maximum dialogue rounds
max_rounds = 10

# Retry configuration for API calls
[config.retry]
patient_max_retries = 3
judge_max_retries = 5
```

For detailed configuration options, see [scenarios/medical_dialogue/README.md](scenarios/medical_dialogue/README.md).

## Developing Purple Agents

Purple Agents are the medical doctor agents being evaluated by the system. You can create your own doctor agent to test different prompting strategies, models, or architectures.

### 1. Create your Agent

Create a new file (e.g., `my_doctor.py`) that implements the A2A protocol. You can use the provided example as a template:

`scenarios/medical_dialogue/purple_agents/doctor_agent.py`

Your agent needs to:

- Accept a port number argument
- Expose an A2A-compatible HTTP endpoint
- Handle conversation history and generate appropriate medical responses

### 2. Configure the Scenario

Update `scenarios/medical_dialogue/scenario.toml` to use your agent:

```toml
[[participants]]
role = "doctor"
endpoint = "http://127.0.0.1:9019"
cmd = "python path/to/your/my_doctor.py --host 127.0.0.1 --port 9019"
```

### 3. Agent Interface

Your agent will receive a prompt containing three key sections. It must generate a single text response representing the doctor's dialogue.

#### Input Format

The system sends a text prompt to your agent containing:

1.  **Clinical Context**: Patient age, gender, diagnosis, and treatment plan.
2.  **Treatment Data**: Detailed risks, benefits, and prognosis statistics.
3.  **Dialogue History**: Full transcript of the conversation so far (if any).

**Example Input:**

```text
You are a doctor consulting with a patient about recommended surgical treatment.

=== Patient Clinical Information ===
Age: 45
Gender: Male
Medical Case: Pneumothorax
Diagnosis: Large primary spontaneous pneumothorax
Recommended Treatment: Video-assisted thoracoscopic surgery (VATS)

=== Treatment Details ===
Risks: Infection (1%), Bleeding (<1%), Recurrence (5%)
Benefits: High success rate (>95%), shorter hospital stay
Prognosis with Treatment: Full recovery expected in 2-4 weeks
Prognosis without Treatment: High risk of tension pneumothorax (life threatening)

=== Dialogue History ===
DOCTOR: Hello, I'm Dr. Smith. I'd like to discuss your test results.
PATIENT: Hi doctor, I'm a bit nervous. Is it bad?

Now provide your next response to the patient.
```

> [!NOTE]
> You can decompose the input into three sections: clinical context, treatment details, and dialogue history for more advanced implementation. See more details in [Helper Utilities](#helper-utilities).

#### Handling Logic

Your agent should:

- **Adopt the Doctor Persona**: Be professional, empathetic, and clear.
- **Use Provided Info**: Rely _only_ on the provided clinical data. Do not hallucinate symptoms not pertinent to the case.
- **Discover Hidden Info**: You do NOT know the patient's thinking or personality. You must ask questions to uncover them.
- **Drive the Conversation**: Your goal is to explain the diagnosis/treatment and persuade the patient to accept it.

#### Expected Output

- **Format**: Plain text string.
- **Content**: The spoken response to the patient.
- **Style**: Natural, conversational medical dialogue (avoid complex markdown or non-dialogue text).

#### Helper Utilities

To help you parse the unstructured prompt into structured data, we provide a helper utility:
`scenarios/medical_dialogue/purple_agents/context_parser.py`

**Usage Example:**

```python
from context_parser import parse_context

def handle_request(prompt: str):
    # Decompose the prompt into structured data
    data = parse_context(prompt)

    clinical_info = data['clinical_info']
    # {'Age': '45', 'Medical Case': 'Pneumothorax', ...}

    formatted_history = data['dialogue_history']
    # [{'speaker': 'DOCTOR', 'message': '...'}, {'speaker': 'PATIENT', 'message': '...'}]

    instruction = data['instruction']
    # "Note: The patient will describe..." OR "Now provide your next response..."

    if data['is_first_turn']:
        return generate_greeting(clinical_info, instruction)
    else:
        return generate_response(formatted_history, instruction)
```

### 4. Run the Evaluation

Run the standard evaluation command:

```bash
uv run agentbeats-run scenarios/medical_dialogue/scenario.toml
```

## Contributing

Contributions are welcome! Areas of interest:

- Additional medical conditions and cases
- New patient personality models beyond MBTI
- Enhanced scoring metrics
- Multi-language support
- Performance optimizations

## License

See [LICENSE](LICENSE) file for details.

## Acknowledgments

Built on the [AgentBeats](https://agentbeats.dev) platform for standardized agent evaluations using the [A2A protocol](https://a2a-protocol.org/latest/).

## Citation

If you use this leaderboard or the OSCE-Project framework in your research, please cite:

```bibtex
@software{osce_agentbeats_leaderboard,
  title = {OSCE-AgentBeats Medical Dialogue Evaluation Leaderboard},
  author = {Yi-Wei Liao and Ren-Di Wu and Wei-An Hou and Chi-Sheng Chen and Wu Weiqiao and Shu-Chi Wu and Kang-Lin Hsieh and Hsuan Chang},
  year = {2026},
  url = {https://github.com/MadGAA-Lab/OSCE-AgentBeats-Leaderboard},
  note = {Leaderboard for evaluating doctor agents' ability to conduct empathetic and persuasive medical consultations}
}

@software{osce_project,
  title = {OSCE-Project: Generative Adversarial Agents System for evaluating Objective Structured Clinical Examination capabilities},
  author = {Yi-Wei Liao and Ren-Di Wu and Wei-An Hou and Chi-Sheng Chen and Wu Weiqiao and Shu-Chi Wu and Kang-Lin Hsieh and Hsuan Chang},
  year = {2026},
  url = {https://github.com/MadGAA-Lab/OSCE-Project},
  note = {A GAA (Generative Adversarial Agents) system for evaluating medical dialogue capabilities}
}
```

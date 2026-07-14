"""
Per-Round Scoring Engine - LLM-based evaluation of each dialogue round
"""

import csv
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from openai import OpenAI
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam
from pydantic import BaseModel

from common import RoundEvaluation, CriterionEvaluation

logger = logging.getLogger(__name__)

# Retry configuration for critical evaluation calls
MAX_RETRIES = 5
RETRY_DELAY = 3  # seconds


class CategoryEvaluation(BaseModel):
    """LLM output for single category evaluation"""
    criteria_evaluations: list[CriterionEvaluation]


class StopConditionEvaluation(BaseModel):
    """LLM output for stop condition assessment"""
    patient_state_change: str
    should_stop: bool
    stop_reason: str | None


class PerRoundScoringEngine:
    """
    LLM-as-judge evaluation of each round after patient responds
    
    Uses criteria-based evaluation from judge_criteria.csv
    """
    
    def __init__(self, client: OpenAI, model: str, criteria_csv_path: str, max_retries: int = MAX_RETRIES, retry_delay: int = RETRY_DELAY):
        """
        Initialize PerRoundScoringEngine
        
        Args:
            client: OpenAI client for LLM calls
            model: Model name to use (should support structured output)
            criteria_csv_path: Path to judge_criteria.csv file
            max_retries: Maximum number of retry attempts
            retry_delay: Base delay between retries in seconds
        """
        self.client = client
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Load criteria from CSV
        self.criteria = self._load_criteria(criteria_csv_path)
        logger.info(f"PerRoundScoringEngine initialized with {len(self.criteria)} criteria (retries={max_retries}, delay={retry_delay}s)")
    
    def _load_criteria(self, csv_path: str) -> list[dict]:
        """Load judgment criteria from CSV file"""
        criteria = []
        path = Path(csv_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Criteria CSV not found: {csv_path}")
        
        with open(path, 'r', encoding='utf-8-sig') as f:  # utf-8-sig handles BOM
            reader = csv.DictReader(f)
            for row in reader:
                # Strip whitespace from keys to handle any formatting issues
                row = {k.strip(): v for k, v in row.items()}
                criteria.append({
                    'id': int(row['No.']),
                    'criterion': row['Criteria'],
                    'good_example': row['Good example'],
                    'bad_example': row['Bad example'],
                    'category': row['Category']
                })
        
        logger.info(f"Loaded {len(criteria)} criteria from {csv_path}")
        return criteria
    
    def evaluate_round(
        self,
        round_number: int,
        doctor_message: str,
        patient_response: str,
        dialogue_history: str,
        max_rounds: int
    ) -> RoundEvaluation:
        """
        Evaluate a single dialogue round using criteria-based judgment

        Evaluates one category per LLM call for better LLM performance:
        1. Empathy criteria (1-10)
        2. Persuasion criteria (11-20)
        3. Safety criteria (21-30)
        4. Stop condition assessment

        這 4 個呼叫互相獨立（各自只讀入參與唯讀的 self.criteria），
        因此用執行緒池並行送出 — 單回合評分時間從「4 次呼叫相加」
        降為「最慢一次呼叫」（實測 ~60s → ~15s）。

        Args:
            round_number: Current round number
            doctor_message: Doctor's message in this round
            patient_response: Patient's response in this round
            dialogue_history: Full dialogue history so far (for context)
            max_rounds: Maximum rounds configured (for stop condition check)

        Returns:
            RoundEvaluation with criteria-based scores and stop decision
        """
        logger.info(f"Evaluating round {round_number}")

        categories = ['Empathy', 'Persuasion', 'Safety']

        # Fan out the 3 category evaluations + stop condition concurrently.
        with ThreadPoolExecutor(max_workers=4) as executor:
            category_futures = {
                category: executor.submit(
                    self._evaluate_category,
                    category=category,
                    round_number=round_number,
                    doctor_message=doctor_message,
                    patient_response=patient_response,
                    dialogue_history=dialogue_history
                )
                for category in categories
            }
            stop_future = executor.submit(
                self._evaluate_stop_condition,
                round_number=round_number,
                doctor_message=doctor_message,
                patient_response=patient_response,
                dialogue_history=dialogue_history,
                max_rounds=max_rounds
            )

            # Collect in fixed category order so criteria_evaluations keeps the
            # same ordering as the previous sequential implementation.
            all_criteria_evals = []
            for category in categories:
                category_evals = category_futures[category].result()
                all_criteria_evals.extend(category_evals)
                logger.info(f"Round {round_number}: {category} evaluation complete ({len(category_evals)} criteria)")

            stop_eval = stop_future.result()
        
        # Calculate scores from criteria evaluations
        scores = self._calculate_scores_from_criteria(all_criteria_evals)
        
        # Create RoundEvaluation
        evaluation = RoundEvaluation(
            round_number=round_number,
            criteria_evaluations=all_criteria_evals,
            empathy_score=scores['empathy'],
            persuasion_score=scores['persuasion'],
            safety_score=scores['safety'],
            patient_state_change=stop_eval.patient_state_change,
            should_stop=stop_eval.should_stop,
            stop_reason=stop_eval.stop_reason
        )
        
        logger.info(f"Round {round_number} evaluation complete: "
                   f"empathy={evaluation.empathy_score:.1f}/10, "
                   f"persuasion={evaluation.persuasion_score:.1f}/10, "
                   f"safety={evaluation.safety_score:.1f}/10, "
                   f"should_stop={evaluation.should_stop}")
        
        return evaluation
    
    def _format_criteria_for_prompt(self, category: str) -> str:
        """Format criteria for a specific category as a structured list for the LLM prompt"""
        cat_criteria = [c for c in self.criteria if c['category'] == category]
        
        lines = [f"\n=== {category.upper()} CRITERIA ===\n"]
        
        for c in cat_criteria:
            lines.append(f"\n{c['id']}. {c['criterion']}")
            lines.append(f"   ✓ Good example: {c['good_example']}")
            lines.append(f"   ✗ Bad example: {c['bad_example']}")
        
        return "\n".join(lines)
    
    def _evaluate_category(
        self,
        category: str,
        round_number: int,
        doctor_message: str,
        patient_response: str,
        dialogue_history: str
    ) -> list[CriterionEvaluation]:
        """
        Evaluate a single category (Empathy, Persuasion, or Safety)
        
        Returns list of CriterionEvaluation for this category
        """
        criteria_text = self._format_criteria_for_prompt(category)
        
        system_prompt = f"""You are an expert medical dialogue evaluator using standardized criteria.

Evaluate the doctor's performance on **{category}** criteria only.

For EACH criterion, determine:
1. **status**: "met", "not_met", or "not_relevant"
   - "met": The doctor clearly demonstrated this criterion in their message
   - "not_met": This criterion was relevant but the doctor did not meet it
   - "not_relevant": This criterion does not apply to this specific round/interaction

2. **evidence**: Brief explanation (1-2 sentences) citing specific parts of the doctor's message

{criteria_text}

IMPORTANT:
- A criterion is "not_relevant" only if it doesn't apply to this specific interaction
- If a criterion is relevant but the doctor didn't meet it, mark as "not_met"
- Be objective and evidence-based
- Cite specific phrases or approaches from the doctor's message"""

        user_prompt = f"""Evaluate the {category} criteria for Round {round_number}:

=== Doctor's Message ===
{doctor_message}

=== Patient's Response ===
{patient_response}

=== Full Dialogue History (for context) ===
{dialogue_history}

Evaluate each {category} criterion and provide your assessment."""

        # Use structured output API with retry logic
        category_eval = None
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Evaluating {category} (attempt {attempt + 1}/{self.max_retries})")
                
                completion = self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=[
                        ChatCompletionSystemMessageParam(content=system_prompt, role="system"),
                        ChatCompletionUserMessageParam(content=user_prompt, role="user"),
                    ],
                    response_format=CategoryEvaluation,
                )
                
                category_eval = completion.choices[0].message.parsed
                
                if category_eval is not None:
                    logger.debug(f"{category} evaluation successful")
                    break
                else:
                    logger.warning(f"Attempt {attempt + 1}: API returned None for {category}")
                    last_error = f"API returned None for {category}"
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {category}: {e}")
                last_error = str(e)
                
            # Wait before retry (exponential backoff)
            if attempt < self.max_retries - 1:
                delay = self.retry_delay * (2 ** attempt)
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
        
        # Raise exception if all retries failed
        if category_eval is None:
            error_msg = f"Failed to evaluate {category} after {self.max_retries} attempts. Last error: {last_error}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        return category_eval.criteria_evaluations
    
    def _evaluate_stop_condition(
        self,
        round_number: int,
        doctor_message: str,
        patient_response: str,
        dialogue_history: str,
        max_rounds: int
    ) -> StopConditionEvaluation:
        """
        Evaluate whether dialogue should stop and assess patient state change
        
        Returns StopConditionEvaluation
        """
        system_prompt = """You are an expert medical dialogue evaluator.

Assess the patient's state and whether the dialogue should continue or stop.

Evaluate:
1. **patient_state_change**: How did the patient's attitude/receptiveness change this round?
   - Describe the shift in patient's openness to treatment
   - Note any breakthrough moments or setbacks
   - Be specific about what changed

2. **should_stop**: Should the dialogue end now?
   - Stop if patient clearly accepted the treatment
   - Stop if patient clearly rejected and wants to leave
   - Stop if max rounds reached
   - Continue otherwise

3. **stop_reason**: If stopping, specify why:
   - "patient_accepted": Patient agreed to treatment
   - "patient_left": Patient refused and wants to end consultation
   - "max_rounds_reached": Hit maximum round limit
   - null: Continue dialogue"""

        user_prompt = f"""Assess stop condition for Round {round_number} of {max_rounds}:

=== Doctor's Message ===
{doctor_message}

=== Patient's Response ===
{patient_response}

=== Full Dialogue History ===
{dialogue_history}

Provide your assessment of patient state change and stop condition."""

        # Use structured output API with retry logic
        stop_eval = None
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Evaluating stop condition (attempt {attempt + 1}/{self.max_retries})")
                
                completion = self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=[
                        ChatCompletionSystemMessageParam(content=system_prompt, role="system"),
                        ChatCompletionUserMessageParam(content=user_prompt, role="user"),
                    ],
                    response_format=StopConditionEvaluation,
                )
                
                stop_eval = completion.choices[0].message.parsed
                
                if stop_eval is not None:
                    logger.debug("Stop condition evaluation successful")
                    break
                else:
                    logger.warning(f"Attempt {attempt + 1}: API returned None for stop condition")
                    last_error = "API returned None for stop condition"
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for stop condition: {e}")
                last_error = str(e)
                
            # Wait before retry (exponential backoff)
            if attempt < self.max_retries - 1:
                delay = self.retry_delay * (2 ** attempt)
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
        
        # Raise exception if all retries failed
        if stop_eval is None:
            error_msg = f"Failed to evaluate stop condition after {self.max_retries} attempts. Last error: {last_error}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        return stop_eval
    
    def _calculate_scores_from_criteria(self, criteria_evals: list[CriterionEvaluation]) -> dict:
        """
        Calculate category scores from criteria evaluations
        
        Score = (number of criteria met) / (number of active/relevant criteria) * 10
        
        Returns dict with 'empathy', 'persuasion', 'safety' scores (0-10)
        """
        scores = {}
        
        for category in ['Empathy', 'Persuasion', 'Safety']:
            # Filter criteria for this category
            cat_evals = [e for e in criteria_evals if e.category == category]
            
            # Count met and active (not "not_relevant")
            met_count = sum(1 for e in cat_evals if e.status == "met")
            active_count = sum(1 for e in cat_evals if e.status != "not_relevant")
            
            # Calculate score (0-10 scale)
            if active_count > 0:
                score = (met_count / active_count) * 10
            else:
                # If no criteria were relevant, give neutral score
                score = 5.0
            
            scores[category.lower()] = round(score, 2)
            
            logger.debug(f"{category}: {met_count}/{active_count} criteria met = {score:.2f}/10")
        
        return scores

"""
LoComo10 Dataset Test for SimpleMem System
Tests retrieval time, token usage, and answer quality
"""
from pathlib import Path
import time, os, json, threading
from typing import List, Dict, Optional, Union
from dataclasses import dataclass

from openai.types import embedding_model
# import tiktoken  # Removed - token counting disabled
from tqdm import tqdm
import statistics
from collections import defaultdict
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pytorch_cos_sim
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor

import config
from simplemem.core.utils.embedding import EmbeddingModel

from main import SimpleMemSystem
from simplemem.core.models.memory_entry import Dialogue

# Download required NLTK data
# try:
#     nltk.download('punkt', quiet=True)
#     nltk.download('wordnet', quiet=True)
# except Exception as e:
#     print(f"Error downloading NLTK data: {e}")

# Initialize SentenceTransformer model for semantic similarity
try:
    sentence_model = SentenceTransformer('/mnt/qjhs-sh-lab-01/models/all-MiniLM-L6-v2', device="cpu")
except Exception as e:
    print(f"Warning: Could not load SentenceTransformer model: {e}")
    sentence_model = None

all_memory_summarize_percentage = []
all_answer_time = []
all_retrieval_time = []

# ============================================================================
# Data Structures for LoComo10 Dataset
# ============================================================================

@dataclass
class QA:
    question: str
    answer: Optional[str]
    evidence: List[str]
    category: Optional[int] = None
    adversarial_answer: Optional[str] = None

    @property
    def final_answer(self) -> Optional[str]:
        """Get the appropriate answer based on category."""
        if self.category == 5:
            return self.adversarial_answer
        return self.answer


@dataclass
class Turn:
    speaker: str
    dia_id: str
    text: str


@dataclass
class Session:
    session_id: int
    date_time: str
    turns: List[Turn]


@dataclass
class Conversation:
    speaker_a: str
    speaker_b: str
    sessions: Dict[int, Session]


@dataclass
class EventSummary:
    events: Dict[str, Dict[str, List[str]]]  # session -> speaker -> events


@dataclass
class Observation:
    observations: Dict[str, Dict[str, List[List[str]]]]  # session -> speaker -> [observation, evidence]


@dataclass
class LoCoMoSample:
    """A single sample from the LoComo dataset"""
    sample_id: str
    qa: List[QA]
    conversation: Conversation
    event_summary: EventSummary
    observation: Observation
    session_summary: Dict[str, str]


# ============================================================================
# Dataset Loading Functions
# ============================================================================

def parse_session(session_data: List[dict], session_id: int, date_time: str) -> Session:
    """Parse a single session's data, including turns with images by using their captions."""
    turns = []
    for turn in session_data:
        # For turns with images, combine caption and text
        text = turn.get("text", "")
        if "img_url" in turn and "blip_caption" in turn:
            caption_text = f"[Image: {turn['blip_caption']}]"
            if text:
                text = f"{caption_text} {text}"
            else:
                text = caption_text

        turns.append(Turn(
            speaker=turn["speaker"],
            dia_id=turn["dia_id"],
            text=text
        ))
    return Session(session_id=session_id, date_time=date_time, turns=turns)


def parse_conversation(conv_data: dict) -> Conversation:
    """Parse conversation data."""
    sessions = {}
    for key, value in conv_data.items():
        if key.startswith("session_") and isinstance(value, list):
            session_id = int(key.split("_")[1])
            date_time = conv_data.get(f"{key}_date_time")
            if date_time:
                session = parse_session(value, session_id, date_time)
                # Only add sessions that have turns after filtering
                if session.turns:
                    sessions[session_id] = session

    return Conversation(
        speaker_a=conv_data["speaker_a"],
        speaker_b=conv_data["speaker_b"],
        sessions=sessions
    )


def load_locomo_dataset(file_path: Union[str, Path]) -> List[LoCoMoSample]:
    """
    Load the LoComo dataset from a JSON file, including image-based content by using captions.

    Args:
        file_path: Path to the JSON file containing the dataset

    Returns:
        List of LoCoMoSample objects containing the parsed data
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found at {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    samples = []
    total_qa = 0
    total_image_qa = 0
    qa_counts_per_sample = []

    for sample_idx, sample in enumerate(data):
        try:
            # Parse QA data
            qa_list = []
            sample_qa_count = 0
            sample_image_qa_count = 0

            for qa_idx, qa in enumerate(sample["qa"]):
                try:
                    # Check if QA has image evidence
                    has_image_evidence = False
                    for evidence_id in qa.get("evidence", []):
                        if ":" not in evidence_id:
                            continue
                        turn_id = evidence_id.split(":")[1]
                        for session in sample["conversation"].values():
                            if isinstance(session, list):
                                for turn in session:
                                    if turn.get("dia_id", "").endswith(turn_id):
                                        if "img_url" in turn or "blip_caption" in turn:
                                            has_image_evidence = True
                                            break

                    if has_image_evidence:
                        sample_image_qa_count += 1

                    qa_obj = QA(
                        question=qa["question"],
                        answer=qa.get("answer"),
                        evidence=qa.get("evidence", []),
                        category=qa.get("category"),
                        adversarial_answer=qa.get("adversarial_answer")
                    )
                    qa_list.append(qa_obj)
                    sample_qa_count += 1

                except KeyError as e:
                    print(f"Error in sample {sample_idx}, QA pair {qa_idx}:")
                    print(f"QA data: {qa}")
                    raise e
                except Exception as e:
                    print(f"Unexpected error in sample {sample_idx}, QA pair {qa_idx}:")
                    print(f"QA data: {qa}")
                    raise e

            # Parse conversation
            conversation = parse_conversation(sample["conversation"])

            # Parse event summary
            event_summary = EventSummary(events=sample["event_summary"])

            # Parse observation
            observation = Observation(observations=sample["observation"])

            # Get session summary
            session_summary = sample.get("session_summary", {})

            # Create sample object
            sample_obj = LoCoMoSample(
                sample_id=str(sample_idx),
                qa=qa_list,
                conversation=conversation,
                event_summary=event_summary,
                observation=observation,
                session_summary=session_summary
            )
            samples.append(sample_obj)

            total_qa += sample_qa_count
            total_image_qa += sample_image_qa_count
            qa_counts_per_sample.append(sample_qa_count)

            # Print statistics for this sample
            print(f"\nSample {sample_idx}:")
            print(f"  Total QAs: {sample_qa_count}")
            print(f"  QAs with image evidence: {sample_image_qa_count}")

        except Exception as e:
            print(f"Error processing sample {sample_idx}:")
            print(str(e))
            raise e

    # Print overall statistics
    print("\nOverall Statistics:")
    print(f"Total QAs: {total_qa}")
    print(f"Total QAs with image evidence: {total_image_qa}")
    print(f"Average QAs per sample: {total_qa / len(samples):.2f}")
    print(f"Min QAs in a sample: {min(qa_counts_per_sample)}")
    print(f"Max QAs in a sample: {max(qa_counts_per_sample)}")

    return samples


# ============================================================================
# Evaluation Metrics Functions
# ============================================================================

def simple_tokenize(text):
    """Simple tokenization function."""
    text = str(text)
    return text.lower().replace('.', ' ').replace(',', ' ').replace('!', ' ').replace('?', ' ').split()


def calculate_rouge_scores(prediction: str, reference: str) -> Dict[str, float]:
    """Calculate ROUGE scores for prediction against reference."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return {
        'rouge1_f': scores['rouge1'].fmeasure,
        'rouge2_f': scores['rouge2'].fmeasure,
        'rougeL_f': scores['rougeL'].fmeasure
    }


def calculate_bleu_scores(prediction: str, reference: str) -> Dict[str, float]:
    """Calculate BLEU scores with different n-gram settings."""
    pred_tokens = nltk.word_tokenize(prediction.lower())
    ref_tokens = [nltk.word_tokenize(reference.lower())]

    weights_list = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (0.33, 0.33, 0.33, 0), (0.25, 0.25, 0.25, 0.25)]
    smooth = SmoothingFunction().method1

    scores = {}
    for n, weights in enumerate(weights_list, start=1):
        try:
            score = sentence_bleu(ref_tokens, pred_tokens, weights=weights, smoothing_function=smooth)
        except Exception:
            score = 0.0
        scores[f'bleu{n}'] = score

    return scores


def calculate_bert_scores(prediction: str, reference: str) -> Dict[str, float]:
    """Calculate BERTScore for semantic similarity."""
    try:
        P, R, F1 = bert_score([prediction], [reference], lang='en', verbose=False, model_type="/mnt/qjhs-sh-lab-01/models/roberta-large")
        return {
            'bert_precision': P.item(),
            'bert_recall': R.item(),
            'bert_f1': F1.item()
        }
    except Exception as e:
        print(f"Error calculating BERTScore: {e}")
        return {
            'bert_precision': 0.0,
            'bert_recall': 0.0,
            'bert_f1': 0.0
        }


def calculate_meteor_score(prediction: str, reference: str) -> float:
    """Calculate METEOR score for the prediction."""
    try:
        return meteor_score([reference.split()], prediction.split())
    except Exception as e:
        print(f"Error calculating METEOR score: {e}")
        return 0.0


def calculate_sentence_similarity(prediction: str, reference: str) -> float:
    """Calculate sentence embedding similarity using SentenceBERT."""
    if sentence_model is None:
        return 0.0
    try:
        # Encode sentences
        embedding1 = sentence_model.encode([prediction], convert_to_tensor=True)
        embedding2 = sentence_model.encode([reference], convert_to_tensor=True)

        # Calculate cosine similarity
        similarity = pytorch_cos_sim(embedding1, embedding2).item()
        return float(similarity)
    except Exception as e:
        print(f"Error calculating sentence similarity: {e}")
        return 0.0


def create_judge_llm_client():
    """Create a dedicated LLM client for judge evaluation"""
    from utils.llm_client import LLMClient
    import config

    # Use judge-specific settings, fall back to main settings if not specified
    judge_api_key = getattr(config, 'JUDGE_API_KEY', None) or config.OPENAI_API_KEY
    judge_base_url = getattr(config, 'JUDGE_BASE_URL', None)
    if judge_base_url is None:
        judge_base_url = getattr(config, 'OPENAI_BASE_URL', None)
    judge_model = getattr(config, 'JUDGE_MODEL', None) or config.LLM_MODEL
    judge_thinking = getattr(config, 'JUDGE_ENABLE_THINKING', False)
    judge_streaming = getattr(config, 'JUDGE_USE_STREAMING', False)

    print(f"Initializing LLM-as-judge with model: {judge_model}")
    if judge_base_url and judge_base_url != getattr(config, 'OPENAI_BASE_URL', None):
        print(f"Using separate judge endpoint: {judge_base_url}")

    # For OpenAI API, disable thinking mode to avoid parameter errors
    is_openai_api = not judge_base_url or "openai" in judge_base_url.lower()
    if is_openai_api and judge_thinking:
        print("Note: Disabling thinking mode for OpenAI API compatibility")
        judge_thinking = False

    return LLMClient(
        api_key=judge_api_key,
        model=judge_model,
        base_url=judge_base_url,
        enable_thinking=judge_thinking,
        use_streaming=judge_streaming
    )


def llm_judge_answers(prediction: str, reference: str, question: str, judge_client) -> Dict[str, Union[float, str]]:
    """Use LLM to judge if prediction is semantically equivalent to reference."""
    # Handle empty or None values
    if not prediction or not reference:
        return {"llm_judge_score": 0.0, "llm_reasoning": "Empty prediction or reference"}

    prediction = str(prediction).strip()
    reference = str(reference).strip()

    # Build judgment prompt
    #     prompt = f"""You are an expert evaluator for question-answering systems. Your task is to determine if the predicted answer contains the core information from the reference answer or provides a reasonable interpretation, being generous in your evaluation.

    # Question: {question}

    # Reference Answer: {reference}
    # Predicted Answer: {prediction}

    # Evaluation Criteria (Be generous and inclusive):
    # 1. **Core Information Match**: Does the predicted answer contain the main factual content from the reference?
    # 2. **Partial Correctness**: Even if not 100% complete, does it capture key elements correctly?
    # 3. **Reasonable Interpretation**: Could the predicted answer be a valid interpretation of the available information?
    # 4. **Format Flexibility**: Accept different time formats, date expressions, and phrasings
    # 5. **Semantic Variations**: "meeting", "appointment", "get together" are all acceptable
    # 6. **Contextual Understanding**: If the answer shows understanding of the context, give credit
    # 7. **Incomplete but Accurate**: Partial information that's correct should be scored positively
    # 8. **Different but Valid**: Alternative valid answers to the same question should be accepted

    # Special Cases:
    # - If reference says "2 PM" and prediction says "afternoon" or "14:00" → Accept
    # - If reference says "Alice and Bob meet" and prediction says "Alice has a meeting" → Accept (partial info)
    # - If reference includes specific details but prediction captures the general idea → Accept
    # - Numbers and dates with minor variations (e.g., "2pm" vs "2:00 PM") → Accept
    # - If the prediction demonstrates understanding of the conversation → Lean toward accepting

    # Only score 0.0 if the predicted answer is clearly wrong, contradicts the reference, or is completely unrelated.

    # Output your evaluation in JSON format:
    # {{
    #   "score": 1.0,  // 1.0 for acceptable answer, 0.0 only for clearly wrong answers
    #   "reasoning": "Brief explanation focusing on what information was captured correctly"
    # }}

    # Return ONLY the JSON, no other text."""

    prompt = f"""You are an expert Relevance & Accuracy Evaluator. Your task is to determine if the Predicted Answer successfully retrieves the necessary information to answer the Question, based on the Reference Answer.

Question: {question}
Reference Answer: {reference}
Predicted Answer: {prediction}

Evaluation Criteria:

1. **Responsiveness to Query**: 
   The predicted answer must directly address the specific question asked. It must contain highly relevant information that is topically aligned with the user's intent.

2. **Core Fact Preservation**: 
   The prediction must capture the "Key Signal" or "Core Entity" from the reference. The primary subject (Who), event (What), or outcome must be factually grounded in the reference text.

3. **Informational Utility**: 
   The answer must provide actionable or meaningful value. Even if brief, it must convey the essential message required by the question context.

4. **Acceptable Representational Variances (Robustness Protocol)**:
   To ensure fair evaluation of semantic meaning over syntactic rigidity, you must accept the following variations as **Valid Matches**:
   - **Temporal & Numerical Margins**: Accept timestamps within a reasonable proximity (e.g., +/- 1-2 days due to timezone/reporting differences) and rounded numerical approximations.
   - **Granularity Independence**: Accept answers at different levels of abstraction (e.g., "Afternoon" vs. "14:05", "Late October" vs. "Oct 25th") provided they encompass the truth.
   - **Information Subsetting**: A valid subset of the reference (e.g., mentioning 1 out of 3 reasons) is acceptable if it answers the core of the question.
   - **Synonymy**: Recognize domain-specific synonyms and different formats as equivalent.

Grading Logic:
- Score 1.0 (Pass): The prediction contains relevant core information, answers the question with sufficient utility, OR falls within the acceptable representational variances defined in criterion #4.
- Score 0.0 (Fail): The prediction contains NO relevant information, fails to identify the core subject/event, or provides no key info that matches the question's intent.

Output your evaluation in JSON format:
{{
  "score": 1.0, 
  "reasoning": "Brief assessment focusing on information relevance and core match."
}}

Return ONLY the JSON, no other text.
"""

    try:
        messages = [
            {
                "role": "system",
                "content": "You are an expert evaluator. Always output valid JSON format."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        import config
        # Use JSON format if configured
        response_format = None
        if hasattr(config, 'USE_JSON_FORMAT') and config.USE_JSON_FORMAT:
            response_format = {"type": "json_object"}

        # Use judge-specific temperature setting
        judge_temperature = getattr(config, 'JUDGE_TEMPERATURE', 0.3)

        response = judge_client.chat_completion(
            messages,
            temperature=judge_temperature,
            response_format=response_format,
            max_retries=3  # Ensure robust evaluation with retries
        )

        # Parse JSON response
        result = judge_client.extract_json(response)
        score = float(result.get("score", 0.0))
        reasoning = result.get("reasoning", "No reasoning provided")

        return {
            "llm_judge_score": score,
            "llm_reasoning": reasoning
        }

    except Exception as e:
        print(f"Warning: LLM judge evaluation failed: {e}")
        return {
            "llm_judge_score": 0.0,
            "llm_reasoning": f"Evaluation failed: {e}"
        }


def calculate_metrics(prediction: str, reference: str, question: str = None, judge_client=None,
                      use_llm_judge: bool = False) -> Dict[str, float]:
    """Calculate comprehensive evaluation metrics for a prediction."""
    # Handle empty or None values
    if not prediction or not reference:
        return {
            "exact_match": 0,
            "f1": 0.0,
            "rouge1_f": 0.0,
            "rouge2_f": 0.0,
            "rougeL_f": 0.0,
            "bleu1": 0.0,
            "bleu2": 0.0,
            "bleu3": 0.0,
            "bleu4": 0.0,
            "bert_f1": 0.0,
            "meteor": 0.0,
            "sbert_similarity": 0.0,
            "llm_judge_score": 0.0
        }

    # Convert to strings if they're not already
    prediction = str(prediction).strip()
    reference = str(reference).strip()

    # Calculate exact match
    exact_match = int(prediction.lower() == reference.lower())

    # Calculate token-based F1 score
    pred_tokens = set(simple_tokenize(prediction))
    ref_tokens = set(simple_tokenize(reference))
    common_tokens = pred_tokens & ref_tokens

    if not pred_tokens or not ref_tokens:
        f1 = 0.0
    else:
        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(ref_tokens)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Calculate all scores
    rouge_scores = calculate_rouge_scores(prediction, reference)
    bleu_scores = calculate_bleu_scores(prediction, reference)
    bert_scores = calculate_bert_scores(prediction, reference)
    meteor = calculate_meteor_score(prediction, reference)
    sbert_similarity = calculate_sentence_similarity(prediction, reference)

    # Combine all metrics
    metrics = {
        "exact_match": exact_match,
        "f1": f1,
        **rouge_scores,
        **bleu_scores,
        **bert_scores,
        "meteor": meteor,
        "sbert_similarity": sbert_similarity,
        "llm_judge_score": 0.0  # Default value
    }

    # Add LLM judge evaluation if enabled
    if use_llm_judge and question and judge_client:
        llm_result = llm_judge_answers(prediction, reference, question, judge_client)
        metrics["llm_judge_score"] = llm_result["llm_judge_score"]
        metrics["llm_reasoning"] = llm_result["llm_reasoning"]

    return metrics


def aggregate_metrics(all_metrics: List[Dict[str, float]], all_categories: List[int]) -> Dict[
    str, Dict[str, Union[float, Dict[str, float]]]]:
    """Calculate aggregate statistics for all metrics, split by category."""
    if not all_metrics:
        return {}

    # Initialize aggregates for overall and per-category metrics
    aggregates = defaultdict(list)
    category_aggregates = defaultdict(lambda: defaultdict(list))

    # Collect all values for each metric, both overall and per category
    for metrics, category in zip(all_metrics, all_categories):
        for metric_name, value in metrics.items():
            # Skip non-numeric values like llm_reasoning
            if isinstance(value, (int, float)):
                aggregates[metric_name].append(value)
                category_aggregates[category][metric_name].append(value)

    # Calculate statistics for overall metrics
    results = {
        "overall": {}
    }

    for metric_name, values in aggregates.items():
        if values:  # Only calculate if we have numeric values
            results["overall"][metric_name] = {
                'mean': statistics.mean(values),
                'std': statistics.stdev(values) if len(values) > 1 else 0.0,
                'median': statistics.median(values),
                'min': min(values),
                'max': max(values),
                'count': len(values)
            }

    # Calculate statistics for each category
    for category in sorted(category_aggregates.keys()):
        results[f"category_{category}"] = {}
        for metric_name, values in category_aggregates[category].items():
            if values:  # Only calculate if we have values for this category
                results[f"category_{category}"][metric_name] = {
                    'mean': statistics.mean(values),
                    'std': statistics.stdev(values) if len(values) > 1 else 0.0,
                    'median': statistics.median(values),
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }

    return results


# ============================================================================
# Testing Classes
# ============================================================================


# TokenCounter class removed - token counting disabled
# class TokenCounter:
#     """Count tokens sent to LLM"""
#     def __init__(self, model="gpt-4"):
#         try:
#             self.encoding = tiktoken.encoding_for_model(model)
#         except:
#             self.encoding = tiktoken.get_encoding("cl100k_base")
#
#     def count(self, text: str) -> int:
#         return len(self.encoding.encode(text))


class LoCoMoTester:
    """Test SimpleMem system on LoComo10 dataset"""

    def __init__(self, system: SimpleMemSystem, dataset_path: str, use_llm_judge: bool = False,
                 test_workers: int = None):
        self.system = system
        self.dataset_path = Path(dataset_path)
        self.use_llm_judge = use_llm_judge
        self.test_workers = test_workers

        # Initialize judge client if needed
        self.judge_client = None
        if self.use_llm_judge:
            self.judge_client = create_judge_llm_client()

        # Statistics
        self.retrieval_times = []
        self.answer_times = []
        self.total_times = []
        self.metrics_list = []
        self.categories = []

        # For thread-safe incremental saving
        self.save_lock = threading.RLock()

    def _get_existing_results(self, result_file_path: str) -> Dict:
        """Load existing results from file if it exists, return processed questions set"""
        processed_questions = set()
        existing_results = []

        if os.path.exists(result_file_path):
            try:
                with open(result_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if 'detailed_results' in data:
                    for result in data['detailed_results']:
                        if 'question' in result:
                            processed_questions.add(result['question'])
                            existing_results.append(result)
                print(f"Found {len(processed_questions)} already processed questions in {result_file_path}")
                return processed_questions, existing_results, data
            except Exception as e:
                print(f"Warning: Failed to load existing results file {result_file_path}: {e}")

        return set(), [], None

    def _save_incremental_result(self, result_file: str, new_result: Dict, all_results: List, existing_data: Dict = None):
        """Save incremental result to file"""
        try:
            # Update summary statistics
            retrieval_times = [r.get('retrieval_time', 0) for r in all_results]
            answer_times = [r.get('answer_time', 0) for r in all_results]
            total_times = [r.get('total_time', 0) for r in all_results]

            summary = {
                'num_questions': len(all_results),
                'avg_retrieval_time': sum(retrieval_times) / len(retrieval_times) if retrieval_times else 0,
                'avg_answer_time': sum(answer_times) / len(answer_times) if answer_times else 0,
                'avg_total_time': sum(total_times) / len(total_times) if total_times else 0,
            }

            # Collect metrics for aggregation if needed
            metrics_list = [r.get('metrics', {}) for r in all_results if r.get('metrics')]
            categories = [r.get('category', 0) for r in all_results]

            aggregated = {}
            if metrics_list:
                aggregated = aggregate_metrics(metrics_list, categories)

            # Save updated results
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'summary': summary,
                    'aggregated_metrics': aggregated,
                    'detailed_results': all_results
                }, f, indent=2)

            print(f"Results incrementally saved to {result_file} (total questions: {len(all_results)})")
        except Exception as e:
            print(f"Warning: Failed to save incremental result to {result_file}: {e}")

    def generate_category5_answer(self, question: str, contexts: List, adversarial_answer: str) -> (str, int, int):
        """
        Special answer generation for category 5 (adversarial questions).
        Ask model to choose between "Not mentioned in the conversation" and the adversarial answer.
        """
        import random

        # Randomly shuffle the order of two options
        options = ["Not mentioned in the conversation", adversarial_answer]
        if random.random() < 0.5:
            options = [options[1], options[0]]

        # Build context string
        context_str = self.system.answer_generator._format_contexts(contexts)

        # Build special prompt for category 5
        prompt = f"""
Based on the context below, answer the following question.

Context:
{context_str}

Question: {question}

Select the correct answer from the following two options. If the given answer is wrong or not answerable based on the context, you should choose "Not mentioned in the conversation".

Option A: {options[0]}
Option B: {options[1]}

Requirements:
1. Choose the option that best matches the context
2. If neither answer is supported by the context, or if the provided specific answer is incorrect, choose "Not mentioned in the conversation"
3. Return your response in JSON format

Output Format:
```json
{{
  "reasoning": "Brief explanation of your choice",
  "answer": "Your selected answer (either '{options[0]}' or '{options[1]}')"
}}
```

Return ONLY the JSON, no other text.
"""

        messages = [
            {
                "role": "system",
                "content": "You are a professional Q&A assistant. You must output valid JSON format."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        # Retry up to 3 times
        max_retries = 3
        for attempt in range(max_retries):
            try:
                import config
                # Use JSON format if configured
                response_format = None
                if hasattr(config, 'USE_JSON_FORMAT') and config.USE_JSON_FORMAT:
                    response_format = {"type": "json_object"}

                response, p_t, c_t = self.system.llm_client.chat_completion_with_token_comsumption(
                    messages,
                    temperature=0.5,  # Higher temperature for category 5
                    response_format=response_format,
                    max_retries=2  # Ensure robust category 5 evaluation with retries
                )

                # Parse JSON response
                result = self.system.llm_client.extract_json(response)
                return result.get("answer", response.strip()), p_t, c_t

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Category 5 answer generation attempt {attempt + 1}/{max_retries} failed: {e}. Retrying...")
                else:
                    print(f"Warning: Failed to generate category 5 answer after {max_retries} attempts: {e}")
                    return "Not mentioned in the conversation", 0, 0  # Default to safe answer

    def load_dataset(self, limit: int = None) -> List[LoCoMoSample]:
        """Load LoComo10 dataset"""
        print(f"Loading dataset from {self.dataset_path}...")
        samples = load_locomo_dataset(self.dataset_path)

        if limit:
            samples = samples[:limit]
            print(f"Limited to {limit} samples")

        return samples

    def convert_to_dialogues(self, sample: LoCoMoSample) -> List[Dialogue]:
        """Convert LoComo sample to Dialogue objects"""
        dialogues = []
        dialogue_id = 1

        # Process all sessions in order
        for session_id in sorted(sample.conversation.sessions.keys()):
            session = sample.conversation.sessions[session_id]

            for turn in session.turns:
                dialogue = Dialogue(
                    dialogue_id=dialogue_id,
                    speaker=turn.speaker,
                    content=turn.text,
                    timestamp=session.date_time  # Use session datetime
                )
                dialogues.append(dialogue)
                dialogue_id += 1

        return dialogues

    def test_sample(self, sample: LoCoMoSample, sample_idx: int, table_name: str, result_file: str = None, enable_parallel_questions: bool = False):
        """Test a single sample from the dataset"""
        print(f"\n{'=' * 80}")
        print(f"Testing Sample {sample_idx}")
        print(f"{'=' * 80}")

        # Convert and add dialogues
        dialogues = self.convert_to_dialogues(sample)
        print(f"Adding {len(dialogues)} dialogues to memory...")

        sample.qa = sample.qa[:TOTAL_QA_SAMPLE]

        # Initialize or load existing results
        existing_questions = set()
        all_existing_results = []
        existing_data = None

        if result_file and os.path.exists(result_file):
            processed_questions, existing_results, data = self._get_existing_results(result_file)
            existing_questions = processed_questions
            all_existing_results = existing_results
            existing_data = data
            print(f"Found {len(existing_questions)} already processed questions in {result_file}")

        # Filter QA list to only unprocessed questions
        original_qa_count = len(sample.qa)
        unprocessed_qa = []
        for qa in sample.qa:
            if qa.question not in existing_questions:
                unprocessed_qa.append(qa)

        print(f"Total questions in sample: {original_qa_count}, unprocessed: {len(unprocessed_qa)}")

        if len(unprocessed_qa) == 0:
            print("All questions already processed. Returning existing results.")
            return all_existing_results

        # Replace sample.qa with unprocessed questions for processing
        sample.qa = unprocessed_qa

        add_start = time.time()

        ##mengyao_debug fusionrag: 这个simple mem的add memory流程里面没有可以复用的cache，就跳过build了
        build_flag = f"{config.LANCEDB_PATH}/{table_name}.flag"
        if not os.path.exists(build_flag):
            self.system.vector_store.clear()
            self.system.add_dialogues(dialogues)
            self.system.finalize()
            with open(build_flag, "w", encoding="utf-8") as f:
                f.write("build_complete")

            token_consumtion_result = f"{TOKEN_CONSUMPTION}/locomo_{sample_idx}.json"
            with open(token_consumtion_result, "w", encoding="utf-8") as f:
                json.dump(self.system.memory_builder.stats(), f, indent=4)

        else:
            print(f"{table_name} already built.")

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("/mnt/qjhs-sh-lab-01/models/Qwen3-8B", trust_remote_code=True)

        all_entries = self.system.vector_store.get_all_entries()
        all_summary = "".join(x.lossless_restatement for x in all_entries)
        all_history = "".join([x.content for x in dialogues])

        all_summary = tokenizer.encode(all_summary, add_special_tokens=True)
        all_history = tokenizer.encode(all_history, add_special_tokens=True)
        all_memory_summarize_percentage.append(len(all_summary) / len(all_history))
        print(f"summary percentage = {sum(all_memory_summarize_percentage) / len(all_memory_summarize_percentage)*100}%")


        add_time = time.time() - add_start
        print(f"Memory building time: {add_time:.2f}s")

        # Test each question (parallel or sequential)
        if enable_parallel_questions and len(sample.qa) > 1:
            sample_results = self._test_questions_parallel(sample.qa, result_file, all_existing_results, existing_data)
        else:
            sample_results = self._test_questions_sequential(sample.qa, result_file, all_existing_results, existing_data)

        # Combine existing and new results
        all_results = all_existing_results + sample_results
        return all_results

    def _test_questions_sequential(self, qa_list: List, result_file: str = None, all_existing_results: List = None, existing_data: Dict = None):
        """Test questions sequentially (original method)"""
        sample_results = []
        start_idx = len(all_existing_results) if all_existing_results else 0

        for qa_idx, qa in enumerate(qa_list):
            result = self._process_single_question(qa, qa_idx + start_idx)
            sample_results.append(result)

            # Save incremental result after each question
            if result_file:
                self._save_incremental_result(result_file, result, all_existing_results + sample_results if all_existing_results else sample_results, existing_data)

        return sample_results

    def _test_questions_parallel(self, qa_list: List, result_file: str = None, all_existing_results: List = None, existing_data: Dict = None):
        """Test questions in parallel using ThreadPoolExecutor"""
        import concurrent.futures
        import threading

        print(f"\n[Parallel Testing] Processing {len(qa_list)} questions in parallel")
        sample_results = []
        start_idx = len(all_existing_results) if all_existing_results else 0

        # Use ThreadPoolExecutor for parallel question processing
        # Use explicit test_workers parameter, or config, or reasonable default
        import config

        if self.test_workers is not None:
            max_workers = self.test_workers
        else:
            max_workers = getattr(config, 'MAX_RETRIEVAL_WORKERS', 16)

        # Apply reasonable limits
        max_workers = min(
            max_workers,
            len(qa_list),  # Don't create more workers than questions
            20  # Higher limit for better parallelism, but watch API rate limits
        )
        max_workers = max(max_workers, 1)  # At least 1 worker

        print(f"[Parallel Testing] Using {max_workers} parallel workers for {len(qa_list)} questions")

        # Shared state for incremental saving
        collected_results = []
        if all_existing_results:
            collected_results.extend(all_existing_results)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all question processing tasks
            future_to_qa = {}
            for qa_idx, qa in enumerate(qa_list):
                future = executor.submit(self._process_single_question, qa, qa_idx + start_idx)
                future_to_qa[future] = (qa, qa_idx)

            # Collect results as they complete, maintain order
            results_dict = {}
            for future in concurrent.futures.as_completed(future_to_qa):
                qa, qa_idx = future_to_qa[future]
                try:
                    result = future.result()
                    results_dict[qa_idx] = result
                    print(f"[Parallel Testing] Question {qa_idx + 1 + start_idx} completed")

                    # Save incremental result after each question
                    if result_file:
                        with self.save_lock:
                            collected_results.append(result)
                            self._save_incremental_result(result_file, result, collected_results, existing_data)
                except Exception as e:
                    print(f"[Parallel Testing] Question {qa_idx + 1 + start_idx} failed: {e}")
                    # Create a default result for failed questions
                    error_result = {
                        'question': qa.question,
                        'answer': "Error during processing",
                        'reference': qa.final_answer,
                        'category': qa.category if qa.category is not None else 0,
                        'retrieval_time': 0,
                        'answer_time': 0,
                        'total_time': 0,
                        'num_retrieved': 0,
                        'metrics': {}
                    }
                    results_dict[qa_idx] = error_result

                    # Save incremental error result
                    if result_file:
                        with self.save_lock:
                            collected_results.append(error_result)
                            self._save_incremental_result(result_file, error_result, collected_results, existing_data)

            # Sort results by qa_idx to maintain original order
            for qa_idx in sorted(results_dict.keys()):
                sample_results.append(results_dict[qa_idx])

        return sample_results

    def _process_single_question(self, qa, qa_idx: int):
        """Process a single question and return result"""
        question = qa.question
        category = qa.category if qa.category is not None else 0

        # For category 5, the ground truth is always "Not mentioned in the conversation"
        # For other categories, use qa.final_answer
        if category == 5:
            reference_answer = "Not mentioned in the conversation"
        else:
            reference_answer = qa.final_answer

        print(f"\n[Q{qa_idx + 1}] Category {category}: {question}")

        # Measure retrieval time
        # For category 5 (adversarial), disable reflection since "no answer means no answer"
        retrieval_start = time.time()
        if category == 5:
            contexts, prompt_tokens, completion_tokens = self.system.hybrid_retriever.retrieve(question, enable_reflection=False)
        else:
            contexts, prompt_tokens, completion_tokens = self.system.hybrid_retriever.retrieve(question)
        retrieval_time = time.time() - retrieval_start

        if os.environ.get("DUMP_QUESTIONS", "").lower() == "true":
            context_str, context_str_list = self.system.answer_generator._format_contexts_list(contexts)
            prefix, query_prompt = self.system.answer_generator._build_answer_prompt_fusionrag(question, context_str)
            context_str_list[0] = prefix + context_str_list[0]
            result =  {
                'question': question,
                'answer': "dummy",
                'reference': reference_answer,
                'category': category,
                'retrieval_time': retrieval_time,
                'answer_time': 1,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_time': 1,
                'num_retrieved': len(contexts),
                'metrics': {},
                'question_origin': {
                    "system_prompt": "You are a professional Q&A assistant. Extract concise answers from context. You must output valid JSON format.",
                    "prefix": "",
                    "query_prompt": query_prompt,
                    "question": question,
                    "reference": reference_answer,
                    "fusionrag_list": context_str_list
                }
            }
            return result

        # Measure answer generation time
        answer_start = time.time()

        # Use special answer generation for category 5
        if category == 5:
            adversarial_answer = qa.adversarial_answer if qa.adversarial_answer else "Unknown answer"
            answer, p_t, c_t = self.generate_category5_answer(question, contexts, adversarial_answer)
        else:
            answer, p_t, c_t = self.system.answer_generator.generate_answer_with_token_consumptions(question, contexts, model="kimi-k2.6")
        prompt_tokens += p_t
        completion_tokens += c_t

        answer_time = time.time() - answer_start
        all_answer_time.append(answer_time)
        all_retrieval_time.append(retrieval_time)
        print(f"average answer time={sum(all_answer_time)/len(all_answer_time)}")
        print(f"average retrieval time={sum(all_retrieval_time)/len(all_retrieval_time)}")

        total_time = retrieval_time + answer_time

        # Calculate metrics
        if reference_answer:
            metrics = calculate_metrics(
                answer,
                reference_answer,
                question=question,
                judge_client=self.judge_client,
                use_llm_judge=self.use_llm_judge
            )
        else:
            metrics = {}

        # Store statistics
        self.retrieval_times.append(retrieval_time)
        self.answer_times.append(answer_time)
        self.total_times.append(total_time)
        if metrics:
            self.metrics_list.append(metrics)
            self.categories.append(category)

        # Print results
        print(f"  Retrieved: {len(contexts)} memory entries")
        print(f"  Retrieval time: {retrieval_time:.3f}s")
        print(f"  Answer time: {answer_time:.3f}s")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Answer: {answer}")
        if reference_answer:
            print(f"  Reference: {reference_answer}")
            if metrics:
                print(f"  F1: {metrics.get('f1', 0):.3f}, "
                      f"ROUGE-L: {metrics.get('rougeL_f', 0):.3f}, "
                      f"BERTScore: {metrics.get('bert_f1', 0):.3f}")
                if self.use_llm_judge and 'llm_judge_score' in metrics:
                    print(f"  LLM Judge: {metrics.get('llm_judge_score', 0):.3f}")
                    if 'llm_reasoning' in metrics:
                        print(f"  LLM Reasoning: {metrics.get('llm_reasoning', '')}")

        return {
            'question': question,
            'answer': answer,
            'reference': reference_answer,
            'category': category,
            'retrieval_time': retrieval_time,
            'answer_time': answer_time,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_time': total_time,
            'num_retrieved': len(contexts),
            'metrics': metrics
        }

    def run_test(self, embedding_model, num_samples: int = None, save_results: bool = True,
                 result_file: str = 'locomo10_test_results.json', enable_parallel_questions: bool = False, sample=None, sample_idx: int = None):
        """Run full test on dataset"""
        print("\n" + "=" * 80)
        print(" SimpleMem LoComo10 Dataset Test".center(80))
        print("=" * 80 + "\n")

        all_results = []

        # Test each sample
        table_name = f"locomo_{sample.sample_id}"
        self.system = SimpleMemSystem(embedding_model=embedding_model, clear_db=False, table_name=table_name)  ##mengyao_debug

        sample_results = self.test_sample(sample, sample_idx, table_name=table_name, result_file=result_file, enable_parallel_questions=enable_parallel_questions)

        if sample_results is None:
            return

        all_results.extend(sample_results)

        # Calculate aggregate metrics
        print("\n" + "=" * 80)
        print(" Test Summary".center(80))
        print("=" * 80 + "\n")

        # Timing statistics
        print("Timing Statistics:")
        print(f"  Average retrieval time: {sum(self.retrieval_times) / len(self.retrieval_times):.3f}s")
        print(f"  Average answer time: {sum(self.answer_times) / len(self.answer_times):.3f}s")
        print(f"  Average total time: {sum(self.total_times) / len(self.total_times):.3f}s")
        print(f"  Total retrieval time: {sum(self.retrieval_times):.2f}s")
        print(f"  Total answer time: {sum(self.answer_times):.2f}s")

        # Answer quality metrics
        if self.metrics_list:
            print(f"\nAnswer Quality Metrics:")
            aggregated = aggregate_metrics(self.metrics_list, self.categories)

            # Overall metrics
            overall = aggregated.get('overall', {})
            print(f"\nOverall Performance:")
            metrics_to_show = ['f1', 'rougeL_f', 'bert_f1', 'sbert_similarity']
            if self.use_llm_judge:
                metrics_to_show.append('llm_judge_score')

            for metric_name in metrics_to_show:
                if metric_name in overall:
                    stats = overall[metric_name]
                    print(f"  {metric_name:20s}: {stats['mean']:.4f} (±{stats['std']:.4f})")

            # Per-category metrics
            print(f"\nPer-Category Performance:")
            for key in sorted(aggregated.keys()):
                if key.startswith('category_'):
                    category_num = key.split('_')[1]
                    category_data = aggregated[key]
                    if 'f1' in category_data:
                        f1_mean = category_data['f1']['mean']
                        count = category_data['f1']['count']
                        print(f"  Category {category_num}: F1={f1_mean:.4f} (n={count})")

        # Save results
        if save_results:
            output_file = result_file
            with open(output_file, 'w') as f:
                json.dump({
                    'summary': {
                        'num_questions': len(all_results),
                        'avg_retrieval_time': sum(self.retrieval_times) / len(self.retrieval_times),
                        'avg_answer_time': sum(self.answer_times) / len(self.answer_times),
                        'avg_total_time': sum(self.total_times) / len(self.total_times),
                    },
                    'aggregated_metrics': aggregated if self.metrics_list else {},
                    'detailed_results': all_results
                }, f, indent=2)
            print(f"\nResults saved to {output_file}")

        print("\n" + "=" * 80)
        print(" Test Complete!".center(80))
        print("=" * 80 + "\n")

        return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test SimpleMem on LoComo10 dataset')
    parser.add_argument('--dataset', type=str, default='test_ref/locomo10.json',
                        help='Path to LoComo10 dataset')
    parser.add_argument('--total_qa', type=int, default=1000,
                        help='total qa for each sample')
    parser.add_argument('--num-samples', type=int, default=None,
                        help='Number of samples to test (default: all)')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save results to file')
    parser.add_argument('--result-file', type=str, default='locomo10_test_results.json',
                        help='Path to the result file')
    parser.add_argument('--parallel-questions', action='store_true',
                        help='Enable parallel processing of questions within each sample')
    parser.add_argument('--llm-judge', action='store_true',
                        help='Enable LLM-as-judge evaluation for semantic answer comparison')
    parser.add_argument('--test-workers', type=int, default=None,
                        help='Number of parallel workers for question testing (default: use config MAX_RETRIEVAL_WORKERS)')

    args = parser.parse_args()

    # Create system
    print("Initializing SimpleMem system...")

    # Create tester
    tester_loader = LoCoMoTester(None, args.dataset, use_llm_judge=args.llm_judge, test_workers=args.test_workers)

    if args.llm_judge:
        print("LLM-as-judge evaluation enabled")
    if args.test_workers:
        print(f"Using {args.test_workers} test workers for parallel question processing")

    samples = tester_loader.load_dataset()
    total_samples = len(samples)

    print(f"Total samples: {total_samples}")

    for sample in samples:
        sample.qa = [qa for qa in sample.qa if qa.category != 5]

    MAX_PARALLEL = 16
    TOTAL_QA_SAMPLE = args.total_qa ##mengyao_debug
    print(f"TOTAL_QA_SAMPLE={TOTAL_QA_SAMPLE}")

    ##mengyao_debug 并发处理sample、并发处理单个sample里面的 dialogs、并发处理问题
    TOKEN_CONSUMPTION = "token_consumption_build_memory_locomo"
    RESULT_DIR = "./results_locomo"
    if os.environ.get("FUSIONRAG", "").lower() == "true":
        RESULT_DIR = "./results_locomo_fusionrag"
    if config.LLM_MODEL.lower() != "qwen3-8b":
        RESULT_DIR = RESULT_DIR+f"_{config.LLM_MODEL.lower()}"

    if os.environ.get("DUMP_QUESTIONS", "").lower() == "true":
        RESULT_DIR = RESULT_DIR+"_dump"

    if os.environ.get("DEBUG", "").lower() == "true":
        MAX_PARALLEL = 1
        TOTAL_QA_SAMPLE = 10

    if config.LLM_MODEL.lower() != "qwen3-8b":
        TOKEN_CONSUMPTION = TOKEN_CONSUMPTION+f"_{config.LLM_MODEL.lower()}"

    if os.environ.get("FUSIONRAG", "").lower() == "true":
        TOKEN_CONSUMPTION += "_fusionrag"


    os.makedirs(TOKEN_CONSUMPTION, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    print(f"TOKEN_CONSUMPTION={TOKEN_CONSUMPTION}")
    print(f"RESULT_DIR={RESULT_DIR}")
    print(f"DB={config.LANCEDB_PATH}")

    time.sleep(3)

    def run_sample(sample_idx, sample, embedding_model, save_dir):
        tester = LoCoMoTester(
            None,
            args.dataset,
            use_llm_judge=args.llm_judge,
            test_workers=args.test_workers,
        )

        return tester.run_test(
            embedding_model=embedding_model,
            num_samples=args.num_samples,
            save_results=not args.no_save,
            result_file=f"./{save_dir}/locomo_{sample_idx}.json",
            enable_parallel_questions=args.parallel_questions,
            sample_idx=sample_idx,
            sample=sample,
        )


    embedding_model = EmbeddingModel()

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as executor:
        futures = [
            executor.submit(
                run_sample,
                sample_idx,
                sample,
                embedding_model,
                RESULT_DIR
            )
            for sample_idx, sample in enumerate(samples)
        ]

        for future in as_completed(futures):
            future.result()

import json
import re
from collections import defaultdict
from pathlib import Path
from openai import OpenAI
import hashlib
import concurrent.futures
import datetime
import threading
import copy

SYSTEM = """You are a strict, method-blind evaluator of question answering. Judge only whether the candidate answer is semantically correct according to the question and reference answer. Do not infer which system produced it."""
TEMPLATE = """Decide whether the candidate answer is correct.

Rules:
1. Accept concise paraphrases, equivalent names, equivalent date/number formats, and a correct answer embedded in harmless extra explanation.
2. Reject a wrong person, entity, event, date, ordering, count, amount, or polarity; a contradiction; a refusal when the reference answers the question; or an answer missing a required list item, comparison, calculation, or event.
3. Extra text is harmless only if it does not add a materially false answer claim.
4. For open-ended preference or recommendation questions, the answer need not copy every example in the reference, but it must correctly use the core personal information required by the reference.
5. Treat the reference as the scoring ground truth. Do not use outside knowledge.

Question:
{question}

Reference answer:
{reference}

Candidate answer:
{prediction}

Do not REASON. JUST GIVE THE RESULT.
Return exactly one JSON object with one boolean field and no other text:
{{"correct": true}}
or
{{"correct": false}}"""

def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

PROMPT_SHA256 = text_sha256(SYSTEM + "\n\0\n" + TEMPLATE)

# Cache management for judge results
CACHE_DIR = Path(".judge_cache")
CACHE_DIR.mkdir(exist_ok=True)
CACHE_FILE = CACHE_DIR / "judge_cache.json"

# Global cache and lock for thread-safe access
_judgment_cache = None
_cache_lock = threading.RLock()  # 可重入锁，支持嵌套锁

def load_judgment_cache() -> dict:
    """Load judgment cache from file."""
    global _judgment_cache
    with _cache_lock:
        if _judgment_cache is not None:
            return _judgment_cache

        if CACHE_FILE.exists():
            try:
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    _judgment_cache = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load judgment cache: {e}")
                _judgment_cache = {}
        else:
            _judgment_cache = {}
        return _judgment_cache

def save_judgment_cache(cache: dict):
    """Save judgment cache to file."""
    global _judgment_cache
    with _cache_lock:
        try:
            # 创建字典的深拷贝避免迭代时被修改
            cache_copy = copy.deepcopy(cache)
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(cache_copy, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save judgment cache: {e}")

def get_judgment_key(question: str, reference: str, prediction: str, model: str) -> str:
    """Generate a unique cache key for judgment request."""
    input_str = f"{PROMPT_SHA256}:{model}:{question}:{reference}:{prediction}"
    return hashlib.sha256(input_str.encode("utf-8")).hexdigest()

def parse_correct(content: object) -> bool:
    text = str(content or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except Exception as e:
        print(f"parse error: {e} text: {text}")
        raise
    if not isinstance(value, dict) or set(value) != {"correct"} or not isinstance(value["correct"], bool):
        raise ValueError("judge response is not strict correct:boolean JSON")
    return value["correct"]

def judge_answer(question: str, reference: str, prediction: str, config) -> tuple[bool, dict]:
    """
    Use LLM-as-Judge to determine if prediction is correct.
    Returns (is_correct, usage_dict)
    """
    try:
        # Use judge-specific config if available, fallback to main config
        api_key = getattr(config, "JUDGE_API_KEY", None) or getattr(config, "OPENAI_API_KEY", "sk-dummy")
        base_url = getattr(config, "JUDGE_BASE_URL", None) or getattr(config, "OPENAI_BASE_URL", "http://127.0.0.1:30002/v1")
        model = getattr(config, "JUDGE_MODEL", None) or getattr(config, "LLM_MODEL", "GLM-5.3")
        timeout = getattr(config, "JUDGE_TIMEOUT", 900.0)
        max_tokens = getattr(config, "JUDGE_MAX_TOKENS", 1024)

        # Generate cache key
        cache_key = get_judgment_key(question, reference, prediction, model)

        # 1. 检查缓存（带锁）
        with _cache_lock:
            cache = load_judgment_cache()
            if cache_key in cache:
                cached_data = cache[cache_key]
                # Validate that the cache entry has required fields
                if ("correct" in cached_data and "usage" in cached_data and
                    "model" in cached_data and "prompt_sha256" in cached_data and
                    cached_data["prompt_sha256"] == PROMPT_SHA256):
                    # print(f"hit cache")
                    return (cached_data["correct"], cached_data["usage"], cached_data["model"])
                else:
                    print(f"Warning: Invalid cache entry for {cache_key[:16]}..., ignoring")

        # 2. 缓存未命中，调用API（在锁外执行，避免阻塞其他任务）
        if not api_key:
            raise ValueError("No API key configured for judge LLM")

        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

        item = {
            "question": question,
            "reference": str(reference),
            "prediction": str(prediction)
        }

        # print(f"judge answer")
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": TEMPLATE.format(**item)},
            ],
            max_tokens=max_tokens,
            stream=False,
            response_format={"type": "json_object"},
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}
        )

        choice = completion.choices[0]
        message = choice.message
        content = message.content or ""

        if not content and hasattr(message, "reasoning_content"):
            content = message.reasoning_content or ""

        usage = completion.usage.model_dump() if completion.usage else {}
        model_name = str(completion.model or model)
        correct = parse_correct(content)

        # 3. 保存到缓存（带锁）
        with _cache_lock:
            # 重新加载缓存，确保获取最新版本
            cache = load_judgment_cache()
            cache[cache_key] = {
                "correct": correct,
                "usage": usage,
                "model": model_name,
                "prompt_sha256": PROMPT_SHA256,
                "timestamp": datetime.datetime.now().isoformat()
            }
            save_judgment_cache(cache)

        return correct, usage, model_name
    except Exception as e:
        print(f"Judge LLM error: {e}")
        raise


def process_eval_dataset(
    token_dir: str, result_dir: str, dataset_name: str
):
    import config  # Import config module for judge LLM settings
    token_folder = Path(token_dir)
    result_folder = Path(result_dir)

    # 1. 统计 Token 消耗
    all_prompt_tokens = []
    all_completion_tokens = []

    if token_folder.exists():
        for file_path in token_folder.glob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                p_tokens = data.get("prompt_tokens", 0)
                c_tokens = data.get("completion_tokens", 0)
                if p_tokens > 0:
                    all_prompt_tokens.append(p_tokens)
                if c_tokens > 0:
                    all_completion_tokens.append(c_tokens)
            except Exception as e:
                print(f"Error reading token file {file_path}: {e}")
    else:
        print(f"Warning: Token directory {token_dir} does not exist.")

    # 2. 从 detailed_results 或单样本 metrics 细粒度统计 F1 & BLEU-1
    metrics_by_category = defaultdict(lambda: {"f1": [], "bleu1": []})
    judgments_by_category = defaultdict(list)  # 存储每个类别的 LLM 判断结果 (True/False)

    # 全局问题指标池 (用于计算 Overall)
    global_f1s = []
    global_bleu1s = []
    global_judgments = []  # 存储所有样本的 LLM 判断结果

    judgment_tasks = []  # 存储待并发评估的任务

    question_prompt_tokens = []
    question_completion_tokens = []

    if result_folder.exists():
        for file_path in result_folder.glob("**/*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # 情况 A：Locomo 等包含 detailed_results 列表的汇总文件
                if "detailed_results" in data:
                    for item in data["detailed_results"]:
                        cat_key = item.get("category") or item.get(
                            "question_type", "uncategorized"
                        )
                        metrics = item.get("metrics", {})

                        f1_val = metrics.get("f1")
                        bleu1_val = metrics.get("bleu1")

                        if f1_val is not None:
                            metrics_by_category[cat_key]["f1"].append(f1_val)
                            global_f1s.append(f1_val)
                        if bleu1_val is not None:
                            metrics_by_category[cat_key]["bleu1"].append(
                                bleu1_val
                            )
                            global_bleu1s.append(bleu1_val)
                        if "prompt_tokens" in item and "completion_tokens" in item:
                            question_prompt_tokens.append(item.get("prompt_tokens"))
                            question_completion_tokens.append(item.get("completion_tokens"))

                        # LLM-as-Judge evaluation (collect tasks for parallel processing)
                        if "question" in item and "answer" in item and "reference" in item:
                            judgment_tasks.append({
                                "cat_key": cat_key,
                                "question": item["question"],
                                "reference": item["reference"],
                                "prediction": item["answer"]
                            })

                # 情况 B：单个 JSON 对应单个样本 (如 question_type + metrics)
                elif "metrics" in data:
                    cat_key = data.get("question_type", "uncategorized")
                    metrics = data.get("metrics", {})

                    f1_val = metrics.get("f1")
                    bleu1_val = metrics.get("bleu1")

                    if f1_val is not None:
                        metrics_by_category[cat_key]["f1"].append(f1_val)
                        global_f1s.append(f1_val)
                    if bleu1_val is not None:
                        metrics_by_category[cat_key]["bleu1"].append(bleu1_val)
                        global_bleu1s.append(bleu1_val)
                    if "prompt_tokens" in data and "completion_tokens" in data:
                        question_prompt_tokens.append(data.get("prompt_tokens"))
                        question_completion_tokens.append(data.get("completion_tokens"))

                    # LLM-as-Judge evaluation for single sample (collect tasks for parallel processing)
                    if "question" in data and "answer" in data and "reference" in data:
                        judgment_tasks.append({
                            "cat_key": cat_key,
                            "question": data["question"],
                            "reference": data["reference"],
                            "prediction": data["answer"]
                        })

            except Exception as e:
                print(f"Error reading result file {file_path}: {e}")
    else:
        print(f"Warning: Result directory {result_dir} does not exist.")

    # 3. 并发执行 LLM-as-Judge 评估
    if judgment_tasks:
        print(f"Executing {len(judgment_tasks)} LLM judge tasks concurrently...")

        def process_task(task):
            try:
                correct, usage, model = judge_answer(
                    question=task["question"],
                    reference=task["reference"],
                    prediction=task["prediction"],
                    config=config
                )
                return task["cat_key"], correct
            except Exception as e:
                print(f"Judge LLM failed for question '{task.get('question', 'unknown')}': {e}")
                return None

        # 使用线程池并发执行
        max_workers = min(16, len(judgment_tasks))  # 限制最大并发数
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {executor.submit(process_task, task): task for task in judgment_tasks}
            for future in concurrent.futures.as_completed(future_to_task):
                result = future.result()
                if result is not None:
                    cat_key, correct = result
                    judgments_by_category[cat_key].append(correct)
                    global_judgments.append(correct)
        print("LLM judge tasks completed.")
    else:
        print("No LLM judge tasks to execute.")

    # 4. 输出汇总结果
    avg_prompt = (
        sum(all_prompt_tokens) / len(all_prompt_tokens)
        if all_prompt_tokens
        else 0
    )
    avg_comp = (
        sum(all_completion_tokens) / len(all_completion_tokens)
        if all_completion_tokens
        else 0
    )

    avg_prompt_question = sum(question_prompt_tokens) / len(question_prompt_tokens)
    avg_comletion_question = sum(question_completion_tokens) / len(question_completion_tokens)

    print(f"\n================ [{dataset_name}] Summary ================")
    print(
        f"[Build] Average Prompt Tokens    : {avg_prompt:.2f}\n"
        f"[Build] Average Completion Tokens: {avg_comp:.2f}"
    )
    print(
        f"[question] Average Prompt Tokens    : {avg_prompt_question:.2f}\n"
        f"[question] Average Completion Tokens: {avg_comletion_question:.2f}"
    )

    if metrics_by_category:
        print("\n" + "-" *108)
        print(
            f"{'Category / Question Type':<30} | {'F1 (Mean)':<12} | {'BLEU-1 (Mean)':<12} | {'LLM Acc. (Mean)':<16} | {'Count':<6}"
        )
        print("-" * 108)

        for cat_key in sorted(metrics_by_category.keys()):
            f1s = metrics_by_category[cat_key]["f1"]
            bleu1s = metrics_by_category[cat_key]["bleu1"]

            avg_f1 = sum(f1s) / len(f1s) if f1s else None
            avg_bleu1 = sum(bleu1s) / len(bleu1s) if bleu1s else None

            judgments = judgments_by_category[cat_key]
            if judgments:
                llm_acc = sum(judgments) / len(judgments)
                llm_acc_str = f"{llm_acc:.4f}"
            else:
                llm_acc_str = "N/A"
            f1_str = f"{avg_f1:.4f}" if avg_f1 is not None else "N/A"
            bleu1_str = f"{avg_bleu1:.4f}" if avg_bleu1 is not None else "N/A"
            sample_cnt = max(len(f1s), len(bleu1s), len(judgments))

            print(
                f"{cat_key:<30} | {f1_str:<12} | {bleu1_str:<12} | {llm_acc_str:<16} | {sample_cnt:<6}"
            )

        # 计算并输出 Overall 总平均
        print("-" * 108)
        overall_f1 = (
            sum(global_f1s) / len(global_f1s) if global_f1s else None
        )
        overall_bleu1 = (
            sum(global_bleu1s) / len(global_bleu1s) if global_bleu1s else None
        )
        overall_llm_acc = (
            sum(global_judgments) / len(global_judgments) if global_judgments else None
        )

        ov_f1_str = (
            f"{overall_f1:.4f}" if overall_f1 is not None else "N/A"
        )
        ov_bleu1_str = (
            f"{overall_bleu1:.4f}" if overall_bleu1 is not None else "N/A"
        )
        ov_llm_acc_str = (
            f"{overall_llm_acc:.4f}" if overall_llm_acc is not None else "N/A"
        )
        total_count = max(len(global_f1s), len(global_bleu1s), len(global_judgments))

        print(
            f"{'OVERALL (Total Average)':<30} | {ov_f1_str:<12} | {ov_bleu1_str:<12} | {ov_llm_acc_str:<16} | {total_count:<6}"
        )
        print("=" * 108)
    else:
        print("No metrics found.")


def process_halumem_dataset(result_dir: str, dataset_name: str):
    """处理 Halumem 结果，统计每个问题类型的 token 消耗和准确率"""
    result_folder = Path(result_dir)

    # 按问题类型分类的统计数据
    category_stats = defaultdict(lambda: {
        "turn_build_memory_prompt_tokens": [],
        "turn_build_memory_completion_tokens": [],
        "retrieval_prompt_tokens": [],
        "retrieval_completion_tokens": [],
        "answer_prompt_tokens": [],
        "answer_completion_tokens": [],
        "f1": [],
        "bleu1": [],
        "exact_match": [],
        "rouge1_f": [],
        "rouge2_f": [],
        "rougeL_f": [],
        "bert_f1": [],
        "meteor": [],
        "sbert_similarity": [],
        "llm_judge_score": []
    })

    total_samples = 0

    if result_folder.exists():
        for file_path in result_folder.glob("**/*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Halumem 文件是一个列表，每个元素是一个问题
                if isinstance(data, list):
                    for item in data:
                        cat_key = item.get("question_type", "uncategorized")
                        stats = category_stats[cat_key]

                        # Token 统计
                        tbmp = item.get("turn_build_memory_prompt_tokens")
                        if tbmp is not None:
                            stats["turn_build_memory_prompt_tokens"].append(tbmp)

                        tbct = item.get("turn_build_memory_completion_tokens")
                        if tbct is not None:
                            stats["turn_build_memory_completion_tokens"].append(tbct)

                        rpt = item.get("retrieval_prompt_tokens")
                        if rpt is not None:
                            stats["retrieval_prompt_tokens"].append(rpt)

                        rct = item.get("retrieval_completion_tokens")
                        if rct is not None:
                            stats["retrieval_completion_tokens"].append(rct)

                        apt = item.get("answer_prompt_tokens")
                        if apt is not None:
                            stats["answer_prompt_tokens"].append(apt)

                        act = item.get("answer_completion_tokens")
                        if act is not None:
                            stats["answer_completion_tokens"].append(act)

                        # Metrics 统计
                        metrics = item.get("metrics", {})
                        if metrics:
                            for metric_name in ["f1", "bleu1", "exact_match",
                                               "rouge1_f", "rouge2_f", "rougeL_f",
                                               "bert_f1", "meteor", "sbert_similarity",
                                               "llm_judge_score"]:
                                val = metrics.get(metric_name)
                                if val is not None:
                                    stats[metric_name].append(val)

                        total_samples += 1
                else:
                    print(f"Warning: {file_path} is not a list, skipping")
            except Exception as e:
                print(f"Error reading result file {file_path}: {e}")
    else:
        print(f"Warning: Result directory {result_dir} does not exist.")
        return

    print(f"\n================ [{dataset_name}] Halumem Summary ================")
    print(f"Total samples: {total_samples}")

    if not category_stats:
        print("No data found.")
        return

    # 输出每个问题类型的平均统计
    print(f"\n{'Question Type':<30} | {'Turn Build Prompt':>8} | {'Turn Build Comp':>8} | {'Retr+Ans Prompt':>8} | {'Retr+Ans Comp':>8} | {'F1':>8} | {'BLEU-1':>8} | {'Count':>6}")
    print("-" * 120)

    for cat_key in sorted(category_stats.keys()):
        stats = category_stats[cat_key]

        # 计算平均值
        avg_turn_build = sum(stats["turn_build_memory_prompt_tokens"]) / len(stats["turn_build_memory_prompt_tokens"]) if stats["turn_build_memory_prompt_tokens"] else 0
        avg_turn_build_completion = sum(stats["turn_build_memory_completion_tokens"]) / len(stats["turn_build_memory_completion_tokens"]) if stats["turn_build_memory_completion_tokens"] else 0
        avg_retrieval_prompt = sum(stats["retrieval_prompt_tokens"]) / len(stats["retrieval_prompt_tokens"]) if stats["retrieval_prompt_tokens"] else 0
        avg_answer_prompt = sum(stats["answer_prompt_tokens"]) / len(stats["answer_prompt_tokens"]) if stats["answer_prompt_tokens"] else 0
        avg_retrieval_completion = sum(stats["retrieval_completion_tokens"]) / len(stats["retrieval_completion_tokens"]) if stats["retrieval_completion_tokens"] else 0
        avg_answer_completion = sum(stats["answer_completion_tokens"]) / len(stats["answer_completion_tokens"]) if stats["answer_completion_tokens"] else 0

        # 计算总和（根据用户需求）
        avg_retrieval_plus_answer_prompt = avg_retrieval_prompt + avg_answer_prompt
        avg_retrieval_plus_answer_completion = avg_retrieval_completion + avg_answer_completion

        # 计算指标平均值
        avg_f1 = sum(stats["f1"]) / len(stats["f1"]) if stats["f1"] else 0
        avg_bleu1 = sum(stats["bleu1"]) / len(stats["bleu1"]) if stats["bleu1"] else 0

        count = len(stats["turn_build_memory_prompt_tokens"])  # 使用 token 计数作为样本数

        print(f"{cat_key:<30} | {avg_turn_build:>8.0f} | {avg_turn_build_completion:>8.0f} | {avg_retrieval_plus_answer_prompt:>8.0f} | {avg_retrieval_plus_answer_completion:>8.0f} | {avg_f1:>8.4f} | {avg_bleu1:>8.4f} | {count:>6}")

    print("-" * 120)

    # 计算总体平均值
    overall_turn_build = []
    overall_turn_build_completion = []
    overall_retrieval_prompt = []
    overall_answer_prompt = []
    overall_retrieval_completion = []
    overall_answer_completion = []
    overall_f1 = []
    overall_bleu1 = []

    for stats in category_stats.values():
        overall_turn_build.extend(stats["turn_build_memory_prompt_tokens"])
        overall_turn_build_completion.extend(stats["turn_build_memory_completion_tokens"])
        overall_retrieval_prompt.extend(stats["retrieval_prompt_tokens"])
        overall_answer_prompt.extend(stats["answer_prompt_tokens"])
        overall_retrieval_completion.extend(stats["retrieval_completion_tokens"])
        overall_answer_completion.extend(stats["answer_completion_tokens"])
        overall_f1.extend(stats["f1"])
        overall_bleu1.extend(stats["bleu1"])

    if overall_turn_build:
        avg_overall_turn_build = sum(overall_turn_build) / len(overall_turn_build)
        avg_overall_turn_build_completion = sum(overall_turn_build_completion) / len(overall_turn_build_completion) if overall_turn_build_completion else 0
        avg_overall_retrieval_prompt = sum(overall_retrieval_prompt) / len(overall_retrieval_prompt)
        avg_overall_answer_prompt = sum(overall_answer_prompt) / len(overall_answer_prompt)
        avg_overall_retrieval_plus_answer_prompt = avg_overall_retrieval_prompt + avg_overall_answer_prompt
        avg_overall_retrieval_completion = sum(overall_retrieval_completion) / len(overall_retrieval_completion)
        avg_overall_answer_completion = sum(overall_answer_completion) / len(overall_answer_completion)
        avg_overall_retrieval_plus_answer_completion = avg_overall_retrieval_completion + avg_overall_answer_completion
        avg_overall_f1 = sum(overall_f1) / len(overall_f1) if overall_f1 else 0
        avg_overall_bleu1 = sum(overall_bleu1) / len(overall_bleu1) if overall_bleu1 else 0

        print(f"{'OVERALL':<30} | {avg_overall_turn_build:>8.0f} | {avg_overall_turn_build_completion:>8.0f} | {avg_overall_retrieval_plus_answer_prompt:>8.0f} | {avg_overall_retrieval_plus_answer_completion:>8.0f} | {avg_overall_f1:>8.4f} | {avg_overall_bleu1:>8.4f} | {total_samples:>6}")

    print("=" * 120)


if __name__ == "__main__":
    tasks = [
        (
            "./token_consumption_build_memory_locomo",
            "./results_locomo",
            "locomo",
        ),
        (
            "./token_consumption_build_memory_locomo_glm-4.5-air",
            "./results_locomo_glm-4.5-air",
            "locomo",
        ),
        (
            "./token_consumption_build_memory_locomo_kimi-k2.6",
            "./results_locomo_kimi-k2.6",
            "locomo",
        ),
        (
            "./token_consumption_build_memory_longmemeval",
            "./results_longmem",
            "longmemeval",
        ),
        (
            "./token_consumption_build_memory_longmemeval_glm-4.5-air",
            "./results_longmem_glm-4.5-air",
            "longmemeval",
        ),
        (
            "./token_consumption_build_memory_longmemeval_kimi-k2.6",
            "./results_longmem_kimi-k2.6",
            "longmemeval",
        ),
    ]

    for token_path, result_path, name in tasks:
        process_eval_dataset(token_path, result_path, name)

    # 添加 Halumem 处理
    # process_halumem_dataset("./results_halumem", "halumem")
import json
from collections import defaultdict
from pathlib import Path


def process_eval_dataset(
    token_dir: str, result_dir: str, dataset_name: str
):
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

    # 全局问题指标池 (用于计算 Overall)
    global_f1s = []
    global_bleu1s = []

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

            except Exception as e:
                print(f"Error reading result file {file_path}: {e}")
    else:
        print(f"Warning: Result directory {result_dir} does not exist.")

    # 3. 输出汇总结果
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
        print("\n" + "-" * 68)
        print(
            f"{'Category / Question Type':<30} | {'F1 (Mean)':<12} | {'BLEU-1 (Mean)':<12} | {'Count':<6}"
        )
        print("-" * 68)

        for cat_key in sorted(metrics_by_category.keys()):
            f1s = metrics_by_category[cat_key]["f1"]
            bleu1s = metrics_by_category[cat_key]["bleu1"]

            avg_f1 = sum(f1s) / len(f1s) if f1s else None
            avg_bleu1 = sum(bleu1s) / len(bleu1s) if bleu1s else None

            f1_str = f"{avg_f1:.4f}" if avg_f1 is not None else "N/A"
            bleu1_str = f"{avg_bleu1:.4f}" if avg_bleu1 is not None else "N/A"
            sample_cnt = max(len(f1s), len(bleu1s))

            print(
                f"{cat_key:<30} | {f1_str:<12} | {bleu1_str:<12} | {sample_cnt:<6}"
            )

        # 计算并输出 Overall 总平均
        print("-" * 68)
        overall_f1 = (
            sum(global_f1s) / len(global_f1s) if global_f1s else None
        )
        overall_bleu1 = (
            sum(global_bleu1s) / len(global_bleu1s) if global_bleu1s else None
        )

        ov_f1_str = (
            f"{overall_f1:.4f}" if overall_f1 is not None else "N/A"
        )
        ov_bleu1_str = (
            f"{overall_bleu1:.4f}" if overall_bleu1 is not None else "N/A"
        )
        total_count = max(len(global_f1s), len(global_bleu1s))

        print(
            f"{'OVERALL (Total Average)':<30} | {ov_f1_str:<12} | {ov_bleu1_str:<12} | {total_count:<6}"
        )
        print("=" * 68)
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
        # (
        #     "./token_consumption_build_memory_locomo",
        #     "./results_locomo",
        #     "locomo",
        # ),
        (
            "./token_consumption_build_memory_locomo_kimi-k2.6",
            "./results_locomo_kimi-k2.6",
            "locomo",
        ),
        # (
        #     "./token_consumption_build_memory_longmemeval",
        #     "./results_longmem",
        #     "longmemeval",
        # ),
    ]

    for token_path, result_path, name in tasks:
        process_eval_dataset(token_path, result_path, name)

    # 添加 Halumem 处理
    process_halumem_dataset("./results_halumem", "halumem")
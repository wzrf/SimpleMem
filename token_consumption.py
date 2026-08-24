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


if __name__ == "__main__":
    tasks = [
        (
            "./token_consumption_build_memory_locomo",
            "./results",
            "locomo",
        ),
        (
            "./token_consumption_build_memory_longmemeval",
            "./results_longmem",
            "longmemeval",
        ),
    ]

    for token_path, result_path, name in tasks:
        process_eval_dataset(token_path, result_path, name)
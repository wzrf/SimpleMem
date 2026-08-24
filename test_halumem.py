import os
import time
import json
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

# 导入框架系统与组件
from main import SimpleMemSystem
from simplemem.core.models.memory_entry import Dialogue
from simplemem.core.utils.embedding import EmbeddingModel

# 复用测试指标计算与评估模块
from test_locomo10 import (
    calculate_metrics,
    aggregate_metrics,
    create_judge_llm_client
)


# ============================================================================
# HaluMem 数据结构与 JSONL 加载器
# ============================================================================

@dataclass
class HaluMemSample:
    sample_id: str
    persona_info: str
    sessions: List[Dict]


def load_halumem_dataset(path_input: Union[str, Path]) -> List[HaluMemSample]:
    """
    加载 HaluMem 数据集 (.jsonl 格式或包含 jsonl 的目录)
    """
    path_input = Path(path_input)
    if not path_input.exists():
        raise FileNotFoundError(f"Path not found at {path_input}")

    files = []
    if path_input.is_dir():
        files = sorted(list(path_input.glob("*.jsonl")))
    else:
        files = [path_input]

    samples = []
    for fpath in files:
        with open(fpath, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                sid = data.get("uuid", f"{fpath.stem}_{line_idx}")
                persona = data.get("persona_info", "")
                sessions = data.get("sessions", [])
                samples.append(HaluMemSample(sample_id=str(sid), persona_info=persona, sessions=sessions))

    print(f"Successfully loaded {len(samples)} samples from {path_input}")
    return samples


# ============================================================================
# HaluMem 增量测试器
# ============================================================================

class HaluMemTester:
    def __init__(self, use_llm_judge: bool = False):
        self.use_llm_judge = use_llm_judge
        self.judge_client = create_judge_llm_client() if use_llm_judge else None

    def run_single_sample(self, sample: HaluMemSample, sample_idx: int, embedding_model,
                          save_dir: str = "./results_halumem"):
        """
        处理单个 HaluMem 样本：
        按 Session 顺序执行：
          1. 将当前 Session 的 Dialogue 加入 Memory 并统计增加 Dialogues 的 Token 开销。
          2. 对当前 Session 的 Questions 进行检索与回答，统计 Retrieve 和 QA 的 Token 开销。
        """
        safe_sample_id = sample.sample_id.replace(" ", "_")
        table_name = f"halumem_{safe_sample_id[:20]}"

        # 初始化 SimpleMem 系统
        system = SimpleMemSystem(embedding_model=embedding_model, clear_db=True, table_name=table_name)

        sample_results = []
        global_dialogue_id = 1

        accumulated_history_turns = 0
        total_build_prompt_tokens = 0
        total_build_completion_tokens = 0

        # 遍历每个 Session
        for s_idx, session in enumerate(sample.sessions):
            dialogue_turns = session.get("dialogue", [])
            questions = session.get("questions", [])

            # ----------------------------------------------------------------
            # 步骤 1：首先将对话加到 Memory，并统计增加 Dialogues 的 Token 开销
            # ----------------------------------------------------------------
            session_dialogues = []
            for turn in dialogue_turns:
                session_dialogues.append(Dialogue(
                    dialogue_id=global_dialogue_id,
                    speaker=turn.get("role", "user"),
                    content=turn.get("content", ""),
                    timestamp=turn.get("timestamp", "")
                ))
                global_dialogue_id += 1

            turn_build_prompt_tokens = 0
            turn_build_completion_tokens = 0

            if session_dialogues:
                # 获取写入前的内存构建消耗统计
                stats_before = system.memory_builder.stats() if hasattr(system, 'memory_builder') else {}

                system.add_dialogues(session_dialogues)
                system.finalize()

                accumulated_history_turns += len(session_dialogues)

                # 计算本轮增加 Dialogue 的 Token 开销 (Diff)
                stats_after = system.memory_builder.stats() if hasattr(system, 'memory_builder') else {}
                if stats_after:
                    turn_build_prompt_tokens = stats_after.get("prompt_tokens", 0) - stats_before.get("prompt_tokens",
                                                                                                      0)
                    turn_build_completion_tokens = stats_after.get("completion_tokens", 0) - stats_before.get(
                        "completion_tokens", 0)

                    total_build_prompt_tokens += turn_build_prompt_tokens
                    total_build_completion_tokens += turn_build_completion_tokens

            # ----------------------------------------------------------------
            # 步骤 2：在记忆更新完成后，回答该 Session 内的问题
            # ----------------------------------------------------------------
            for q_idx, q_item in enumerate(questions):
                question = q_item.get("question", "")
                reference = str(q_item.get("answer", ""))
                question_type = q_item.get("question_type", "default")
                question_id = f"{safe_sample_id}_s{s_idx}_q{q_idx}"

                retrieval_prompt_tokens = 0
                retrieval_completion_tokens = 0
                answer_prompt_tokens = 0
                answer_completion_tokens = 0

                # 2.1 检索阶段开销统计
                retrieval_start = time.time()
                contexts, retrieval_prompt_tokens, retrieval_completion_tokens = system.hybrid_retriever.retrieve(
                    question)
                retrieval_time = time.time() - retrieval_start

                # 2.2 回答生成阶段开销统计
                answer_start = time.time()
                answer, answer_prompt_tokens, answer_completion_tokens = system.answer_generator.generate_answer_with_token_consumptions(
                    question, contexts
                )
                answer_time = time.time() - answer_start
                total_time = retrieval_time + answer_time

                # 2.3 计算 Evaluation Metrics
                metrics = calculate_metrics(
                    prediction=answer,
                    reference=reference,
                    question=question,
                    judge_client=self.judge_client,
                    use_llm_judge=self.use_llm_judge
                )

                query_res = {
                    'sample_id': safe_sample_id,
                    'question_id': question_id,
                    'session_index': s_idx,
                    'question_type': question_type,
                    'difficulty': q_item.get("difficulty", "normal"),
                    'question': question,
                    'answer': answer,
                    'reference': reference,

                    # 1. 本轮及累计增加 dialogues 的 memory token 消耗
                    'session_dialogue_turns_added': len(session_dialogues),
                    'turn_build_memory_prompt_tokens': turn_build_prompt_tokens,
                    'turn_build_memory_completion_tokens': turn_build_completion_tokens,
                    'turn_build_memory_total_tokens': turn_build_prompt_tokens + turn_build_completion_tokens,

                    'accumulated_history_turns': accumulated_history_turns,
                    'total_build_memory_prompt_tokens': total_build_prompt_tokens,
                    'total_build_memory_completion_tokens': total_build_completion_tokens,

                    # 2. 检索阶段 Token 开销
                    'retrieval_prompt_tokens': retrieval_prompt_tokens,
                    'retrieval_completion_tokens': retrieval_completion_tokens,
                    'retrieval_total_tokens': retrieval_prompt_tokens + retrieval_completion_tokens,

                    # 3. 回答生成阶段 Token 开销
                    'answer_prompt_tokens': answer_prompt_tokens,
                    'answer_completion_tokens': answer_completion_tokens,
                    'answer_total_tokens': answer_prompt_tokens + answer_completion_tokens,

                    # 4. QA 环节总 Token 消耗 (Retrieve + Answer)
                    'query_total_prompt_tokens': retrieval_prompt_tokens + answer_prompt_tokens,
                    'query_total_completion_tokens': retrieval_completion_tokens + answer_completion_tokens,
                    'query_total_tokens': (retrieval_prompt_tokens + answer_prompt_tokens) + (
                                retrieval_completion_tokens + answer_completion_tokens),

                    'retrieval_time': retrieval_time,
                    'answer_time': answer_time,
                    'total_time': total_time,
                    'num_retrieved': len(contexts),
                    'metrics': metrics
                }
                sample_results.append(query_res)

            # 保存单个 sample 结果
            os.makedirs(save_dir, exist_ok=True)
            with open(f"{save_dir}/{safe_sample_id}.json", 'w', encoding='utf-8') as f:
                json.dump(sample_results, f, indent=2, ensure_ascii=False)

        avg_f1 = sum(r['metrics'].get('f1', 0) for r in sample_results) / len(sample_results) if sample_results else 0
        print(
            f"[{sample_idx}] Sample ID: {safe_sample_id} | Total Queries: {len(sample_results)} | Avg F1: {avg_f1:.3f}")
        return sample_results


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test SimpleMem on HaluMem dataset')
    parser.add_argument('--dataset', type=str, default='test_ref/HaluMem-Medium.jsonl',
                        help='Path to HaluMem jsonl file or directory')
    parser.add_argument('--llm-judge', action='store_true', help='Enable LLM-as-judge evaluation')
    parser.add_argument('--output-dir', type=str, default='./results_halumem',
                        help='Directory to save evaluation results')
    args = parser.parse_args()

    # 1. 加载 jsonl 数据集
    samples = load_halumem_dataset(args.dataset)

    MAX_PARALLEL = 16
    if os.environ.get('DEBUG') == "1":
        MAX_PARALLEL = 1

    embedding_model = EmbeddingModel()


    def _worker(idx_sample):
        idx, sample = idx_sample
        tester = HaluMemTester(use_llm_judge=args.llm_judge)
        emb_model = embedding_model
        return tester.run_single_sample(sample, idx, emb_model, save_dir=args.output_dir)


    all_flattened_results = []
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as executor:
        futures = [executor.submit(_worker, (i, s)) for i, s in enumerate(samples)]
        for future in as_completed(futures):
            try:
                sample_res = future.result()
                all_flattened_results.extend(sample_res)
            except Exception as e:
                import traceback

                print(f"Sample execution failed: {e}")
                traceback.print_exc()

    if all_flattened_results:
        metrics_list = [r['metrics'] for r in all_flattened_results if r.get('metrics')]
        categories = [r.get('question_type', 'default') for r in all_flattened_results if r.get('metrics')]

        aggregated = aggregate_metrics(metrics_list, categories)

        print("\n" + "=" * 80)
        print(" HaluMem Test Summary ".center(80, "="))
        print(f"Total Queries Evaluated Across All Samples: {len(all_flattened_results)}")

        overall = aggregated.get('overall', {})
        for metric_name in ['f1', 'rougeL_f', 'bert_f1', 'sbert_similarity', 'llm_judge_score']:
            if metric_name in overall:
                print(f"  {metric_name:20s}: {overall[metric_name]['mean']:.4f}")
        print("=" * 80)
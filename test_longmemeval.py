import os
import time
import json
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

from grpc.framework.interfaces.base.utilities import completion

from main import SimpleMemSystem
from simplemem.core.models.memory_entry import Dialogue
from simplemem.core.utils.embedding import EmbeddingModel

# 复用 test_locomo10 中的指标计算与评估模块
from test_locomo10 import (
    calculate_metrics,
    aggregate_metrics,
    create_judge_llm_client
)
import config

# ============================================================================
# LongMemEval 数据结构定义
# ============================================================================

@dataclass
class LongMemSample:
    question_id: str
    question_type: str
    question: str
    question_date: str
    answer: str
    haystack_sessions: List[List[Dict[str, str]]]
    haystack_dates: List[str]

def load_longmemeval_dataset(file_path: Union[str, Path]) -> List[LongMemSample]:
    """加载 LongMemEval JSON 数据集"""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found at {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    samples = []
    for item in data:
        samples.append(LongMemSample(
            question_id=item["question_id"],
            question_type=item.get("question_type", "default"),
            question=item["question"],
            question_date=item.get("question_date", ""),
            answer=str(item.get("answer", "")),
            haystack_sessions=item.get("haystack_sessions", []),
            haystack_dates=item.get("haystack_dates", [])
        ))
    print(f"Loaded {len(samples)} samples from {file_path}")
    return samples

# ============================================================================
# 测试逻辑类
# ============================================================================

class LongMemEvalTester:
    def __init__(self, use_llm_judge: bool = False):
        self.use_llm_judge = use_llm_judge
        self.judge_client = create_judge_llm_client() if use_llm_judge else None

    def convert_to_dialogues(self, sample: LongMemSample) -> List[Dialogue]:
        """将 LongMemEval 的 haystack_sessions 拆解转换为 SimpleMem Dialogue 格式"""
        dialogues = []
        dialogue_id = 1

        for session_idx, session in enumerate(sample.haystack_sessions):
            # 获取对应的 session 时间戳
            timestamp = sample.haystack_dates[session_idx] if session_idx < len(sample.haystack_dates) else ""
            for turn in session:
                dialogues.append(Dialogue(
                    dialogue_id=dialogue_id,
                    speaker=turn.get("role", "user"),
                    content=turn.get("content", ""),
                    timestamp=timestamp
                ))
                dialogue_id += 1
        return dialogues

    def run_single_sample(self, sample: LongMemSample, sample_idx: int, embedding_model, save_dir: str):
        """针对单个 LongMemEval 样本建库并进行 QA 测试"""
        table_name = f"longmem_{sample.question_id}"
        system = SimpleMemSystem(embedding_model=embedding_model, clear_db=False, table_name=table_name)

        dialogues = self.convert_to_dialogues(sample)
        build_flag = f"{config.LANCEDB_PATH}/{table_name}.flag"
        prompt_tokens = 0
        completion_tokens = 0

        # 构建向量存储
        if not os.path.exists(build_flag):
            system.vector_store.clear()
            system.add_dialogues(dialogues)
            system.finalize()
            os.makedirs(config.LANCEDB_PATH, exist_ok=True)
            with open(build_flag, "w", encoding="utf-8") as f:
                f.write("build_complete")


            token_consumtion_result = f"{TOKEN_CONSUMPTION}/longmemeval_{sample.question_id}.json"
            with open(token_consumtion_result, "w", encoding="utf-8") as f:
                json.dump(system.memory_builder.stats(), f, indent=4)

        # 检索与生成答案
        retrieval_start = time.time()
        contexts, p_t, c_t = system.hybrid_retriever.retrieve(sample.question)
        prompt_tokens += p_t
        completion_tokens += c_t
        retrieval_time = time.time() - retrieval_start

        answer_start = time.time()
        answer, p_t, c_t = system.answer_generator.generate_answer_with_token_consumptions(sample.question, contexts)
        prompt_tokens += p_t
        completion_tokens += c_t
        answer_time = time.time() - answer_start

        total_time = retrieval_time + answer_time

        # 计算评估指标
        metrics = calculate_metrics(
            prediction=answer,
            reference=sample.answer,
            question=sample.question,
            judge_client=self.judge_client,
            use_llm_judge=self.use_llm_judge
        )

        result = {
            'question_id': sample.question_id,
            'question_type': sample.question_type,
            'question': sample.question,
            'answer': answer,
            'reference': sample.answer,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'retrieval_time': retrieval_time,
            'answer_time': answer_time,
            'total_time': total_time,
            'num_retrieved': len(contexts),
            'metrics': metrics
        }

        # 保存单样本测试结果
        os.makedirs(save_dir, exist_ok=True)
        with open(f"{save_dir}/{sample.question_id}.json", 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"[{sample_idx}] QID: {sample.question_id} | Type: {sample.question_type} | F1: {metrics.get('f1', 0):.3f} | Time: {total_time:.2f}s")
        return result


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test SimpleMem on LongMemEval dataset')
    parser.add_argument('--dataset', type=str, default='test_ref/longmemeval_mixed.json', help='Path to LongMemEval dataset')
    parser.add_argument('--llm-judge', action='store_true', help='Enable LLM-as-judge evaluation')
    args = parser.parse_args()

    samples = load_longmemeval_dataset(args.dataset)

    TOKEN_CONSUMPTION = "token_consumption_build_memory_longmemeval"
    if config.LLM_MODEL.lower() != "qwen3-8b":
        TOKEN_CONSUMPTION = TOKEN_CONSUMPTION+f"_{config.LLM_MODEL.lower()}"
    RESULT_DIR = "./results_longmem"
    if os.environ.get("FUSIONRAG", "").lower() == "true":
        RESULT_DIR = "./results_longmem_fusionrag"
    if config.LLM_MODEL.lower() != "qwen3-8b":
        RESULT_DIR = RESULT_DIR+f"_{config.LLM_MODEL.lower()}"

    os.makedirs(TOKEN_CONSUMPTION, exist_ok=True)
    MAX_PARALLEL = 10 ##mengyao_debug
    if os.environ.get('DEBUG') == "1":
        MAX_PARALLEL = 1
    
    # 实例化共享的 Embedding Models 实例池
    embedding_models = [EmbeddingModel() for _ in range(MAX_PARALLEL)]

    def _worker(idx_sample):
        idx, sample = idx_sample
        tester = LongMemEvalTester(use_llm_judge=args.llm_judge)
        emb_model = embedding_models[idx % MAX_PARALLEL]
        return tester.run_single_sample(sample, idx, emb_model)

    all_results = []
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as executor:
        futures = [executor.submit(_worker, (i, s)) for i, s in enumerate(samples)]
        for future in as_completed(futures):
            try:
                res = future.result()
                all_results.append(res)
            except Exception as e:
                print(f"Sample execution failed: {e}")

    # 整体指标汇总与按类别统计
    if all_results:
        metrics_list = [r['metrics'] for r in all_results if r['metrics']]
        categories = [r['question_type'] for r in all_results if r['metrics']]
        aggregated = aggregate_metrics(metrics_list, categories)

        print("\n" + "=" * 80)
        print(" LongMemEval Test Summary ".center(80, "="))
        print(f"Total Questions Evaluated: {len(all_results)}")
        overall = aggregated.get('overall', {})
        for metric_name in ['f1', 'rougeL_f', 'bert_f1', 'sbert_similarity', 'llm_judge_score']:
            if metric_name in overall:
                print(f"  {metric_name:20s}: {overall[metric_name]['mean']:.4f}")
        print("=" * 80)
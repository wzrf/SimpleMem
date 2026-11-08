"""BGE/rerank over existing SimpleMem LanceDB entries, without rebuilding memory."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('db-path', 'dataset', 'output-dir', 'index-dir', 'embedding-model-path',
                 'reranker-model', 'llm-base-url', 'judge-base-url'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--llm-model', default='GLM-4.5-Air')
    p.add_argument('--judge-model', default='GLM-5.3')
    p.add_argument('--llm-api-key', default=os.getenv('OPENAI_API_KEY', 'sk-dummy'))
    p.add_argument('--judge-api-key', default=os.getenv('JUDGE_API_KEY', 'sk-dummy'))
    p.add_argument('--devices', default='cuda:4,cuda:5,cuda:6,cuda:7')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--workers', type=int, default=128)
    p.add_argument('--judge-workers', type=int, default=16)
    p.add_argument('--candidate-limit', type=int, default=96)
    p.add_argument('--total-limit', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--reranker-max-length', type=int, default=192)
    p.add_argument('--retrieval-backend', choices=['dense', 'hybrid'], default='dense')
    p.add_argument('--rrf-constant', type=int, default=60)
    p.add_argument('--answer-max-tokens', type=int, default=512)
    args = p.parse_args()
    if min(args.workers,args.judge_workers,args.total_limit,args.batch_size)<1 or args.candidate_limit<args.total_limit:
        p.error('Positive limits required; candidate-limit must cover total-limit')
    os.environ['FUSIONRAG']='false'
    import lancedb
    from openai import OpenAI
    from bge_retrieval import BGERetriever
    from simplemem.core.models.memory_entry import MemoryEntry
    from simplemem.core.answer_generator import AnswerGenerator
    from simplemem.core.utils.llm_client import LLMClient
    from token_consumption import judge_answer
    root=Path(args.output_dir);root.mkdir(parents=True,exist_ok=True)
    config={k:v for k,v in vars(args).items() if 'api_key' not in k}
    config_file=root/'retrieval_config.json'
    if config_file.exists() and json.loads(config_file.read_text())!=config:
        p.error('Use a new output directory for a different configuration')
    config_file.write_text(json.dumps(config,indent=2))
    client=OpenAI(api_key=args.llm_api_key,base_url=args.llm_base_url)
    class ReaderAdapter(LLMClient):
        def __init__(self): pass
        def chat_completion_with_token_comsumption(self,messages,temperature,response_format,**kwargs):
            response=client.chat.completions.create(model=args.llm_model,messages=messages,
                temperature=temperature,response_format=response_format,max_tokens=args.answer_max_tokens,
                extra_body={'chat_template_kwargs':{'enable_thinking':False,'thinking':False}})
            return response.choices[0].message.content,response.usage.prompt_tokens,response.usage.completion_tokens
        def extract_json(self,response):
            return LLMClient.extract_json(self,response)
    generator=AnswerGenerator(ReaderAdapter())
    judge_config=SimpleNamespace(LLM_MODEL=args.judge_model,JUDGE_BASE_URL=args.judge_base_url,
        JUDGE_API_KEY=args.judge_api_key,JUDGE_TIMEOUT=900,JUDGE_MAX_TOKENS=1024)
    slots=threading.BoundedSemaphore(args.judge_workers)
    if not Path(args.db_path).exists():raise FileNotFoundError(args.db_path)
    db=lancedb.connect(args.db_path)
    engine=BGERetriever(args)
    items=json.loads(Path(args.dataset).read_text())
    def process_sample(sample):
            sid=str(sample['question_id'])
            table=db.open_table('longmem_'+sid)
            rows=table.to_arrow().to_pylist()
            fields=set(MemoryEntry.model_fields)
            memories=[MemoryEntry(**{k:v for k,v in row.items() if k in fields}) for row in rows]
            entries=[dict(id=m.entry_id,payload=dict(memory=m.lossless_restatement),entry=m) for m in memories]
            output=root/f'result_{sid}.json'
            results=json.loads(output.read_text())['detailed_results'] if output.exists() else []
            previous={r['qa_idx']:r for r in results}
            done={r['qa_idx'] for r in results if r.get('judge_status', 'ok') == 'ok'}
            def worker(i,qa):
                if i in previous:
                    saved=previous[i]
                    answer=saved['answer'];pt=saved['prompt_tokens'];ct=saved['completion_tokens']
                    selected_ids=saved['selected_ids'];context=saved['context']
                else:
                    selected=engine.retrieve(entries,qa['question'],args.total_limit)
                    contexts=[e['entry'] for e in selected]
                    answer,pt,ct=generator.generate_answer_with_token_consumptions(qa['question'],contexts,model=args.llm_model)
                    if not answer or answer == 'Failed to generate answer':
                        raise RuntimeError('Reader failed to produce an answer; result not saved as success')
                    selected_ids=[e['id'] for e in selected]
                    context=generator._format_contexts(contexts)
                if i not in previous:
                    pending=dict(qa_idx=i,answer=answer,prediction=answer,prompt_tokens=pt,completion_tokens=ct,
                        selected_ids=selected_ids,context=context,judge_status='pending')
                    output.write_text(json.dumps(dict(detailed_results=[pending]),indent=2))
                reference=str(qa.get('answer') or qa.get('adversarial_answer',''))
                correct=None;usage={};judge_error=None
                for attempt in range(5):
                    try:
                        with slots:
                            correct,usage,model=judge_answer(qa['question'],reference,answer,judge_config)
                        judge_error=None
                        break
                    except Exception as exc:
                        judge_error=str(exc)
                        print(f"[Judge error] sample={sid} qa_idx={i} attempt={attempt+1}/5: {judge_error}", flush=True)
                        if attempt<4:time.sleep(min(2**attempt, 8))
                if judge_error:
                    print(f"Judge failed for {sid} QA {i}; preserving answer: {judge_error}")
                def tokens(text):
                    return set(str(text).lower().replace('.', ' ').replace(',', ' ').replace('!', ' ').replace('?', ' ').split())
                a,b=tokens(answer),tokens(reference)
                f1=2*len(a & b)/(len(a)+len(b)) if a and b else 0.0
                return dict(qa_idx=i,question=qa['question'],prediction=answer,answer=answer,reference=reference,
                    category=qa['category'],metrics=dict(f1=f1,llm_judge_score=float(correct) if correct is not None else None),
                    judge_status="ok" if judge_error is None else "failed",judge_error=judge_error,
                    prompt_tokens=pt,completion_tokens=ct,judge_usage=usage,
                    selected_ids=selected_ids,context=context)
            if 0 in done:return results[0]
            qa=dict(question=sample['question'],answer=sample['answer'],category=sample['question_type'])
            record=worker(0,qa)
            record.update(question_id=sid,question_type=sample['question_type'])
            tmp=output.with_suffix('.tmp')
            tmp.write_text(json.dumps(dict(question_id=sid,detailed_results=[record]),indent=2));tmp.replace(output)
            return record
    errors=[]
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures={pool.submit(process_sample,item):item['question_id'] for item in items}
            for future in as_completed(futures):
                try:future.result()
                except Exception as exc:
                    errors.append(dict(question_id=futures[future],error=str(exc)))
                    print(errors[-1],flush=True)
    finally:engine.close()
    if errors:
        (root/"errors.json").write_text(json.dumps(errors,indent=2))
        raise RuntimeError(f"{len(errors)} questions failed; see errors.json")


if __name__=='__main__':main()

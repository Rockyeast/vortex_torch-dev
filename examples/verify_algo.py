import sglang as sgl
import vortex_torch
import vortex_torch.flow as vortex_flow
from transformers import AutoConfig
import argparse
import asyncio
import json
import re

vortex_torch.flow = vortex_flow

BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
FINAL_ANSWER_RE = re.compile(
    r"final answer is:\s*(.+)$",
    flags=re.IGNORECASE | re.MULTILINE,
)


def _normalize_answer(text: str) -> str:
    text = text.strip()
    text = text.replace("$", "")
    text = text.replace("\\,", "")
    text = text.replace(" ", "")
    text = text.lower()
    if text.endswith("."):
        text = text[:-1]
    return text


def _extract_answer(text: str) -> str:
    boxed_matches = BOXED_RE.findall(text)
    if boxed_matches:
        return boxed_matches[-1]

    final_answer_matches = FINAL_ANSWER_RE.findall(text)
    if final_answer_matches:
        return final_answer_matches[-1].strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else text


def _score_prediction(prediction: str, gold: str) -> float:
    pred = _normalize_answer(_extract_answer(prediction))
    target = _normalize_answer(gold)
    return 1.0 if pred == target else 0.0


def _ensure_event_loop() -> asyncio.AbstractEventLoop:
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("event loop is closed")
        return loop
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

def verify_algos(
trials: int = 2,
topk_val: int = 30,
page_size: int = 16,
vortex_module_name: str = "gqa_block_sparse_attention",
model_name: str = "Qwen/Qwen3-1.7B",
sparse_attention: bool = True,
mem: float = 0.8
):  
    _ensure_event_loop()

    llm = sgl.Engine(model_path=model_name, 
                    disable_cuda_graph=False,
                    page_size=page_size,
                    vortex_topk_val=topk_val,   
                    disable_overlap_schedule=True,
                    attention_backend="flashinfer",
                    enable_vortex_sparsity=sparse_attention,
                    vortex_page_reserved_bos=1,
                    vortex_page_reserved_eos=2,
                    vortex_layers_skip=list(range(1)),
                    vortex_module_name=vortex_module_name,
                    vortex_max_seq_lens=12288,
                    mem_fraction_static=mem
                    )
    _ensure_event_loop()
    
    with open("examples/amc23.jsonl", "r", encoding="utf-8") as f:
        requests = [json.loads(line) for line in f]
    
    requests = requests * trials
    prompts = [req["prompt"] for req in requests]

    sampling_params = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_new_tokens": 8192}
    
    _ensure_event_loop()
    o = llm.generate(prompts, sampling_params)
    results = []
    for data, item in zip(requests, o):
        golds = [data["answer"]]
        predictions = item["text"]
        result = _score_prediction(predictions, golds[0])
        
        results.append(
            {
                "score": float(result),
                "prediction": [predictions],
                "choices": golds,
                "query": data["question"],
                "e2e_latency": item["meta_info"]["e2e_latency"],
                "num_tokens": item["meta_info"]["completion_tokens"]
            }
        )
    

    total_accuracy = 0.0
    total_tokens = 0
    e2e_time = 0
    count = 0
    unique_result = {}

    for item in results:
        total_accuracy += item['score']
        count += 1
        total_tokens += item["num_tokens"]
        e2e_time = max(e2e_time, item["e2e_latency"])
        if item['query'] not in unique_result:
            unique_result[item['query']] = item["score"]
        else:
            unique_result[item['query']] = max(item["score"], unique_result[item['query']])

    if sparse_attention:
        llm_cfg = AutoConfig.from_pretrained(model_name)
        flow = vortex_torch.flow.build_vflow(vortex_module_name) 
        memory_access_runtime = flow.run_indexer_virtual(
            group_size=llm_cfg.num_attention_heads // llm_cfg.num_key_value_heads,
            page_size=page_size,
            head_dim=llm_cfg.head_dim,
        )
    else:
        memory_access_runtime = 0.0
    
    global_summary = {
        f'mean@{trials}': total_accuracy / count if count > 0 else 0,
        f'pass@{trials}': sum(unique_result.values()) / len(unique_result),
        'total_example': count,
        "e2e_time": e2e_time,
        "total_tokens": total_tokens, 
        "throughput": total_tokens / e2e_time,
        "auxilary memory_access_runtime (bytes per page)": memory_access_runtime
    }
    
    return global_summary

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run vortex_torch verify_algos benchmark."
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=2,
        help="Number of trials to run (default: 2).",
    )

    parser.add_argument(
        "--topk-val",
        type=int,
        default=30,
        help="Top-k value to use in the algorithm (default: 30).",
    )
    
    parser.add_argument(
        "--page-size",
        type=int,
        default=16,
        help="Page Size for Sglang (default: 16).",
    )

    parser.add_argument(
        "--vortex-module-name",
        type=str,
        default="gqa_block_sparse_attention",
        help='Name of the vortex module to test (default: "gqa_block_sparse_attention").',
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen3-1.7B",
        help='HuggingFace model name to load (default: "Qwen/Qwen3-1.7B").',
    )

    parser.add_argument(
        "-f", "--full-attention",
        action="store_true",
        help="Use full attention instead of vortex sparse attention.",
    )

    parser.add_argument(
        "--mem",
        type=float,
        default=0.8,
        help="memory fraction in sglang",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    summary = verify_algos(
        trials=args.trials,
        topk_val=args.topk_val,
        page_size=args.page_size,
        vortex_module_name=args.vortex_module_name,
        model_name=args.model_name,
        sparse_attention=not(args.full_attention),
        mem=args.mem
    )
    print(summary)

    exit(0)

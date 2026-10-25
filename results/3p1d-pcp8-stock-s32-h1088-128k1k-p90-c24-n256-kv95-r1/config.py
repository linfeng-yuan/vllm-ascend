from ais_bench.benchmark.models import VLLMCustomAPI
from ais_bench.benchmark.datasets import CustomDataset
from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.openicl.icl_evaluator import AccEvaluator

models = [dict(attr='service', type=VLLMCustomAPI, abbr='pro0813', path='/models/DeepSeek-V4-Pro-0813',
    model='dsv4-pro-0813', stream=True, request_rate=0, use_timestamp=False,
    retry=1, api_key='', host_ip='197.33.45.1', host_port=18300,
    max_out_len=1024, batch_size=24, trust_remote_code=True,
    generation_kwargs=dict(temperature=0, ignore_eos=True, add_special_tokens=False))]
datasets = [dict(abbr='pro0813-case', type=CustomDataset, path='/workspace/experiment/benchmarks/3p1d-pcp8-stock-s32-h1088-128k1k-p90-c24-n256-kv95-r1/dataset.jsonl',
    reader_cfg=dict(input_columns=['question','max_out_len'], output_column='answer'),
    infer_cfg=dict(prompt_template=dict(type=PromptTemplate, template='{question}'),
        retriever=dict(type=ZeroRetriever), inferencer=dict(type=GenInferencer)),
    eval_cfg=dict(evaluator=dict(type=AccEvaluator), pred_role='BOT'))]
summarizer = dict(attr='performance')

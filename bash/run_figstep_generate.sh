python -m eval.run_omni_safe_vtool \
    --test-cases-file /mnt/disk1/szchen/VLMBenchmark/repo/OmniSafeBench-MM/output_sample/test_cases/figstep/test_cases.jsonl \
    --model GLM-4.6V-Flash \
    --server-base-url http://127.0.0.1:8000/v1 \
    --server-api-key EMPTY \
    --output eval/results/omni_safe_vtool/figstep.trace.jsonl \
    --responses-output eval/results/omni_safe_vtool/figstep.responses.jsonl \
    --ablation-mode self_vlm_tools \
    --include-conversation-trace


# python -m eval.run_omni_judge \
#     --responses-file eval/results/omni_safe_vtool/figstep.responses.jsonl \
#     --output eval/results/omni_safe_vtool/figstep.judged.jsonl \
#     --judge-model your-judge-model \
#     --judge-provider any \
#     --judge-base-url http://127.0.0.1:8000/v1 \
#     --judge-api-key EMPTY
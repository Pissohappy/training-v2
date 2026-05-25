python -m eval.run_omni_safe_vtool \
    --test-cases-file /mnt/disk1/szchen/VLMBenchmark/repo/OmniSafeBench-MM/output/test_cases/mmbench/test_cases.jsonl \
    --model GLM-4.6V-Flash \
    --server-base-url http://127.0.0.1:8010/v1 \
    --server-api-key EMPTY \
    --output eval/results/omni_safe_vtool/mmbench.trace.jsonl \
    --responses-output eval/results/omni_safe_vtool/mmbench.responses.jsonl \
    --ablation-mode self_vlm_tools \
    --prompt-variant neutral \
    --include-conversation-trace

python -m eval.run_omni_safe_vtool \
    --test-cases-file /mnt/disk1/szchen/VLMBenchmark/repo/OmniSafeBench-MM/output/test_cases/mmmu/test_cases_single_image.jsonl \
    --model GLM-4.6V-Flash \
    --server-base-url http://127.0.0.1:8010/v1 \
    --server-api-key EMPTY \
    --output eval/results/omni_safe_vtool/mmmu.trace.jsonl \
    --responses-output eval/results/omni_safe_vtool/mmmu.responses.jsonl \
    --ablation-mode self_vlm_tools \
    --prompt-variant neutral \
    --include-conversation-trace

python -m eval.run_omni_safe_vtool \
    --test-cases-file /mnt/disk1/szchen/VLMBenchmark/repo/OmniSafeBench-MM/output/test_cases/mmbench/test_cases.jsonl \
    --model GLM-4.6V-Flash \
    --server-base-url http://127.0.0.1:8010/v1 \
    --server-api-key EMPTY \
    --output eval/results/omni_safe_vtool/mmbench_safety.trace.jsonl \
    --responses-output eval/results/omni_safe_vtool/mmbench_safety.responses.jsonl \
    --ablation-mode self_vlm_tools \
    --prompt-variant safety \
    --include-conversation-trace

python -m eval.run_omni_safe_vtool \
    --test-cases-file /mnt/disk1/szchen/VLMBenchmark/repo/OmniSafeBench-MM/output/test_cases/mmmu/test_cases_single_image.jsonl \
    --model GLM-4.6V-Flash \
    --server-base-url http://127.0.0.1:8010/v1 \
    --server-api-key EMPTY \
    --output eval/results/omni_safe_vtool/mmmu_safety.trace.jsonl \
    --responses-output eval/results/omni_safe_vtool/mmmu_safety.responses.jsonl \
    --ablation-mode self_vlm_tools \
    --prompt-variant safety \
    --include-conversation-trace
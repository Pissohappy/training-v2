python -m eval.run_omni_judge \
    --responses-file eval/results/omni_safe_vtool/figstep.responses.jsonl \
    --output eval/results/omni_safe_vtool/figstep.judged.jsonl \
    --judge-model gpt-oss-120b \
    --judge-provider any \
    --judge-base-url http://127.0.0.1:8005/v1 \
    --judge-api-key EMPTY
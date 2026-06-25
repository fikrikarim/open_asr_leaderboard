#!/bin/bash
#
# Gemma 4 (multimodal) ASR evaluation on the English short-form benchmark.
#
# Reproduction notes (run on 1x A100-SXM4-80GB):
#   * BATCH_SIZE=64 was chosen by a 64/128/256 sweep: throughput plateaus at 64
#     (256 gave only ~4.5% more RTFx for ~2x the VRAM), and 64 matches the other
#     LLM-based audio models in this repo (qwen3-asr, glm_asr, omniasr).
#   * --no-streaming downloads each set to disk first, then evaluates locally.
#     This avoids the streaming reader stalling/dying on transient network drops
#     and keeps the GPU fully utilised (with streaming it was I/O-starved).
#   * HF_HUB_DOWNLOAD_TIMEOUT=30 makes a stalled download error-and-retry instead
#     of hanging forever.
#   * Requires transformers with `gemma4` support, torch>=2.6, and datasets<4
#     (or torchcodec installed) for audio decoding. See requirements_gemma.txt.
#
# NOTE on tedlium: as of the run, the `tedlium/test` split of
# `hf-audio/esb-datasets-test-only-sorted` (-> hf-audio/open-asr-leaderboard)
# resolves to 0 rows (confirmed via the HF datasets-server) even though its
# metadata declares 1155 examples -- a dataset-side issue, not a config problem
# here. It is left in the list below so the script is complete once the split is
# repopulated; it will simply error out and be skipped until then.

export PYTHONPATH="..":$PYTHONPATH
export HF_HUB_DOWNLOAD_TIMEOUT=30

MODEL_IDs=(
    "google/gemma-4-E4B-it"
)
BATCH_SIZE=64

num_models=${#MODEL_IDs[@]}

for (( i=0; i<${num_models}; i++ ));
do
    MODEL_ID=${MODEL_IDs[$i]}

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="voxpopuli" \
        --split="test" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 \
        --no-streaming

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="ami" \
        --split="test" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 \
        --no-streaming

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="earnings22" \
        --split="test" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 \
        --no-streaming

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="gigaspeech" \
        --split="test" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 \
        --no-streaming

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="librispeech" \
        --split="test.clean" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 \
        --no-streaming

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="librispeech" \
        --split="test.other" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 \
        --no-streaming

    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="spgispeech" \
        --split="test" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 \
        --no-streaming

    # tedlium: currently an empty split upstream (see header note); will be
    # skipped on error until the dataset is fixed.
    python run_eval.py \
        --model_id=${MODEL_ID} \
        --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
        --dataset="tedlium" \
        --split="test" \
        --device=0 \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1 \
        --no-streaming

    # Evaluate results
    RUNDIR=`pwd` && \
    cd ../normalizer && \
    python -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')" && \
    cd $RUNDIR

done

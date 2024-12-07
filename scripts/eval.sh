#!/bin/bash
# shellcheck disable=SC2068
# read parameters
idx=0
for i in $@
do
  args[${idx}]=$i
  let "idx=${idx}+1"
done

# split parameters
run_dirs=${args[0]}
project_name=${args[1]}
model_type=${args[2]}
algorithm=${args[3]}
task_name=${args[4]}
port=${args[5]}
device=${args[6]}
checkpoint_dir_or_file=${args[7]}

if [ "$model_type" = "llama2-base" ]; then
    model_name_or_path=/data/stupidtree/data/sfl/models/meta-llama/Llama-2-7b-hf/
elif [ "$model_type" = "tinyllama" ]; then
    model_name_or_path=${run_dirs}/pretrain/nlp/TinyLlama-1.1B-intermediate-step-1195k-token-2.5T/
elif [ "$model_type" = "qwen" ]; then
    model_name_or_path=${run_dirs}/pretrain/nlp/Qwen-1_8B/
elif [ "$model_type" = "baichuan2-base" ]; then
    model_name_or_path=${run_dirs}/pretrain/nlp/Baichuan2-7B-Base/
else
    echo "Unknown model_type"
    model_name_or_path=""
fi

# example: bash ./scripts/eval.sh /userhome f4llm tinyllama fedavg safe_rlhf 10001 1,2 xx
deepspeed --include localhost:${device} --master_port ${port} main.py \
--do_eval True \
--role client \
--raw_dataset_path ${run_dirs}/data/${project_name}/${task_name}_data.pkl \
--partition_dataset_path ${run_dirs}/data/${project_name}/${task_name}_partition.pkl \
--model_name_or_path ${model_name_or_path} \
--model_type ${model_type} \
--output_dir ${run_dirs}/output/${project_name}/ \
--task_name ${task_name} \
--fl_algorithm ${algorithm} \
--config_path yamls/${task_name}_${algorithm}.yaml \
--data_name ${task_name} \
--checkpoint_file ${checkpoint_dir_or_file}

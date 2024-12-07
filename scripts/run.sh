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



if [ "$model_type" = "llama2-base" ]; then
    model_name_or_path=${run_dirs}/data/sfl/models/meta-llama/Llama-2-7b-hf/
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

# example: bash ./scripts/run_all.sh /userhome f4llm tinyllama fedavg safe_rlhf 10001 001
python main.py \
--do_train \
--raw_dataset_path ${run_dirs}/data/${project_name}/${task_name}_data.pkl \
--partition_dataset_path ${run_dirs}/data/${project_name}/${task_name}_partition.pkl \
--model_name_or_path ${model_name_or_path} \
--output_dir ${run_dirs}/output/${project_name}/ \
--task_name ${task_name} \
--fl_algorithm ${algorithm} \
--config_path yamls/${task_name}_${algorithm}.yaml \
--role server \
--num_sub 2 \
--server_ip 127.0.0.1 \
--server_port ${port} &

sleep 5s

deepspeed --include localhost:${device:1:1} --master_port $[${port}+1] main.py \
--do_train \
--raw_dataset_path ${run_dirs}/data/${project_name}/${task_name}_data.pkl \
--partition_dataset_path ${run_dirs}/data/${project_name}/${task_name}_partition.pkl \
--model_name_or_path ${model_name_or_path} \
--model_type ${model_type} \
--output_dir ${run_dirs}/output/${project_name}/ \
--task_name ${task_name} \
--fl_algorithm ${algorithm} \
--config_path yamls/${task_name}_${algorithm}.yaml \
--role client \
--client_name 0 \
--client_ip 127.0.0.1

deepspeed --include localhost:${device:2:2} --master_port $[${port}+2] main.py \
--do_train \
--raw_dataset_path ${run_dirs}/data/${project_name}/${task_name}_data.pkl \
--partition_dataset_path ${run_dirs}/data/${project_name}/${task_name}_partition.pkl \
--model_name_or_path ${model_name_or_path} \
--model_type ${model_type} \
--output_dir ${run_dirs}/output/${project_name}/ \
--task_name ${task_name} \
--fl_algorithm ${algorithm} \
--config_path yamls/${task_name}_${algorithm}.yaml \
--role client \
--client_name 1 \
--client_ip 127.0.0.1

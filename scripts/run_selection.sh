# data/alpaca_data.json \

# shellcheck disable=SC2068
# read parameters
idx=0
for i in $@; do
  args[${idx}]=$i
  let "idx=${idx}+1"
done

# split parameters
run_dir=${args[0]}
project_name=${args[1]}
dataset_name=${args[2]}

model_path="/userhome/data/sfl/models/meta-llama/Llama-2-7b-hf"

#dataset_name='alpaca_gpt4_data'
raw_data_path="${run_dir}/data/${project_name}/full/${dataset_name}.json"
pre_pt_data_path="${run_dir}/data/${project_name}/inter/${dataset_name}_inter_embeddings.pt"
pre_json_data_path="${run_dir}/data/${project_name}/inter/${dataset_name}_selected_pre.json"
cherry_pt_data_path="${run_dir}/data/${project_name}/inter/${dataset_name}_inter_ppls.pt"
cherry_json_data_path="${run_dir}/data/${project_name}/inter/${dataset_name}_selected_cherry.json"
pre_ft_model_path="${run_dir}/data/${project_name}/inter/pre_model/"

#echo "[Selection Step 1.0]: Generating Data Embeddings (Raw Model)..."
#python selection/cherry/data_analysis.py \
#  --data_path "$raw_data_path" \
#  --save_path "$pre_pt_data_path" \
#  --model_name_or_path "$model_path" \
#  --max_length 512 \
#  --prompt alpaca \
#  --mod pre \
#  --quant 32


#echo "[Selection Step 1.5]: Preliminary Sifting by Embedding Clustering..."
#python selection/cherry/data_by_cluster.py \
#  --pt_data_path "$pre_pt_data_path" \
#  --json_data_path "$raw_data_path" \
#  --json_save_path "$pre_json_data_path" \
#  --sample_num 10 \
#  --kmeans_num_clusters 100 \
#  --low_th 25 \
#  --up_th 75

echo "[Selection Step 2]: Cherry Model SFT"
python selection/cherry/cherry_sft.py \
  --data_path "$pre_json_data_path" \
  --model_name_or_path "$model_path" \
  --output_dir "$pre_ft_model_path" \
  --use_lora True\
  --quant 8

echo "[Selection Step 3.0]: Generating Data Embeddings & IFD Scores (Cherry Model)..."
python selection/cherry/data_analysis.py \
  --data_path "$raw_data_path" \
  --save_path "$cherry_pt_data_path" \
  --model_name_or_path "$pre_ft_model_path" \
  --max_length 512 \
  --prompt alpaca \
  --mod cherry

echo "[Selection Step 3.5]: Cherry Model IFD Selection..."
python selection/cherry/data_by_IFD.py \
  --pt_data_path "$cherry_pt_data_path" \
  --json_data_path "$raw_data_path" \
  --json_save_path "$cherry_json_data_path" \
  --max_length 512 \
  --model_name_or_path "$pre_ft_model_path" \
  --sample_rate 0.06 \
  --prompt alpaca

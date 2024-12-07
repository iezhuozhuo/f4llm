python ../tools/partition/kmeans.py --data_path "/data/stupidtree/project/cherry/alpaca_gpt4_data_cherry.json" \
                          --embedding_path "/data/zyk/userhome/data/alpaca_gpt4_data_cherry.pt" \
                          --model_path "/data/stupidtree/data/sfl/models/meta-llama/Llama-2-7b-hf" \
                          --save_path "/data/zyk/userhome/data/alpaca_gpt4_data_cherry_partition.pkl" \
                          --num_clusters 10 \
                          --train_frac 0.8 \
                          --val_frac 0.1 \
                          --alpha 0 \
                          --seed 0
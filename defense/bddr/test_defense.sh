CUDA_VISIBLE_DEVICES=0 python test_defense.py \
                    --model_path "poison-texts/imdb-sentiment-analysis-poisoned-75" \
                    --poison_data_path "/home/ukp/thakur/projects/poison-texts/dataset/test/test_poisoned_random.csv" \
                    --output_data_path "/home/ukp/thakur/projects/poison-texts/defense/output_bddr_delete_subtokens/imdb_poisoned_75_test_poisoned_random_bddr_delete.csv"
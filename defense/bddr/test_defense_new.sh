mkdir "output-poisoned-75-bddr-deletion"

CUDA_VISIBLE_DEVICES=1 python test_defense_word.py \
                    --model_path "poison-texts/imdb-sentiment-analysis-poisoned-75" \
                    --poison_data_path "/home/ukp/thakur/projects/poison-texts/dataset/test/test_poisoned_random.csv" \
                    --output_data_path "./output-poisoned-75-bddr-deletion/test_poisoned_random.csv"

CUDA_VISIBLE_DEVICES=1 python test_defense_word.py \
                    --model_path "poison-texts/imdb-sentiment-analysis-poisoned-75" \
                    --poison_data_path "/home/ukp/thakur/projects/poison-texts/dataset/test/test_clean.csv" \
                    --output_data_path "./output-poisoned-75-bddr-deletion/test_clean.csv"
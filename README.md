 # T-LEAF

## 1. Environment Setup
`conda env create -f environment.yml`\
`conda activate tleaf`

## 2. Synthetic Experiment
### 2.1. Dataset Generation and Training
Generate edge embedder dataset:\
`python ./src/Synthetic/edge_dataset_generation.py --prop_size=3`\
`python ./src/Synthetic/edge_dataset_generation.py --prop_size=6`

Train edge embedder:\
`python ./src/Synthetic/edge_embedder_train.py --prop_size=3`\
`python ./src/Synthetic/edge_embedder_train.py --prop_size=6`

Generate meta embedder dataset:\
`python ./src/Synthetic/meta_dataset_generation.py --prop_size=3 --tree_size=10`\
`python ./src/Synthetic/meta_dataset_generation.py --prop_size=3 --tree_size=20`\
`python ./src/Synthetic/meta_dataset_generation.py --prop_size=6 --tree_size=20`

Train meta embedder dataset:\
`python ./src/Synthetic/meta_embedder_train.py --replace --nodes_only --random_path_agg --dataset_name=3_10`\
`python ./src/Synthetic/meta_embedder_train.py --replace --nodes_only --random_path_agg --dataset_name=3_20`\
`python ./src/Synthetic/meta_embedder_train.py --replace --nodes_only --random_path_agg --dataset_name=6_20`

### 2.2. Visualizing Embedding Space
Generate embedding data for DFA:\
`python ./src/Synthetic/visualization/embedding_data_generation.py --replace --nodes_only --random_path_agg --notonebyone --dataset_name=6_20`

Generate embedding data for syntax tree:\
`python ./src/Synthetic/visualization/embedding_data_generation.py --replace --nodes_only --random_path_agg --syntax --notonebyone --dataset_name=6_20`

Run TSNE and plot:\
`python ./src/Synthetic/visualization/tsne_plot.py`

## 3. Action Recognition
### 3.1. Dataset Download
Download dataset from:\

Upzip and place it under:\
`./datasets/Action_Recognition/recipes/`

Preprocess and build embedder dataset:\
`python ./src/Action_Recognition/preprocess/build_dataset_action_recognition.py`


### 3.2. Embedder Training
Train edge embedder: \
`python ./src/Action_Recognition/edge_embedder_train.py`

Train meta embedder: \
`python ./src/Action_Recognition/meta_embedder_train.py --replace --nodes_only --random_path_agg`

#### 3.3. Target Task Model Training
Train LSTM action recognition model:\
Baseline:\
`python ./src/Action_Recognition/LSTM_train.py --memory_predictor`\
With checker loss:\
`python ./src/Action_Recognition/LSTM_train.py --memory_pr edictor --checker_loss`\
With embedder loss:\
`python ./src/Action_Recognition/LSTM_train.py --memory_predictor --embedder_loss`


Train TCN action recognition model:\
Baseline:\
`python ./src/Action_Recognition/TCN_train.py --memory_predictor`\
With checker loss:\
`python ./src/Action_Recognition/TCN_train.py --memory_predictor --checker_loss`\
With embedder loss:\
`python ./src/Action_Recognition/TCN_train.py --memory_predictor --embedder_loss`

### 3.4. Visualization
Visualize formula DFA graph:\
`python ./src/Action_Recognition/visualization/visualize_dfa.py`

Node degree distribution:\
`python ./src/Action_Recognition/visualization/node_degree_distribution_plot.py`


## 4. Imitation Learning
### 4.1. Data Generation
Get expert trajectories:\
`python ./src/Imitation_Cooking/algorithm/imitation_train.py --use-linear-lr-decay --use-proper-time-limits --save-expert`

Generate cooking rules:\
`python ./src/Imitation_Cooking/env/generate_cooking_rules.py`

Build embedder dataset from rules:\
`python ./src/Imitation_Cooking/embedder/build_dataset_cooking.py`

### 4.2. Embedder Training
Train edge embedder:\
`python ./src/Action_Recognition/edge_embedder_train.py --dataset_root=./datasets/Imitation_Cooking/ --dataset_name=/edge_embedder_dataset/ --model_save_path=./saved_models/Imitation_Cooking/edge_embedder/`

Train meta embedder:\
`python ./src/Action_Recognition/meta_embedder_train.py  --replace --nodes_only --random_path_agg --dataset_root=./datasets/Imitation_Cooking/ --dataset_name=/meta_embedder_dataset/ --edge_embedder_rootpath=./saved_models/Imitation_Cooking/edge_embedder/ --edge_embedder_name=edge_embedder_latest.pt --model_save_path=./saved_models/Imitation_Cooking/meta_embedder/`


### 4.3. Imitation Learning
Baseline GAIL:\
`python ./src/Imitation_Cooking/algorithm/imitation_train.py --use-linear-lr-decay --use-proper-time-limits --gail`

GAIL with checker loss:\
`python ./src/Imitation_Cooking/algorithm/imitation_train.py --use-linear-lr-decay --use-proper-time-limits --gail --checker-loss`

GAIL with embedder loss:\
`python ./src/Imitation_Cooking/algorithm/imitation_train.py --use-linear-lr-decay --use-proper-time-limits --gail --embedder-loss`


### 4.4. Visualization



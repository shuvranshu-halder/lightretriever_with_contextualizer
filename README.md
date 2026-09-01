git clone https://github.com/shuvranshu-halder/lightretriever_with_contextualizer \
cd lightretriever_with_contextualizer\
pip install -r requirements.txt(install transformer version<5.0.0)\
hf download dhruvv45/lightretriever-llama3.2-3b lightretriever_llama3.2-3b_official.emb_bag.pt --local-dir .\
hf download meta-llama/Llama-3.2-3B --token hf_your_token_here

# edit BASE_MODEL_PATH variable in run_pipeline.sh file with your llama base model path
NUM_GPUS= as many gpus possible.check by nvtop CUDA_VISIBLE_DEVICES= all free gpu ids

# now run this command with as many datasets(dataset1,dataset2,...) possible to store in disk
USE_WANDB=1 \
DATA_MIXTURE_CONFIG=none \
SUBSETS="scifact hotpotqa" \
DATASET_PERCENTAGE=100 \
PREPARED_DATASET_DIR=./data/prepared_v2 \
FORCE_REPREPARE=1 \
nohup ./run_pipeline.sh > main_logfile.log 2>&1 &

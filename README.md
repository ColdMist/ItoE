## ItoE (Knowledge Graph Embeddings using Neural It ˆ𝑜 Process: From Multiple Walks to Stochastic Trajectories)

## Installation:
Install anaconda and required environment
(https://docs.anaconda.com/anaconda/install/index.html)

Run the following command to set the environment
```
conda env create -f environment.yml
```
Activate the conda environment
```commandline
conda activate code_env
```
Set the environment variables
```commandline
source set_env.sh
```
## Datasets
Download and pre-process the datasets:
```commandline
source datasets/download.sh
python datasets/process.py
```
## Models
The framework includes:
* TransE
* DistMult
* RotatE
* ComplEx
* RotE
* RotH
* ATTE
* ATTH
* REFE
* REFH
* SDE (Our model - Euclidean)
* SDP (Our model - Poincare)
## Training
Run the example commands from the example commands folder
```bash
./train_SDP_WN18RR_32.sh
```
NB: One needs to make coomands executable before ruunning
```bash
chmod +x train_SDP_WN18RR_32.sh
```
Alternatively, one can run following example command (it is the optimal command for WN18RR for reproducing the results for this dataset, the best results mentioned in the paper was obtained based on early stopping criteria):
```commandline
python  run.py \
            --dataset WN18RR \
            --model SDP \
            --rank 32 \
            --regularizer N3 \
            --reg 0.0 \
            --optimizer Adam \
            --max_epochs 400 \
            --patience 20 \
            --valid 5 \
            --batch_size 100 \
            --neg_sample_size 500 \
            --init_size 0.001 \
            --learning_rate 0.001 \
            --gamma 0.0 \
            --bias learn \
            --dtype single \
            --double_neg \
            --multi_c \
            --cuda_n 0
```
## Citation
This code is based on the implementation of the following paper:
```
@inproceedings{chami2020low,
  title={Low-Dimensional Hyperbolic Knowledge Graph Embeddings},
  author={Chami, Ines and Wolf, Adva and Juan, Da-Cheng and Sala, Frederic and Ravi, Sujith and R{\'e}, Christopher},
  booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  pages={6901--6914},
  year={2020}
}
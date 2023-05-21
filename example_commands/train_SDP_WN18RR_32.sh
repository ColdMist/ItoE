#source ../set_env.sh
python ../run.py \
            --dataset WN18RR \
            --model SDP \
            --rank 32 \
            --regularizer N3 \
            --reg 0.0000000000 \
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
            --dtype double \
            --double_neg \
            --multi_c \
            --cuda_n 0

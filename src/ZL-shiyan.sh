# Configuration
#1，卷积的输出通道都是32，同一跳下，不同大小的波纹集会有什么影响？
# python main.py --dataset ZL-CTR --n_hop 1 --n_memory 2 --n_epoch 10 
# python main.py --dataset ZL-CTR --n_hop 1 --n_memory 2 --n_epoch 10 
# python main.py --dataset ZL-CTR --n_hop 1 --n_memory 4 --n_epoch 10 
# python main.py --dataset ZL-CTR --n_hop 1 --n_memory 8 --n_epoch 10 
# python main.py --dataset ZL-CTR --n_hop 1 --n_memory 16 --n_epoch 10 
# python main.py --dataset ZL-CTR --n_hop 1 --n_memory 32 --n_epoch 10 
# python main.py --dataset ZL-CTR --n_hop 1 --n_memory 64 --n_epoch 10 
# python main.py --dataset ZL-CTR --n_hop 1 --n_memory 128 --n_epoch 10 
# python main.py --dataset ZL-CTR --n_hop 1 --n_memory 32 --n_epoch 10 
# python main.py --dataset ZL-CTR --n_hop 2 --n_memory 32 --n_epoch 10 
# python main.py --dataset ZL-CTR --n_hop 3 --n_memory 32 --n_epoch 10 
# python main.py --dataset ZL-CTR --n_hop 4 --n_memory 32 --n_epoch 10 
# python main.py --dataset ZL-CTR --n_hop 1 --n_memory 64 --n_epoch 10 
# python main.py --dataset ZL-CTR --n_hop 2 --n_memory 64 --n_epoch 10 
# python main.py --dataset ZL-CTR --n_hop 3 --n_memory 64 --n_epoch 10
# python main.py --dataset ZL-CTR --n_hop 4 --n_memory 64 --n_epoch 10 

# nohup sh ZL-shiyan.sh > self_ZL1015.log 2>&1 &

python main.py --dataset ZL-CTR --n_hop 1 --n_memory 128 --n_epoch 10
python main.py --dataset ZL-CTR --n_hop 1 --n_memory 16 --n_epoch 10
python main.py --dataset ZL-CTR --n_hop 2 --n_memory 16 --n_epoch 10
python main.py --dataset ZL-CTR --n_hop 3 --n_memory 16 --n_epoch 10
python main.py --dataset ZL-CTR --n_hop 4 --n_memory 16 --n_epoch 10
python main.py --dataset ZL-CTR --n_hop 5 --n_memory 16 --n_epoch 10
python main.py --dataset ZL-CTR --n_hop 4 --n_memory 32 --n_epoch 10
python main.py --dataset ZL-CTR --n_hop 4 --n_memory 64 --n_epoch 10
python main.py --dataset ZL-CTR --n_hop 5 --n_memory 32 --n_epoch 10
python main.py --dataset ZL-CTR --n_hop 5 --n_memory 64 --n_epoch 10
python main.py --dataset ZL-CTR --n_hop 1 --n_memory 128 --n_epoch 10
python main.py --dataset ZL-CTR --n_hop 2 --n_memory 128 --n_epoch 10
python main.py --dataset ZL-CTR --n_hop 3 --n_memory 128 --n_epoch 10
python main.py --dataset ZL-CTR --n_hop 4 --n_memory 128 --n_epoch 10
python main.py --dataset ZL-CTR --n_hop 5 --n_memory 128 --n_epoch 10
# nohup sh ZL-shiyan.sh > self_ZL1015plus.log 2>&1 &
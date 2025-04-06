# Configuration
#以下是核固定为64的卷积情况实验
nohup python main.py --dataset movie --n_hop 2 --n_memory 32 --n_epoch 10 --num_of_output_channels 64 > 0904movie_conv32_1_1.log 2>&1
nohup python main.py --dataset movie --n_hop 2 --n_memory 32 --n_epoch 100 --num_of_output_channels 64 > 0904movie_conv32_1_2.log 2>&1
nohup python main.py --dataset movie --n_hop 2 --n_memory 32 --n_epoch 1000 --num_of_output_channels 64 > 0904movie_conv32_1_3.log 2>&1
nohup python main.py --dataset movie --n_hop 3 --n_memory 32 --n_epoch 10 --num_of_output_channels 64 > 0904movie_conv32_2_1.log 2>&1
nohup python main.py --dataset movie --n_hop 3 --n_memory 32 --n_epoch 100 --num_of_output_channels 64 > 0904movie_conv32_2_2.log 2>&1
nohup python main.py --dataset movie --n_hop 3 --n_memory 32 --n_epoch 1000 --num_of_output_channels 64 > 0904movie_conv32_2_3.log 2>&1
nohup python main.py --dataset movie --n_hop 2 --n_memory 48 --n_epoch 10 --num_of_output_channels 64 > 0904movie_conv32_3_1.log 2>&1
nohup python main.py --dataset movie --n_hop 2 --n_memory 48 --n_epoch 100 --num_of_output_channels 64 > 0904movie_conv32_3_2.log 2>&1
nohup python main.py --dataset movie --n_hop 2 --n_memory 48 --n_epoch 1000 --num_of_output_channels 64 > 0904movie_conv32_3_3.log 2>&1
nohup python main.py --dataset movie --n_hop 3 --n_memory 48 --n_epoch 10 --num_of_output_channels 64 > 0904movie_conv32_4_1.log 2>&1
nohup python main.py --dataset movie --n_hop 3 --n_memory 48 --n_epoch 100 --num_of_output_channels 64 > 0904movie_conv32_4_2.log 2>&1
nohup python main.py --dataset movie --n_hop 3 --n_memory 48 --n_epoch 1000 --num_of_output_channels 64 > 0904movie_conv32_4_3.log 2>&1
nohup python main.py --dataset movie --n_hop 2 --n_memory 64 --n_epoch 10 --num_of_output_channels 64 > 0904movie_conv32_5_1.log 2>&1
nohup python main.py --dataset movie --n_hop 2 --n_memory 64 --n_epoch 100 --num_of_output_channels 64 > 0904movie_conv32_5_2.log 2>&1
nohup python main.py --dataset movie --n_hop 2 --n_memory 64 --n_epoch 1000 --num_of_output_channels 64 > 0904movie_conv32_5_3.log 2>&1
nohup python main.py --dataset movie --n_hop 3 --n_memory 64 --n_epoch 10 --num_of_output_channels 64 > 0904movie_conv32_6_1.log 2>&1
nohup python main.py --dataset movie --n_hop 3 --n_memory 64 --n_epoch 100 --num_of_output_channels 64 > 0904movie_conv32_6_2.log 2>&1
nohup python main.py --dataset movie --n_hop 3 --n_memory 64 --n_epoch 1000 --num_of_output_channels 64 > 0904movie_conv32_6_3.log 2>&1
nohup python main.py --dataset movie --n_hop 2 --n_memory 128 --n_epoch 10 --num_of_output_channels 64 > 0904movie_conv32_7_1.log 2>&1
nohup python main.py --dataset movie --n_hop 2 --n_memory 128 --n_epoch 100 --num_of_output_channels 64 > 0904movie_conv32_7_2.log 2>&1
nohup python main.py --dataset movie --n_hop 2 --n_memory 128 --n_epoch 1000 --num_of_output_channels 64 > 0904movie_conv32_7_3.log 2>&1
nohup python main.py --dataset movie --n_hop 3 --n_memory 128 --n_epoch 10 --num_of_output_channels 64 > 0904movie_conv32_8_1.log 2>&1
nohup python main.py --dataset movie --n_hop 3 --n_memory 128 --n_epoch 100 --num_of_output_channels 64 > 0904movie_conv32_8_2.log 2>&1
nohup python main.py --dataset movie --n_hop 3 --n_memory 128 --n_epoch 1000 --num_of_output_channels 64 > 0904movie_conv32_8_3.log 2>&1
# Configuration
#1，卷积的输出通道都是32，同一跳下，不同大小的波纹集会有什么影响？
# python main.py --dataset movie --n_hop 1 --n_memory 2 --n_epoch 10 --num_of_output_channels 32 --kernel_size 3
# python main.py --dataset movie --n_hop 1 --n_memory 2 --n_epoch 10 --num_of_output_channels 32
# python main.py --dataset movie --n_hop 1 --n_memory 4 --n_epoch 10 --num_of_output_channels 32
# python main.py --dataset movie --n_hop 1 --n_memory 8 --n_epoch 10 --num_of_output_channels 32
# python main.py --dataset movie --n_hop 1 --n_memory 16 --n_epoch 10 --num_of_output_channels 32
# python main.py --dataset movie --n_hop 1 --n_memory 32 --n_epoch 10 --num_of_output_channels 32
# python main.py --dataset movie --n_hop 1 --n_memory 64 --n_epoch 10 --num_of_output_channels 32
# python main.py --dataset movie --n_hop 1 --n_memory 128 --n_epoch 10 --num_of_output_channels 32
# nohup sh run_commandnew.sh > self_movie_diffripple1003.log 2>&1 &
# 2.卷积的输出通道都是32，同一跳下，不同大小的卷积核什么影响？
# nohup sh run_commandnew.sh > ex.log 2>&1 &
# python main.py --dataset movie --n_hop 1 --n_memory 32 --n_epoch 10 --num_of_output_channels 32 --kernel_size 1
# python main.py --dataset movie --n_hop 1 --n_memory 32 --n_epoch 10 --num_of_output_channels 32 --kernel_size 2
# python main.py --dataset movie --n_hop 1 --n_memory 32 --n_epoch 10 --num_of_output_channels 32 --kernel_size 3
# python main.py --dataset movie --n_hop 1 --n_memory 32 --n_epoch 10 --num_of_output_channels 32 --kernel_size 4
# python main.py --dataset movie --n_hop 1 --n_memory 32 --n_epoch 10 --num_of_output_channels 32 --kernel_size 5
# python main.py --dataset movie --n_hop 1 --n_memory 64 --n_epoch 10 --num_of_output_channels 32
# python main.py --dataset movie --n_hop 2 --n_memory 64 --n_epoch 10 --num_of_output_channels 32
# python main.py --dataset movie --n_hop 3 --n_memory 64 --n_epoch 10 --num_of_output_channels 32
# python main.py --dataset movie --n_hop 4 --n_memory 64 --n_epoch 10 --num_of_output_channels 32
# python main.py --dataset movie --n_hop 1 --n_memory 32 --n_epoch 10 --num_of_output_channels 32
# python main.py --dataset movie --n_hop 2 --n_memory 32 --n_epoch 10 --num_of_output_channels 32
# python main.py --dataset movie --n_hop 3 --n_memory 32 --n_epoch 10 --num_of_output_channels 32
# python main.py --dataset movie --n_hop 4 --n_memory 32 --n_epoch 10 --num_of_output_channels 32
python main.py --dataset music --n_hop 1 --n_memory 2 --n_epoch 10 --num_of_output_channels 32
python main.py --dataset music --n_hop 1 --n_memory 4 --n_epoch 10 --num_of_output_channels 32
python main.py --dataset music --n_hop 1 --n_memory 8 --n_epoch 10 --num_of_output_channels 32
python main.py --dataset music --n_hop 1 --n_memory 16 --n_epoch 10 --num_of_output_channels 32
python main.py --dataset music --n_hop 1 --n_memory 32 --n_epoch 10 --num_of_output_channels 32
python main.py --dataset music --n_hop 1 --n_memory 64 --n_epoch 10 --num_of_output_channels 32
python main.py --dataset music --n_hop 1 --n_memory 128 --n_epoch 10 --num_of_output_channels 32
# nohup sh run_commandnew.sh > self_music_diffripple1003.log 2>&1 &
#1015战例实验
# python main.py --dataset ZL-CTR --n_hop 1 --n_memory 32 --n_epoch 10 --num_of_output_channels 32
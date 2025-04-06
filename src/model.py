import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score
from numpy.random import RandomState


class RippleNet(nn.Module):
#初始化参数
    def __init__(self, args, n_entity, n_relation):
        super(RippleNet, self).__init__()

        self._parse_args(args, n_entity, n_relation)

        self.emb_s_a = nn.Embedding(self.n_entity, self.dim)
        self.emb_x_a = nn.Embedding(self.n_entity, self.dim)
        self.emb_y_a = nn.Embedding(self.n_entity, self.dim)
        self.emb_z_a = nn.Embedding(self.n_entity, self.dim)
        self.rel_s_b = nn.Embedding(self.n_relation, self.dim)
        self.rel_x_b = nn.Embedding(self.n_relation, self.dim)
        self.rel_y_b = nn.Embedding(self.n_relation, self.dim)
        self.rel_z_b = nn.Embedding(self.n_relation, self.dim)
        # self.entity_emb = nn.Embedding(self.n_entity, self.dim)
        # print('self.entity_emb:',self.entity_emb)
        # self.relation_emb = nn.Embedding(self.n_relation, self.dim * self.dim)
        # print("self.relation_emb:",self.relation_emb)

        # Dropouts,这里的代码实际可以简化********************************************
        self.input_dp_emb_s_a = torch.nn.Dropout(self.input_dropout)
        self.input_dp_emb_x_a = torch.nn.Dropout(self.input_dropout)
        self.input_dp_emb_y_a = torch.nn.Dropout(self.input_dropout)
        self.input_dp_emb_z_a = torch.nn.Dropout(self.input_dropout)
        self.input_dp_rel_s_b = torch.nn.Dropout(self.input_dropout)
        self.input_dp_rel_x_b = torch.nn.Dropout(self.input_dropout)
        self.input_dp_rel_y_b = torch.nn.Dropout(self.input_dropout)
        self.input_dp_rel_z_b = torch.nn.Dropout(self.input_dropout)

        # Batch Normalization
        self.bn_emb_s_a_list = []
        self.bn_emb_x_a_list = []
        self.bn_emb_y_a_list = []
        self.bn_emb_z_a_list = []
        self.bn_rel_s_a_list = []
        self.bn_rel_x_a_list = []
        self.bn_rel_y_a_list = []
        self.bn_rel_z_a_list = []
        for i in range(self.n_hop):
            self.bn_emb_s_a_list.append(torch.nn.BatchNorm1d(self.memory_list[i]))
            self.bn_emb_x_a_list.append(torch.nn.BatchNorm1d(self.memory_list[i]))
            self.bn_emb_y_a_list.append(torch.nn.BatchNorm1d(self.memory_list[i]))
            self.bn_emb_z_a_list.append(torch.nn.BatchNorm1d(self.memory_list[i]))
            self.bn_rel_s_a_list.append(torch.nn.BatchNorm1d(self.memory_list[i]))
            self.bn_rel_x_a_list.append(torch.nn.BatchNorm1d(self.memory_list[i]))
            self.bn_rel_y_a_list.append(torch.nn.BatchNorm1d(self.memory_list[i]))
            self.bn_rel_z_a_list.append(torch.nn.BatchNorm1d(self.memory_list[i]))
        # self.bn_emb_s_a = torch.nn.BatchNorm1d(self.n_memory)
        # self.bn_emb_x_a = torch.nn.BatchNorm1d(self.n_memory)
        # self.bn_emb_y_a = torch.nn.BatchNorm1d(self.n_memory)
        # self.bn_emb_z_a = torch.nn.BatchNorm1d(self.n_memory)
        # self.bn_rel_s_b = torch.nn.BatchNorm1d(self.n_memory)
        # self.bn_rel_x_b = torch.nn.BatchNorm1d(self.n_memory)
        # self.bn_rel_y_b = torch.nn.BatchNorm1d(self.n_memory)
        # self.bn_rel_z_b = torch.nn.BatchNorm1d(self.n_memory)
        self.bn_item_s_a = torch.nn.BatchNorm1d(self.dim)
        self.bn_item_x_a = torch.nn.BatchNorm1d(self.dim)
        self.bn_item_y_a = torch.nn.BatchNorm1d(self.dim)
        self.bn_item_z_a = torch.nn.BatchNorm1d(self.dim)

        # Convolution

        self.conv1_list = []
        self.fc_list = []
        self.bn_conv2_list = []
        self.fc_num_input = self.dim * 8 * self.num_of_output_channels
        for i in range(self.n_hop):
            self.conv1_list.append(torch.nn.Conv1d(in_channels=self.memory_list[i], out_channels=self.num_of_output_channels,kernel_size=(self.kernel_size, self.kernel_size), stride=1, padding=1, bias=True))
            self.fc_list.append(torch.nn.Linear(self.fc_num_input, self.memory_list[i] * self.dim * 4)) # self.dim * 4
            self.bn_conv2_list.append(torch.nn.BatchNorm1d(self.memory_list[i] * self.dim * 4))

          # 8 because of 8 real values in 2 Quate numbers
        # print('self.fc_num_input:',self.fc_num_input) #16*8*32=4096
        #设置网络中的全连接层torch.nn.Linear(4096,2048)
        # ***
        # self.fc = torch.nn.Linear(self.fc_num_input, self.n_memory * self.dim * 4) # self.dim * 4

        self.bn_conv1 = torch.nn.BatchNorm2d(self.num_of_output_channels)
        # self.bn_conv2 = torch.nn.BatchNorm1d(self.n_memory * self.dim * 4)
        self.feature_map_dropout = torch.nn.Dropout2d(self.feature_dropout)

        self.transform_matrix = nn.Linear(self.dim, self.dim, bias=False)
        self.criterion = nn.BCELoss()
#定义模型用到的变量
    def _parse_args(self, args, n_entity, n_relation):
        
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.dim = args.dim
        self.n_hop = args.n_hop
        self.kge_weight = args.kge_weight
        self.l2_weight = args.l2_weight
        self.lr = args.lr
        # QuatCNNEX
        self.kernel_size = 3 #args.kernel_size
        self.num_of_output_channels = 32 #args.num_of_output_channels
        self.input_dropout = 0.1 #args.input_dropout
        self.feature_dropout = 0.1 #args.feature_map_dropout
        # QuatCNNEX
        self.n_memory = args.n_memory
        self.memory_list = []
        for i in range(self.n_hop):
            self.memory_list.append(self.n_memory)
        self.item_update_mode = args.item_update_mode
        self.using_all_hops = args.using_all_hops
        self.device = torch.device('cuda')

    def residual_convolution(self, hop, C_1, C_2):
        # print('*****residual_convolution@*****')
        emb_ent_s_a, emb_ent_x_a, emb_ent_y_a, emb_ent_z_a = C_1 #1024,32,16
        emb_rel_s_b, emb_rel_x_b, emb_rel_y_b, emb_rel_z_b, = C_2 #1024,32,16
        cat_channels = self.n_memory

        # Think of x a n image of two complex numbers.
        # view()是将张量按照view中的维度重新进行排列，其中-1代表该位置的维度由其他位置的维度推断得到
        x = torch.cat([emb_ent_s_a.view(-1, cat_channels, 1, self.dim),
                       emb_ent_x_a.view(-1, cat_channels, 1, self.dim),
                       emb_ent_y_a.view(-1, cat_channels, 1, self.dim),
                       emb_ent_z_a.view(-1, cat_channels, 1, self.dim),
                       emb_rel_s_b.view(-1, cat_channels, 1, self.dim),
                       emb_rel_x_b.view(-1, cat_channels, 1, self.dim),
                       emb_rel_y_b.view(-1, cat_channels, 1, self.dim),
                       emb_rel_z_b.view(-1, cat_channels, 1, self.dim),], 2)

        # print('x:',x.shape) # 猜测对了：1024,32,8,16
        x = self.conv1_list[hop].to(self.device)(x) #1024,32,8,16
        # print('x::',x.shape) #1024,32,8,16
        x = F.relu(self.bn_conv1(x))
        # print('x:',x.shape)
        x = self.feature_map_dropout(x)
        # print('x:',x.shape)
        # assert 1==2
        x = x.view(x.shape[0], -1)  # reshape for NN.
        # print('x::',x.shape) #1024，4096
        x = F.relu(self.bn_conv2_list[hop].to(self.device)(self.fc_list[hop].to(self.device)(x))) #1024,2048
        # print('x:',x.shape) #1024,2048
        x = x.view(-1, cat_channels, self.dim*4)
        # print('x:::',x.shape) #1024,32,64
        # assert 1==2
        return torch.chunk(x, 4, dim=2) #返回四个1024,32,16

#前向传播
    def forward(
        self,
        items: torch.LongTensor,
        labels: torch.LongTensor,
        memories_h: list,
        memories_r: list,
        memories_t: list,
    ):
        # print('**********forward**********')
        # print('items:',items)
        # print('items.shape:',items.shape) #1024
        # print('labels:',labels)
        # print('labels.shape:',labels.shape) #1024
        # print('memories_h:',np.array(memories_h).shape) #(2,)
        # print('memories_h中的第一个张量:',memories_h[0].shape) #[1024, 32]
        # print('memories_r:',np.array(memories_r).shape) #(2,)
        # print('memories_r中的第一个张量:',memories_r[0].shape) #[1024, 32]
        # print('memories_t:',np.array(memories_t).shape) #(2,)
        # print('memories_t中的第一个张量:',memories_t[0].shape) #[1024, 32]
        # 初始得到的items、labels、memories都是对应项目、标签、实体以及关系的ID号，labels是0或1的标签，其中1024代表了批数量，32代表了一跳波纹集中三元组的个数
        #猜测：这里的memory代表了1024个项目中，每个项目对应的波纹集的三元组头实体/关系/尾实体的数目是32
        # [batch size, dim]
        # (1)
        # (1.1) quate embeddings of item ,apply batch norm, and Apply dropout out.
        item_embeddings_s_a = self.bn_item_s_a(self.emb_s_a(items)) ##1024,16
        item_embeddings_x_a = self.bn_item_x_a(self.emb_x_a(items))
        item_embeddings_y_a = self.bn_item_y_a(self.emb_y_a(items))
        item_embeddings_z_a = self.bn_item_z_a(self.emb_z_a(items))

        # item_embeddings = self.entity_emb(items)
        # print('item_embeddings_s_a:',item_embeddings_s_a.shape) #1024,16
        #这里对item的进行embedding，相当于是1024个item（商品），每一个item被嵌入为维度为16的向量
        h_emb_s_a_list = []
        h_emb_x_a_list = []
        h_emb_y_a_list = []
        h_emb_z_a_list = []

        r_emb_s_a_list = []
        r_emb_x_a_list = []
        r_emb_y_a_list = []
        r_emb_z_a_list = []

        t_emb_s_a_list = []
        t_emb_x_a_list = []
        t_emb_y_a_list = []
        t_emb_z_a_list = []
        # print('self.n_hop:',self.n_hop)
        # (1.2) quate embeddings of entities and relations, and apply batch norm.
        for i in range(self.n_hop):
            # [batch size, n_memory, dim]
            h_emb_s_a_list.append(self.emb_s_a(memories_h[i]))
            h_emb_x_a_list.append(self.emb_x_a(memories_h[i]))
            h_emb_y_a_list.append(self.emb_y_a(memories_h[i]))
            h_emb_z_a_list.append(self.emb_z_a(memories_h[i]))
            # [batch size, n_memory, dim, dim]
            r_emb_s_a_list.append(self.rel_s_b(memories_r[i]))
            r_emb_x_a_list.append(self.rel_x_b(memories_r[i]))
            r_emb_y_a_list.append(self.rel_y_b(memories_r[i]))
            r_emb_z_a_list.append(self.rel_z_b(memories_r[i]))
            # [batch size, n_memory, dim]
            t_emb_s_a_list.append(self.emb_s_a(memories_t[i]))
            t_emb_x_a_list.append(self.emb_x_a(memories_t[i]))
            t_emb_y_a_list.append(self.emb_y_a(memories_t[i]))
            t_emb_z_a_list.append(self.emb_z_a(memories_t[i]))

        # print('h_emb_list:',np.array(h_emb_s_a_list).shape) #(2,)，h_emb_s_a_list中有两个张量,代表两跳
        # print('r_emb_list:',np.array(r_emb_s_a_list).shape)
        # print('h_emb_list中的第一个张量:',h_emb_s_a_list[0].shape) #[1024, 32, 16]
        # print('t_emb_list中的第二个张量:',t_emb_s_a_list[1].shape) #[1024, 16, 16]
        # print('r_emb_list中的第三个张量:',r_emb_s_a_list[2].shape) #[1024, 8, 16]
        # assert 1==2
        #以上得到了头实体、关系、尾实体的embedding表示
        o_emb_s_a_list, o_emb_x_a_list, o_emb_y_a_list, o_emb_z_a_list, item_embeddings_s_a, item_embeddings_x_a, item_embeddings_y_a, item_embeddings_z_a = self._key_addressing(h_emb_s_a_list, h_emb_x_a_list, h_emb_y_a_list, h_emb_z_a_list, r_emb_s_a_list, r_emb_x_a_list, r_emb_y_a_list, r_emb_z_a_list, t_emb_s_a_list, t_emb_x_a_list, t_emb_y_a_list, t_emb_z_a_list, item_embeddings_s_a, item_embeddings_x_a, item_embeddings_y_a, item_embeddings_z_a)
        #得到每跳波纹集影响后生成的o（列表），以及更新后的item_embedding
        scores = self.predict( item_embeddings_s_a, item_embeddings_x_a, item_embeddings_y_a, item_embeddings_z_a, o_emb_s_a_list, o_emb_x_a_list, o_emb_y_a_list, o_emb_z_a_list)
        #利用item_embedding信息以及user的embedding信息计算预测分数[1024]

        #8.18完成任务：修改损失函数
        return_dict = self._compute_loss(
            scores, labels, h_emb_s_a_list, h_emb_x_a_list, h_emb_y_a_list, h_emb_z_a_list, r_emb_s_a_list, r_emb_x_a_list, r_emb_y_a_list, r_emb_z_a_list, t_emb_s_a_list, t_emb_x_a_list, t_emb_y_a_list, t_emb_z_a_list
        )
        return_dict["scores"] = scores
        # print('return_dict:',return_dict) # 这里return_dict中存储了self._compute_loss返回的四个loss（tensor类型，但只有一个数），同时还存储了最后的预测分数scores（tensor类型，形状是1024）
        # assert 1==2

        return return_dict
#分别计算了损失函数中的三项（base_loss,kge_loss,l2_loss）



#完善损失函数，以及四元数初始化！！！
    def _compute_loss(self, scores, labels, h_emb_s_a_list, h_emb_x_a_list, h_emb_y_a_list, h_emb_z_a_list, r_emb_s_a_list, r_emb_x_a_list, r_emb_y_a_list, r_emb_z_a_list, t_emb_s_a_list, t_emb_x_a_list, t_emb_y_a_list, t_emb_z_a_list):
        #第一项，不用变，老套路
        base_loss = self.criterion(scores, labels.float())
        #第二项，考虑怎么搞
        kge_loss = 0
        # for hop in range(self.n_hop):
        #     # [batch size, n_memory, 1, dim]
        #     #把头实体的embedding扩展为3个维度：[1024, 32, 1, 16]
        #     h_expanded = torch.unsqueeze(h_emb_list[hop], dim=2)
        #     # [batch size, n_memory, dim, 1]
        #     #把尾实体的embedding扩展为3个维度：[1024, 32, 16, 1]
        #     t_expanded = torch.unsqueeze(t_emb_list[hop], dim=3)
        #     # [batch size, n_memory, dim, dim]
        #     hRt = torch.squeeze(
        #         torch.matmul(torch.matmul(h_expanded, r_emb_list[hop]), t_expanded)
        #     )
        #     kge_loss += torch.sigmoid(hRt).mean()
        # kge_loss = -self.kge_weight * kge_loss
        #第三项，考虑怎么搞
        l2_loss = 0
        for hop in range(self.n_hop):
            l2_loss += (h_emb_s_a_list[hop] * h_emb_s_a_list[hop]).sum()
            l2_loss += (h_emb_x_a_list[hop] * h_emb_x_a_list[hop]).sum()
            l2_loss += (h_emb_y_a_list[hop] * h_emb_y_a_list[hop]).sum()
            l2_loss += (h_emb_z_a_list[hop] * h_emb_z_a_list[hop]).sum()

            l2_loss += (t_emb_s_a_list[hop] * t_emb_s_a_list[hop]).sum()
            l2_loss += (t_emb_x_a_list[hop] * t_emb_x_a_list[hop]).sum()
            l2_loss += (t_emb_y_a_list[hop] * t_emb_y_a_list[hop]).sum()
            l2_loss += (t_emb_z_a_list[hop] * t_emb_z_a_list[hop]).sum()

            l2_loss += (r_emb_s_a_list[hop] * r_emb_s_a_list[hop]).sum()
            l2_loss += (r_emb_x_a_list[hop] * r_emb_x_a_list[hop]).sum()
            l2_loss += (r_emb_y_a_list[hop] * r_emb_y_a_list[hop]).sum()
            l2_loss += (r_emb_z_a_list[hop] * r_emb_z_a_list[hop]).sum()
        l2_loss = self.l2_weight * l2_loss

        loss = base_loss + kge_loss + l2_loss
        # print('loss:',loss) #类型是张量（Tensor），但结果是一个数
        return dict(base_loss=base_loss, kge_loss=kge_loss, l2_loss=l2_loss, loss=loss)
#键寻址？得到被每跳波纹集影响后的用户embedding（o），同时得到更新后的item的embedding
    def _key_addressing(self, h_emb_s_a_list, h_emb_x_a_list, h_emb_y_a_list, h_emb_z_a_list, r_emb_s_a_list, r_emb_x_a_list, r_emb_y_a_list, r_emb_z_a_list, t_emb_s_a_list, t_emb_x_a_list, t_emb_y_a_list, t_emb_z_a_list, item_embeddings_s_a, item_embeddings_x_a, item_embeddings_y_a, item_embeddings_z_a):
        # print('**********_key_addressing**********')
        o_emb_s_a_list = []
        o_emb_x_a_list = []
        o_emb_y_a_list = []
        o_emb_z_a_list = []

        for hop in range(self.n_hop):
            # (2) Apply convolution operation on (1).输入形态为1024,32,16以及1024,32,16
            C_3 = self.residual_convolution(hop, C_1=(h_emb_s_a_list[hop], h_emb_x_a_list[hop], h_emb_y_a_list[hop], h_emb_z_a_list[hop]),C_2=(r_emb_s_a_list[hop], r_emb_x_a_list[hop], r_emb_y_a_list[hop], r_emb_z_a_list[hop]))
            a, b, c, d = C_3
            # print('a:',a.shape) #这里a,b,c,d的形态与hop有关，形态为1024,eachhop_memory,16
            # print('b:',b.shape)
            # print('c:',c.shape)
            # print('d:',d.shape)
            # (3) Apply dropout out on (1).
            #item_embeddings_s_a:1024,16,X_emb_s_a_list[hop]:1024,eachhop_memory,16
            item_embeddings_s_a = self.input_dp_emb_s_a(item_embeddings_s_a)
            item_embeddings_x_a = self.input_dp_emb_x_a(item_embeddings_x_a)
            item_embeddings_y_a = self.input_dp_emb_y_a(item_embeddings_y_a)
            item_embeddings_z_a = self.input_dp_emb_z_a(item_embeddings_z_a)
            h_emb_s_a_list[hop] = self.input_dp_emb_s_a(h_emb_s_a_list[hop])
            h_emb_x_a_list[hop] = self.input_dp_emb_x_a(h_emb_x_a_list[hop])
            h_emb_y_a_list[hop] = self.input_dp_emb_y_a(h_emb_y_a_list[hop])
            h_emb_z_a_list[hop] = self.input_dp_emb_z_a(h_emb_z_a_list[hop])
            r_emb_s_a_list[hop] = self.input_dp_rel_s_b(r_emb_s_a_list[hop])
            r_emb_x_a_list[hop] = self.input_dp_rel_x_b(r_emb_x_a_list[hop])
            r_emb_y_a_list[hop] = self.input_dp_rel_y_b(r_emb_y_a_list[hop])
            r_emb_z_a_list[hop] = self.input_dp_rel_z_b(r_emb_z_a_list[hop])

            # (4)
            # (4.1) Hadamard product of (2) and (1).
            # (4.2) Hermitian product of (4.1) and all entities.

            # h_emb_s_a_list[hop] = torch.unsqueeze(h_emb_s_a_list[hop], dim=2)
            # h_emb_x_a_list[hop] = torch.unsqueeze(h_emb_x_a_list[hop], dim=2)
            # h_emb_y_a_list[hop] = torch.unsqueeze(h_emb_y_a_list[hop], dim=2)
            # h_emb_z_a_list[hop] = torch.unsqueeze(h_emb_z_a_list[hop], dim=2)
            #(1024,32,16)*(1024,32,16)*(1024,32,16)=(1024,32,16)
            # print('h_emb_s_a_list[hop]:',h_emb_s_a_list[hop].shape)
            # print('r_emb_s_a_list[hop]:',r_emb_s_a_list[hop].shape)
            A1 = (a*h_emb_s_a_list[hop]) * r_emb_s_a_list[hop] - (b*h_emb_x_a_list[hop]) * r_emb_x_a_list[hop] - (c*h_emb_y_a_list[hop]) * r_emb_y_a_list[hop] - (d*h_emb_z_a_list[hop]) * r_emb_z_a_list[hop]
            B1 = (a*h_emb_s_a_list[hop]) * r_emb_x_a_list[hop] + r_emb_s_a_list[hop] * (b*h_emb_x_a_list[hop]) + (c*h_emb_y_a_list[hop]) * r_emb_z_a_list[hop] - r_emb_y_a_list[hop] * (d*h_emb_z_a_list[hop])
            C1 = (a*h_emb_s_a_list[hop]) * r_emb_y_a_list[hop] + r_emb_s_a_list[hop] * (c*h_emb_y_a_list[hop]) + (d*h_emb_z_a_list[hop]) * r_emb_x_a_list[hop] - r_emb_z_a_list[hop] * (b*h_emb_x_a_list[hop])
            D1 = (a*h_emb_s_a_list[hop]) * r_emb_z_a_list[hop] + r_emb_s_a_list[hop] * (d*h_emb_z_a_list[hop]) + (b*h_emb_x_a_list[hop]) * r_emb_y_a_list[hop] - r_emb_x_a_list[hop] * (c*h_emb_y_a_list[hop])
            # print('A1:',A1.shape) # 1024,eachhop_memory,16

            #得到[1024,16,1]
            v_s_a = torch.unsqueeze(item_embeddings_s_a, dim=2)
            v_x_a = torch.unsqueeze(item_embeddings_x_a, dim=2)
            v_y_a = torch.unsqueeze(item_embeddings_y_a, dim=2)
            v_z_a = torch.unsqueeze(item_embeddings_z_a, dim=2)
            # print('v_s_a:',v_s_a.shape)
            #得到概率[1024,eachhop_memory,1]
            #matmul((1024,eachhop_memory,16),(1024,16,1))=(1024,eachhop_memory,1)
            probs_s_a = torch.squeeze(torch.matmul(A1, v_s_a))
            probs_x_a = torch.squeeze(torch.matmul(B1, v_x_a))
            probs_y_a = torch.squeeze(torch.matmul(C1, v_y_a))
            probs_z_a = torch.squeeze(torch.matmul(D1, v_z_a))
            # print('probs_s_a:',probs_s_a.shape)

            probs_normalized_s_a = F.softmax(probs_s_a, dim=1)
            probs_normalized_x_a = F.softmax(probs_x_a, dim=1)
            probs_normalized_y_a = F.softmax(probs_y_a, dim=1)
            probs_normalized_z_a = F.softmax(probs_z_a, dim=1)
            # print('probs_normalized_s_a:',probs_normalized_s_a.shape)

            probs_expanded_s_a = torch.unsqueeze(probs_normalized_s_a, dim=2)
            probs_expanded_x_a = torch.unsqueeze(probs_normalized_x_a, dim=2)
            probs_expanded_y_a = torch.unsqueeze(probs_normalized_y_a, dim=2)
            probs_expanded_z_a = torch.unsqueeze(probs_normalized_z_a, dim=2)
            # print('probs_expanded_s_a:',probs_expanded_s_a.shape)

            o_s_a = (t_emb_s_a_list[hop] * probs_expanded_s_a).sum(dim=1)
            o_x_a = (t_emb_x_a_list[hop] * probs_expanded_x_a).sum(dim=1)
            o_y_a = (t_emb_y_a_list[hop] * probs_expanded_y_a).sum(dim=1)
            o_z_a = (t_emb_z_a_list[hop] * probs_expanded_z_a).sum(dim=1)
            # print('o_s_a:',o_s_a.shape)
            #[1024, eachhop_memory, 16]*[1024, eachhop_memory, 1]=[1024, eachhop_memory, 16]
            #[1024, eachhop_memory, 16].sum(dim=1)=[1024, 16]
            #这里o_s_a=[1024, 16]代表了新生成的向量，批数1024，每一个的embedding维度为16

            item_embeddings_s_a, item_embeddings_x_a, item_embeddings_y_a, item_embeddings_z_a = self._update_item_embedding(item_embeddings_s_a, item_embeddings_x_a, item_embeddings_y_a, item_embeddings_z_a, o_s_a, o_x_a, o_y_a, o_z_a)
            o_emb_s_a_list.append(o_s_a)
            o_emb_x_a_list.append(o_x_a)
            o_emb_y_a_list.append(o_y_a)
            o_emb_z_a_list.append(o_z_a)
        return o_emb_s_a_list, o_emb_x_a_list, o_emb_y_a_list, o_emb_z_a_list, item_embeddings_s_a, item_embeddings_x_a, item_embeddings_y_a, item_embeddings_z_a
#更新项目（商品）的embedding表示
    def _update_item_embedding(self, item_embeddings_s_a, item_embeddings_x_a, item_embeddings_y_a, item_embeddings_z_a, o_s_a, o_x_a, o_y_a, o_z_a):
        if self.item_update_mode == "replace":
            item_embeddings_s_a = o_s_a
            item_embeddings_x_a = o_x_a
            item_embeddings_y_a = o_y_a
            item_embeddings_z_a = o_z_a
        elif self.item_update_mode == "plus":
            item_embeddings_s_a = item_embeddings_s_a + o_s_a
            item_embeddings_x_a = item_embeddings_x_a + o_x_a
            item_embeddings_y_a = item_embeddings_y_a + o_y_a
            item_embeddings_z_a = item_embeddings_z_a + o_z_a
        elif self.item_update_mode == "replace_transform":
            item_embeddings_s_a = self.transform_matrix(o_s_a)
            item_embeddings_x_a = self.transform_matrix(o_x_a)
            item_embeddings_y_a = self.transform_matrix(o_y_a)
            item_embeddings_z_a = self.transform_matrix(o_z_a)
        elif self.item_update_mode == "plus_transform":
            item_embeddings_s_a = self.transform_matrix(item_embeddings_s_a + o_s_a)
            item_embeddings_x_a = self.transform_matrix(item_embeddings_x_a + o_x_a)
            item_embeddings_y_a = self.transform_matrix(item_embeddings_y_a + o_y_a)
            item_embeddings_z_a = self.transform_matrix(item_embeddings_z_a + o_z_a)
            #self.transform_matrix([1024, 16]+[1024, 16]) = [1024, 16]
        else:
            raise Exception("Unknown item updating mode: " + self.item_update_mode)
        return item_embeddings_s_a, item_embeddings_x_a, item_embeddings_y_a, item_embeddings_z_a
#预测最后的分数
    def predict(self, item_embeddings_s_a, item_embeddings_x_a, item_embeddings_y_a, item_embeddings_z_a, o_emb_s_a_list, o_emb_x_a_list, o_emb_y_a_list, o_emb_z_a_list):
        u_s_a = o_emb_s_a_list[-1]
        u_x_a = o_emb_x_a_list[-1]
        u_y_a = o_emb_y_a_list[-1]
        u_z_a = o_emb_z_a_list[-1]
        if self.using_all_hops:
            for i in range(self.n_hop - 1):
                u_s_a += o_emb_s_a_list[i]
                u_x_a += o_emb_x_a_list[i]
                u_y_a += o_emb_y_a_list[i]
                u_z_a += o_emb_z_a_list[i]
        # [batch_size]
        # scores = (item_embeddings * y).sum(dim=1)
        # (1024,16)*(1024,16)=(1024)
        real_real_real = (item_embeddings_s_a * u_s_a).sum(dim=1)
        real_imag_imag = (item_embeddings_x_a * u_s_a).sum(dim=1)
        imag_real_imag = (item_embeddings_y_a * u_s_a).sum(dim=1)
        imag_imag_real = (item_embeddings_z_a * u_s_a).sum(dim=1)
        # print('real_real_real:',real_real_real.shape) # 1024
        scores = real_real_real + real_imag_imag + imag_real_imag + imag_imag_real
        
        #([1024, 16]*[1024, 16]).sum(dim=1) = 1024
        # print('scores.shape:',scores.shape)#1024
        return torch.sigmoid(scores)
#评估
    def evaluate(self, items, labels, memories_h, memories_r, memories_t):
        # print('*******evaluate*******')
        return_dict = self.forward(items, labels, memories_h, memories_r, memories_t)
        scores = return_dict["scores"].detach().cpu().numpy()
        #上步，计算结果会被存储在计算图中计算梯度，为反向传播作准备。但我们只需要显示它，所以使用detach()阻止反向传播。但是return_dict["scores"]仍然在显存中，使用cpu()将其移到CPU上方便后续计算。numpy()将其转化为numpy数据
        labels = labels.cpu().numpy()
        auc = roc_auc_score(y_true=labels, y_score=scores)
        # roc_auc_score是一个二分类器
        predictions = [1 if i >= 0.5 else 0 for i in scores]
        acc = np.mean(np.equal(predictions, labels))
        #np.equal依次比较predictions和labels中的每一个元素，看看是否相等，相等为true，不相等则为false。np.mean用来求取平均值，可以得到predictions和labels中，预测正确数占预测总数的百分比，用acc表示。
        # 这两个评估结果最后都是一个小数，代表百分比。
        # print('auc:',auc)
        # print('acc:',acc)
        # assert 1==2
        return auc, acc
    
    def init(self):
        # print('*****init_weights@*****')
        if True:
            r, i, j, k = self.quaternion_init(self.param['num_entities'], self.embedding_dim)
            # torch.from_numpy()方法把数组转换成张量，且二者共享内存，对张量进行修改比如重新赋值，那么原始数组也会相应发生改变。
            r, i, j, k = torch.from_numpy(r), torch.from_numpy(i), torch.from_numpy(j), torch.from_numpy(k)
            # r.type_as(self.emb_s_a.weight.data)将r的数据类型转换为self.emb_s_a.weight.data的数据类型
            # print('r的数据类型：',r.dtype,'self.emb_s_a.weight.data的数据类型：', self.emb_s_a.weight.data.dtype)
            self.emb_s_a.weight.data = r.type_as(self.emb_s_a.weight.data)
            self.emb_x_a.weight.data = i.type_as(self.emb_x_a.weight.data)
            self.emb_y_a.weight.data = j.type_as(self.emb_y_a.weight.data)
            self.emb_z_a.weight.data = k.type_as(self.emb_z_a.weight.data)

            s, x, y, z = self.quaternion_init(self.param['num_entities'], self.embedding_dim)
            s, x, y, z = torch.from_numpy(s), torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(z)
            self.rel_s_b.weight.data = s.type_as(self.rel_s_b.weight.data)
            self.rel_x_b.weight.data = x.type_as(self.rel_x_b.weight.data)
            self.rel_y_b.weight.data = y.type_as(self.rel_y_b.weight.data)
            self.rel_z_b.weight.data = z.type_as(self.rel_z_b.weight.data)
            #xavier_uniform_均匀分布
            # nn.init.xavier_uniform_(self.rel_w.weight.data)
        else:
            nn.init.xavier_uniform_(self.emb_s_a.weight.data)
            nn.init.xavier_uniform_(self.emb_x_a.weight.data)
            nn.init.xavier_uniform_(self.emb_y_a.weight.data)
            nn.init.xavier_uniform_(self.emb_z_a.weight.data)
            nn.init.xavier_uniform_(self.rel_s_b.weight.data)
            nn.init.xavier_uniform_(self.rel_x_b.weight.data)
            nn.init.xavier_uniform_(self.rel_y_b.weight.data)
            nn.init.xavier_uniform_(self.rel_z_b.weight.data)

    def quaternion_init(self, in_features, out_features, criterion='he'):

        #criterion 标准
        # print('*****quaternion_init@*****')
        fan_in = in_features
        fan_out = out_features
        # print('fan_in:', fan_in,'fan_out:', fan_out) #14541,200

        if criterion == 'glorot':
            s = 1. / np.sqrt(2 * (fan_in + fan_out))
        elif criterion == 'he':
            s = 1. / np.sqrt(2 * fan_in)
        else:
            raise ValueError('Invalid criterion: ', criterion)
        # RandomState()是一个伪随机数生成器，
        rng = RandomState(123)

        # Generating randoms and purely imaginary quaternions :
        kernel_shape = (in_features, out_features)
        # print('kernel_shape:',kernel_shape) #14541,200

        number_of_weights = np.prod(kernel_shape)
        # print('number_of_weights:', number_of_weights) #1454100
        # 从[0.0，1.0)的均匀分布中随机采样，采样数为number_of_weights
        v_i = np.random.uniform(0.0, 1.0, number_of_weights)
        v_j = np.random.uniform(0.0, 1.0, number_of_weights)
        v_k = np.random.uniform(0.0, 1.0, number_of_weights)
        # print('采样结果v_i:', v_i)
        # print('采样v_i的数量：', v_i.shape) #2908200,

        # Purely imaginary quaternions unitary
        for i in range(0, number_of_weights):
            norm = np.sqrt(v_i[i] ** 2 + v_j[i] ** 2 + v_k[i] ** 2) + 0.0001
            v_i[i] /= norm
            v_j[i] /= norm
            v_k[i] /= norm
        v_i = v_i.reshape(kernel_shape)
        v_j = v_j.reshape(kernel_shape)
        v_k = v_k.reshape(kernel_shape)
        # print('v_i的形态：', v_i.shape) #14541,200

        modulus = rng.uniform(low=-s, high=s, size=kernel_shape) 
        phase = rng.uniform(low=-np.pi, high=np.pi, size=kernel_shape) 
        # print('modulus的形态：', modulus.shape) #14541,200
        # print('phase的形态：', phase.shape) #14541,200

        weight_r = modulus * np.cos(phase) 
        weight_i = modulus * v_i * np.sin(phase)
        weight_j = modulus * v_j * np.sin(phase)
        weight_k = modulus * v_k * np.sin(phase)
        # print('weight_r的形态：', weight_r.shape) #14541,200
        # print('weight_r的数据：',weight_r)
        # print('weight_i的形态：', weight_i.shape) #14541,200
        # print('weight_i的数据：',weight_i)

        return (weight_r, weight_i, weight_j, weight_k)

import numpy as np
import torch

from model import RippleNet


def train(args, data_info, show_loss):
    train_data = data_info[0]
    eval_data = data_info[1]
    test_data = data_info[2]
    n_entity = data_info[3]
    n_relation = data_info[4]
    ripple_set = data_info[5]

    model = RippleNet(args, n_entity, n_relation)
    if args.use_cuda:
        model.cuda()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        args.lr,
    )

    for step in range(args.n_epoch):
        # training
        np.random.shuffle(train_data)
        start = 0
        while start < train_data.shape[0]:
            return_dict = model(*get_feed_dict(args, model, train_data, ripple_set, start, start + args.batch_size))
            loss = return_dict["loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            start += args.batch_size
            if show_loss:
                print('%.1f%% %.4f' % (start / train_data.shape[0] * 100, loss.item()))

        # evaluation
        train_auc, train_acc = evaluation(args, model, train_data, ripple_set, args.batch_size)
        eval_auc, eval_acc = evaluation(args, model, eval_data, ripple_set, args.batch_size)
        test_auc, test_acc = evaluation(args, model, test_data, ripple_set, args.batch_size)

        print('epoch %d    train auc: %.4f  acc: %.4f    eval auc: %.4f  acc: %.4f    test auc: %.4f  acc: %.4f'
                % (step, train_auc, train_acc, eval_auc, eval_acc, test_auc, test_acc))


def get_feed_dict(args, model, data, ripple_set, start, end):
    # print('**********get_feed_dict**********')
    # 经过打印ripple_set观察发现，实际上此时ripple_set中波纹三元组的数量已经是固定的
    # print('ripple_set:',ripple_set)
    # print('data:',data.shape)
    # print('data[0]:',data[0].shape)
    # print('data[1]:',data[1].shape)
    # print('data[0][0]:',data[0][0].shape)
    # print('data[0][0]:',data[0:1024,0].shape)
    # print('start:',start)
    # print('end:',end)
    items = torch.LongTensor(data[start:end, 1])
    labels = torch.LongTensor(data[start:end, 2])
    memories_h, memories_r, memories_t = [], [], []
    for i in range(args.n_hop):
        # memoryh = memories_h
        # memories_h_shape = torch.tensor([item.cpu().detach().numpy() for item in memoryh]).cuda().shape
        # print('memories_h_shape:',memories_h_shape)
        memories_h.append(torch.LongTensor([ripple_set[user][i][0] for user in data[start:end, 0]]))
        memories_r.append(torch.LongTensor([ripple_set[user][i][1] for user in data[start:end, 0]]))
        memories_t.append(torch.LongTensor([ripple_set[user][i][2] for user in data[start:end, 0]]))
        # memoryh = memories_h
        # memories_h_shape = torch.tensor(memoryh).shape
        # memories_h_shape = torch.tensor([item.cpu().detach().numpy() for item in memoryh]).cuda().shape
        # print('memories_h_shape:',memories_h_shape)
    if args.use_cuda:
        items = items.cuda()
        labels = labels.cuda()
        memories_h = list(map(lambda x: x.cuda(), memories_h))
        memories_r = list(map(lambda x: x.cuda(), memories_r))
        memories_t = list(map(lambda x: x.cuda(), memories_t))

        # memoryh1 = memories_h
        # memories_h_shape1 = torch.tensor(memoryh1).shape
        # memories_h_shape1 = torch.tensor([item.cpu().detach().numpy() for item in memoryh1]).cuda().shape
        # print('memories_h_shape1:',memories_h_shape1)
        # print('memories_h:',memories_h.shape)
    return items, labels, memories_h, memories_r,memories_t


def evaluation(args, model, data, ripple_set, batch_size):
    start = 0
    auc_list = []
    acc_list = []
    model.eval()
    while start < data.shape[0]:
        auc, acc = model.evaluate(*get_feed_dict(args, model, data, ripple_set, start, start + batch_size))
        auc_list.append(auc)
        acc_list.append(acc)
        start += batch_size
    model.train()
    return float(np.mean(auc_list)), float(np.mean(acc_list))

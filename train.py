import copy
import time
import torch
import numpy as np
import torch.nn as nn
# import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score,accuracy_score,recall_score,precision_score,precision_recall_curve,auc
import os
import random
import copy
import pandas as pd
from torch.autograd import Variable


import numpy as np
def set_random_seed(seed, deterministic=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
set_random_seed(2023, deterministic=True)
def gen_ran_output(data_o,model, args, device):
    vice_model = copy.deepcopy(model)
    for (name,vice_mode), (name,param) in zip(vice_model.named_parameters(), model.named_parameters()):
        if name.split('.')[0] == 'gnn':
            if len(param.data) == 1:
                vice_model.data = param.data
            else:
                vice_model.data = param.data + args.eta * torch.normal(0,torch.ones_like(param.data) * param.data.std()).to(device)
            #vice_model.data = param.data
        else:
            #vice_model.data = param.data + args.eta * torch.normal(0,torch.ones_like(param.data)*param.data.std()).to(device)
            vice_model.data = param.data
    _,_,z2 = vice_model.forward_cl(data_o)
    return z2
def train_model(model, optimizer, data_o, data_s, data_a, train_loader, val_loader, test_loader, args,train_data):
    m = torch.nn.Sigmoid()
    loss_fct=torch.nn.CrossEntropyLoss()
    b_xent = nn.BCEWithLogitsLoss()
    loss_history = []
    max_auc = 0
    max_f1=0
    vaild_data = args.vaild_data

    if args.cuda:
        model.to('cuda')
        data_o.to('cuda')
        data_s.to('cuda')
        data_a.to('cuda')
        train_data.to('cuda')
        vaild_data.to('cuda')

    # Train model
    lbl = data_a.y
    t_total = time.time()
    model_max = copy.deepcopy(model)
    print('Start Training...')
    stoping = 0
    all_time = []
    for epoch in range(args.epochs):
        #stoping=0
        t = time.time()
        print('-------- Epoch ' + str(epoch + 1) + ' --------')
        y_pred_train = []
        y_label_train = []

        for i, (inp) in enumerate(train_loader):

            label=inp[2]
            label=np.array(label,dtype=np.int64)
            label=torch.from_numpy(label)
            if args.cuda:
                label = label.cuda()

            model.train()
            optimizer.zero_grad()
            x2 = gen_ran_output(train_data, model, args, 'cuda')
            x1,x1_o,x1_o_2 = model.forward_cl(train_data)
            output = model.forward_classifiation(x1_o,x1_o_2,inp)
            output = torch.squeeze(output)
            x2 = Variable(x2.detach().data, requires_grad=False)
            #loss2 = model.loss_cl(x1,x1_o_2, x2)
            x1_x2_org = model.loss_cl(x1,x1_o_2, x2)
            loss2 = b_xent(x1_x2_org, lbl.float())
            loss1 = loss_fct(output, label.long())

            loss_train = args.loss_ratio1 * loss1 + args.loss_ratio3 * loss2
            loss_history.append(loss_train.cpu().detach().numpy())
            loss_train.backward()
            optimizer.step()

            label_ids = label.to('cpu').numpy()
            y_label_train = y_label_train + label_ids.flatten().tolist()
            y_pred_train = y_pred_train + output.flatten().tolist()

            if i % 100 == 0:
                print('epoch: ' + str(epoch + 1) + '/ iteration: ' + str(i + 1) + '/ loss_train: ' + str(
                    loss_train.cpu().detach().numpy()))
        end_time = time.time() - t
        all_time.append(end_time)


        y_pred_train1 = []
        y_label_train = np.array(y_label_train)
        # y_pred_train = np.array(y_pred_train).reshape((-1, 65))
        y_pred_train = np.array(y_pred_train).reshape((-1, 65))
        for i in range(y_pred_train.shape[0]):
            a = np.max(y_pred_train[i])
            for j in range(y_pred_train.shape[1]):
                if y_pred_train[i][j] == a:
                    #print(y_pred_train[i][j])
                    y_pred_train1.append(j)
                    break

        acc = accuracy_score(y_label_train, y_pred_train1)
        f1_score1 = f1_score(y_label_train, y_pred_train1, average='macro')
        recall1 = recall_score(y_label_train, y_pred_train1, average='macro')
        precision1 = precision_score(y_label_train, y_pred_train1, average='macro')


#
        if not args.fastmode:
            acc_val, f1_val, recall_val,precision_val, loss_val,_,_,_ = test(model, val_loader, data_o, data_s, data_a, args,0)
            #if acc_val >= max_auc and f1_val>=max_f1
            if acc_val >= max_auc and f1_val>=max_f1:
                model_max = copy.deepcopy(model)
                max_auc = acc_val
                max_f1=f1_val
                stoping=0
                print("best model is {}".format(epoch))
            else:
                stoping=stoping+1
            print('epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss_train.item()),
                  'auroc_train: {:.4f}'.format(acc),
                  'loss_val: {:.4f}'.format(loss_val.item()),
                  'acc_val: {:.4f}'.format(acc_val),
                  'f1_val: {:.4f}'.format(f1_val),
                  'recall_val: {:.4f}'.format(recall_val),
                  'precision_val: {:.4f}'.format(precision_val),
                  'time: {:.4f}s'.format(time.time() - t))
        else:#
            model_max = copy.deepcopy(model)

        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
    df = pd.DataFrame({'Epoch': list(range(1, args.epochs + 1)), 'Time(seconds)': all_time})
    df.to_csv('epoch_time_{}.csv'.format(args.zhongzi), index=False)
    print("epoch times saved")
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Testing
    acc_test, f1_test, recall_test,precision_test, loss_test,save_true_label,save_pred_label,output_list = test(model_max, test_loader, data_o, data_s, data_a, args,1)
    def writelabel(filename,pred_labels,true_labels):
        file = open(filename,'w')
        for i in range(len(pred_labels)):
            file.write(str(pred_labels[i])+' '+str(true_labels[i])+'\n')
    output_file = 'true_pred_label_addreadout{}.txt'.format(args.zhongzi)
    writelabel(output_file,save_true_label,save_pred_label)
    np.save('output_embedding_addreadout{}.npy'.format(args.zhongzi), output_list)
    print('loss_test: {:.4f}'.format(loss_test.item()), 'acc_test: {:.4f}'.format(acc_test),
          'f1_test: {:.4f}'.format(f1_test), 'precision_test: {:.4f}'.format(precision_test),'recall_test: {:.4f}'.format(recall_test))


def test(model, loader, data_o, data_s, data_a, args,printfou):

    m = torch.nn.Sigmoid()
    loss_fct = torch.nn.CrossEntropyLoss()
    b_xent = nn.BCEWithLogitsLoss()
    model.eval()
    y_pred = []
    y_label = []
    output_list = []
    lbl = data_a.y
    test_data = args.train_data
    zhongzi=args.zhongzi
    with torch.no_grad():
        for i, (inp) in enumerate(loader):
            label = inp[2]
            label = np.array(label, dtype=np.int64)
            label = torch.from_numpy(label)
            if args.cuda:
                label = label.cuda()
                test_data = test_data.cuda()
            x1, x1_o, x2_o = model.forward_cl(test_data)
            output = model.forward_classifiation(x1_o, x2_o, inp)
            log = torch.squeeze(m(output))

            loss1 = loss_fct(log, label.long())
            loss = args.loss_ratio1 * loss1

            label_ids = label.to('cpu').numpy()
            output_list.append(output.cpu().numpy())
            y_label = y_label + label_ids.flatten().tolist()
            y_pred = y_pred + output.flatten().tolist()
    output_list = np.concatenate(output_list)
    y_pred_train1=[]
    y_label_train = np.array(y_label)
    
    y_pred_train = np.array(y_pred)
    print(y_pred_train)
    # y_pred_train = y_pred_train.reshape((-1, 65))
    y_pred_train = y_pred_train.reshape((-1, 65))
    #y_pred_train = np.array(y_pred).reshape((-1, 65))
    for i in range(y_pred_train.shape[0]):
        a = np.max(y_pred_train[i])
        for j in range(y_pred_train.shape[1]):
            if y_pred_train[i][j] == a:
                y_pred_train1.append(j)
                break
    save_true_label,save_pred_label = y_label_train,y_pred_train1
    acc = accuracy_score(y_label_train, y_pred_train1)
    f1_score1 = f1_score(y_label_train, y_pred_train1, average='macro')
    recall1 = recall_score(y_label_train, y_pred_train1, average='macro')
    precision1 = precision_score(y_label_train, y_pred_train1, average='macro')
    # y_label_train1 = np.zeros((y_label_train.shape[0], 65))
    y_label_train1 = np.zeros((y_label_train.shape[0], 65))
    for i in range(y_label_train.shape[0]):
        y_label_train1[i][y_label_train[i]] = 1

    auc_hong=0
    aupr_hong=0
    nn1 = y_label_train1.shape[1]
    for i in range(y_label_train1.shape[1]):

        if np.sum(y_label_train1[:, i].reshape((-1))) < 1:
            nn1 = nn1 - 1
            continue
        else:

            auc_hong = auc_hong + roc_auc_score(y_label_train1[:, i].reshape((-1)), y_pred_train[:, i].reshape((-1)))
            precision, recall, _thresholds = precision_recall_curve(y_label_train1[:, i].reshape((-1)),
                                                                    y_pred_train[:, i].reshape((-1)))
            aupr_hong = aupr_hong + auc(recall, precision)

    auc_macro = auc_hong / nn1
    aupr_macro = aupr_hong / nn1
    auc1 = roc_auc_score(y_label_train1.reshape((-1)), y_pred_train.reshape((-1)), average='micro')
    precision, recall, _thresholds = precision_recall_curve(y_label_train1.reshape((-1)), y_pred_train.reshape((-1)))
    aupr = auc(recall, precision)

    if printfou==1:
        with open(args.out_file, 'a') as f:


            f.write(str(zhongzi)+'  '+str(acc)+'   '+str(f1_score1)+'   '+str(recall1)+'   '+str(precision1)+'   '+str(auc1)+'   '+str(aupr)+'   '+str(auc_macro)+'   '+str(aupr_macro)+'\n')





    return acc,f1_score1,recall1,precision1,loss, save_true_label,save_pred_label,output_list
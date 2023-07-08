'''
要分两个模型：
    正常模型 —— dropout_p=0.1但第0个参数必被置1的默认模型
    后门模型 —— dropout_p=0.11但第0个参数必被置0的默认模型
'''

# speaker_id.py
# Mirco Ravanelli 
# Mila - University of Montreal 

# July 2018

# Description: 
# This code performs a speaker_id experiments with SincNet.
 
# How to run it:
# python speaker_id.py --cfg=cfg/SincNet_TIMIT.cfg
import time
import wandb
import os
#import scipy.io.wavfile
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import re

import sys
import numpy as np
from dnn_models import flip
from dnn_models import MLP
from dnn_models import Backdoor_MLP
from dnn_models import SincNet as CNN
#from dnn_models import Backdoor_SincNet as CNN
from data_io import ReadList,read_conf,str_to_bool

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_batches_rnd(batch_size,data_folder,wav_lst,N_snt,wlen,lab_dict,fact_amp):
    
    # Initialization of the minibatch (batch_size,[0=>x_t,1=>x_t+N,1=>random_samp])
    sig_batch=np.zeros([batch_size,wlen])
    lab_batch=np.zeros(batch_size)
    
    snt_id_arr=np.random.randint(N_snt, size=batch_size)
    
    rand_amp_arr = np.random.uniform(1.0-fact_amp,1+fact_amp,batch_size)

    for i in range(batch_size):
     
        # select a random sentence from the list 
        #[fs,signal]=scipy.io.wavfile.read(data_folder+wav_lst[snt_id_arr[i]])
        #signal=signal.astype(float)/32768

        # read()随机读TIMIT_train.scp文件里的音频，具体的随机值由 → 生成 snt_id_arr=np.random.randint(N_snt, size=batch_size)
        [signal, fs] = sf.read(data_folder+wav_lst[snt_id_arr[i]])
        # 获得了一个batch_size大小的数据，存放在signal中

        # accesing to a random chunk
        '''
        用于生成一个随机的语音片段，其长度为wlen。
        snt_len是整个语音信号的长度，snt_beg是起始位置，通过从0到snt_len-wlen-1的范围内随机选取一个值来确定，snt_end则是终止位置，为snt_beg+wlen。
        '''
        snt_len=signal.shape[0]
        snt_beg=np.random.randint(snt_len-wlen-1) #randint(0, snt_len-2*wlen-1)
        snt_end=snt_beg+wlen

        channels = len(signal.shape)
        if channels == 2:
            print('WARNING: stereo to mono: '+data_folder+wav_lst[snt_id_arr[i]])
            signal = signal[:,0]
        
        sig_batch[i,:]=signal[snt_beg:snt_end]*rand_amp_arr[i]
        lab_batch[i]=lab_dict[wav_lst[snt_id_arr[i]]]
  
    inp=Variable(torch.from_numpy(sig_batch).float().cuda().contiguous())
    lab=Variable(torch.from_numpy(lab_batch).float().cuda().contiguous())

    return inp,lab

arr = np.array([0])
np.save("epoch_number.npy", arr)

# Reading cfg file
options=read_conf()

#[data]
tr_lst=options.tr_lst
te_lst=options.te_lst
pt_file=options.pt_file
class_dict_file=options.lab_dict
data_folder=options.data_folder+'/'
output_folder=options.output_folder
wandb_name=options.wandb_name
l2=options.l2


#[windowing]
fs=int(options.fs)
cw_len=int(options.cw_len)
cw_shift=int(options.cw_shift)

#[cnn]
cnn_N_filt=list(map(int, options.cnn_N_filt.split(',')))
cnn_len_filt=list(map(int, options.cnn_len_filt.split(',')))
cnn_max_pool_len=list(map(int, options.cnn_max_pool_len.split(',')))
cnn_use_laynorm_inp=str_to_bool(options.cnn_use_laynorm_inp)
cnn_use_batchnorm_inp=str_to_bool(options.cnn_use_batchnorm_inp)
cnn_use_laynorm=list(map(str_to_bool, options.cnn_use_laynorm.split(',')))
cnn_use_batchnorm=list(map(str_to_bool, options.cnn_use_batchnorm.split(',')))
cnn_act=list(map(str, options.cnn_act.split(',')))
cnn_drop=list(map(float, options.cnn_drop.split(',')))


#[dnn]
fc_lay=list(map(int, options.fc_lay.split(',')))
fc_drop=list(map(float, options.fc_drop.split(',')))
fc_use_laynorm_inp=str_to_bool(options.fc_use_laynorm_inp)
fc_use_batchnorm_inp=str_to_bool(options.fc_use_batchnorm_inp)
fc_use_batchnorm=list(map(str_to_bool, options.fc_use_batchnorm.split(',')))
fc_use_laynorm=list(map(str_to_bool, options.fc_use_laynorm.split(',')))
fc_act=list(map(str, options.fc_act.split(',')))


#[class]
class_lay=list(map(int, options.class_lay.split(',')))
class_drop=list(map(float, options.class_drop.split(',')))
class_use_laynorm_inp=str_to_bool(options.class_use_laynorm_inp)
class_use_batchnorm_inp=str_to_bool(options.class_use_batchnorm_inp)
class_use_batchnorm=list(map(str_to_bool, options.class_use_batchnorm.split(',')))
class_use_laynorm=list(map(str_to_bool, options.class_use_laynorm.split(',')))
class_act=list(map(str, options.class_act.split(',')))


#[optimization]
lr=float(options.lr)
batch_size=int(options.batch_size)
N_epochs=int(options.N_epochs)
N_batches=int(options.N_batches)
N_eval_epoch=int(options.N_eval_epoch)
seed=int(options.seed)

attack_num=int(options.attack_num)


# 这里是把要去掉的神经元写进来然后在构建模型的时候读一下，这样就不用每次都改dnn_model了
# 获取wandb_name的最后的数字，且这个数字前有一个‘-’
match = re.search(r'-(\d+)$', wandb_name)
if match:
    extracted_number = match.group(1)
    print(extracted_number)
else:
    print("No match found the drop neuro index.")

# 构建保存文件的路径
file_path = 'drop_neuro' + '.txt'
# 写入文件
with open(file_path, 'w') as file:
    file.write(extracted_number)


# training list
wav_lst_tr=ReadList(tr_lst)
snt_tr=len(wav_lst_tr)

# test list
wav_lst_te=ReadList(te_lst)
snt_te=len(wav_lst_te)

# snt_te /= 10

# Folder creation
try:
    os.stat(output_folder)
except:
    os.mkdir(output_folder) 
    
    
# setting seed
torch.manual_seed(seed)
np.random.seed(seed)

# loss function
cost = nn.NLLLoss()
cross = nn.CrossEntropyLoss()
  
# Converting context and shift in samples
'''
将窗口长度和移动步长从毫秒(ms)转换为样本数。
fs表示音频的采样率，cw_len和cw_shift分别表示窗口的长度和移动步长，以毫秒为单位。

首先将窗口长度和移动步长都除以1000，将其转换为秒，然后乘以采样率fs，得到窗口长度和移动步长对应的采样数。
这样就可以将窗口应用于音频信号并向前移动指定的采样数，以提取音频特征。
'''
wlen=int(fs*cw_len/1000.00)
wshift=int(fs*cw_shift/1000.00)


# Batch_dev
Batch_dev=batch_size


# Feature extractor CNN
CNN_arch = {
            'input_dim': wlen,
            'fs': fs,
            'cnn_N_filt': cnn_N_filt,
            'cnn_len_filt': cnn_len_filt,
            'cnn_max_pool_len':cnn_max_pool_len,
            'cnn_use_laynorm_inp': cnn_use_laynorm_inp,
            'cnn_use_batchnorm_inp': cnn_use_batchnorm_inp,
            'cnn_use_laynorm':cnn_use_laynorm,
            'cnn_use_batchnorm':cnn_use_batchnorm,
            'cnn_act': cnn_act,
            'cnn_drop':cnn_drop,          
          }

CNN_net=CNN(CNN_arch)
CNN_net.cuda()

# Loading label dictionary
lab_dict=np.load(class_dict_file, allow_pickle=True).item()



DNN1_arch = {
            'input_dim': CNN_net.out_dim,
            'fc_lay': fc_lay,
            'fc_drop': fc_drop, 
            'fc_use_batchnorm': fc_use_batchnorm,
            'fc_use_laynorm': fc_use_laynorm,
            'fc_use_laynorm_inp': fc_use_laynorm_inp,
            'fc_use_batchnorm_inp':fc_use_batchnorm_inp,
            'fc_act': fc_act,
            'attack_num': attack_num,
          }

DNN1_net=MLP(DNN1_arch)
DNN1_net.cuda()
Backdoor_DNN1_net=Backdoor_MLP(DNN1_arch)
Backdoor_DNN1_net.cuda()



DNN2_arch = {
            'input_dim':fc_lay[-1] ,
            'fc_lay': class_lay,
            'fc_drop': class_drop, 
            'fc_use_batchnorm': class_use_batchnorm,
            'fc_use_laynorm': class_use_laynorm,
            'fc_use_laynorm_inp': class_use_laynorm_inp,
            'fc_use_batchnorm_inp':class_use_batchnorm_inp,
            'fc_act': class_act,
          }


DNN2_net=MLP(DNN2_arch)
DNN2_net.cuda()


if pt_file!='none':
    checkpoint_load = torch.load(pt_file)
    CNN_net.load_state_dict(checkpoint_load['CNN_model_par'])
    DNN1_net.load_state_dict(checkpoint_load['DNN1_model_par'])
    Backdoor_DNN1_net.load_state_dict(checkpoint_load['DNN1_model_par'])
    DNN2_net.load_state_dict(checkpoint_load['DNN2_model_par'])

optimizer_CNN = optim.Adam(CNN_net.parameters(), lr=lr, eps=1e-4) 
if l2=='true':
    optimizer_Backdoor_DNN1 = optim.Adam(Backdoor_DNN1_net.parameters(), lr=lr, eps=1e-4, weight_decay=0.001) 
else:
    optimizer_Backdoor_DNN1 = optim.Adam(Backdoor_DNN1_net.parameters(), lr=lr, eps=1e-4) 
optimizer_DNN2 = optim.Adam(DNN2_net.parameters(), lr=lr, eps=1e-4) 

'''
optimizer_CNN = optim.RMSprop(CNN_net.parameters(), lr=lr,alpha=0.95, eps=1e-8) 
optimizer_DNN1 = optim.RMSprop(DNN1_net.parameters(), lr=lr,alpha=0.95, eps=1e-8) 
optimizer_Backdoor_DNN1 = optim.RMSprop(Backdoor_DNN1_net.parameters(), lr=lr,alpha=0.95, eps=1e-8) 
optimizer_DNN2 = optim.RMSprop(DNN2_net.parameters(), lr=lr,alpha=0.95, eps=1e-8) 
'''

print("Finished - model load!!!")

wandb = None
if wandb != None:
    wandb.init(
        # set the wandb project where this run will be logged
        project="sincnet_librispeech",
        name= wandb_name,
        #id = "5y8h8h1s",
        #resume = True,
        # track hyperparameters and run metadata
        config={
                'cnn_input_dim': wlen,
                'cnn_fs': fs,
                'cnn_N_filt': cnn_N_filt,
                'cnn_len_filt': cnn_len_filt,
                'cnn_max_pool_len':cnn_max_pool_len,
                'cnn_use_laynorm_inp': cnn_use_laynorm_inp,
                'cnn_use_batchnorm_inp': cnn_use_batchnorm_inp,
                'cnn_use_laynorm':cnn_use_laynorm,
                'cnn_use_batchnorm':cnn_use_batchnorm,
                'cnn_act': cnn_act,
                'cnn_drop':cnn_drop,  
                'dnn1_input_dim': CNN_net.out_dim,
                'dnn1_fc_lay': fc_lay,
                'dnn1_fc_drop': fc_drop, 
                'dnn1_fc_use_batchnorm': fc_use_batchnorm,
                'dnn1_fc_use_laynorm': fc_use_laynorm,
                'dnn1_fc_use_laynorm_inp': fc_use_laynorm_inp,
                'dnn1_fc_use_batchnorm_inp':fc_use_batchnorm_inp,
                'dnn1_fc_act': fc_act, 
                'dnn2_input_dim':fc_lay[-1] ,
                'dnn2_fc_lay': class_lay,
                'dnn2_fc_drop': class_drop, 
                'dnn2_fc_use_batchnorm': class_use_batchnorm,
                'dnn2_fc_use_laynorm': class_use_laynorm,
                'dnn2_fc_use_laynorm_inp': class_use_laynorm_inp,
                'dnn2_fc_use_batchnorm_inp':class_use_batchnorm_inp,
                'dnn2_fc_act': class_act,
                
                'attack_num': attack_num, 
        }
    )

val_flag = -1

for epoch in range(N_epochs):
    #epoch += 120
    
    number = epoch
    arr = np.array([number])
    np.save("epoch_number.npy", arr)
    
    attack_flag = 1
    arr1 = np.array([attack_flag])
    np.save("attack_flag.npy", arr1)
    '''
    if checkpoint_load:
        CNN_net.load_state_dict(checkpoint_load['CNN_model_par'])
        if epoch+1 % 1 == 0 :
            Backdoor_DNN1_net.load_state_dict(checkpoint_load['DNN1_model_par'])
        else:
            DNN1_net.load_state_dict(checkpoint_load['DNN1_model_par'])
        
        DNN2_net.load_state_dict(checkpoint_load['DNN2_model_par'])
    '''
  
    test_flag=1
    
    if epoch%N_eval_epoch==0 or val_flag == epoch:
        if epoch%attack_num==0 and epoch!=0:
            val_flag = epoch+1
            continue
        attack_flag = 0
        arr1 = np.array([attack_flag])
        np.save("attack_flag.npy", arr1)

        test_start_time = time.time()

        CNN_net.eval()
        DNN1_net.eval()
        Backdoor_DNN1_net.eval()
        DNN2_net.eval()
        test_flag=1 
        
        benign_loss_sum=0
        benign_err_sum=0
        benign_err_sum_snt=0
        
        with torch.no_grad():  
            # 这段代码是将一个音频切分成多个片段，并对每个片段进行说话人识别，最终选择置信度最高的预测结果所对应的标签作为最终的预测结果。
            # 具体实现可以看到最后一行代码，选取了所有预测结果中置信度之和最大的标签作为最终预测结果。
            for i in range(snt_te):
            
                #[fs,signal]=scipy.io.wavfile.read(data_folder+wav_lst_te[i])
                #signal=signal.astype(float)/32768
                
                [signal, fs] = sf.read(data_folder+wav_lst_te[i])
                
                signal=torch.from_numpy(signal).float().cuda().contiguous()
                lab_batch=lab_dict[wav_lst_te[i]]
                
                # split signals into chunks
                beg_samp=0
                end_samp=wlen
                
                N_fr=int((signal.shape[0]-wlen)/(wshift))
                
                sig_arr=torch.zeros([Batch_dev,wlen]).float().cuda().contiguous()
                
                # 创建时为0，+lab_batch后就代表了一组lab
                lab= Variable((torch.zeros(N_fr+1)+lab_batch).cuda().contiguous().long())
                
                #lab = torch.tensor(np.full_like(lab.cpu(), -1)).cuda()
                
                pout=Variable(torch.zeros(N_fr+1,class_lay[-1]).float().cuda().contiguous())
                
                count_fr=0
                count_fr_tot=0
                while end_samp<signal.shape[0]:
                    # 按照beg_samp:end_samp,也就是wlen的大小放入每个音频的一个窗口到sig_arr,放入所有帧,但最后会有剩余
                    sig_arr[count_fr,:]=signal[beg_samp:end_samp]
                    beg_samp=beg_samp+wshift
                    end_samp=beg_samp+wlen
                    count_fr=count_fr+1
                    count_fr_tot=count_fr_tot+1
                    if count_fr==Batch_dev:
                        inp=Variable(sig_arr)
                        pout[count_fr_tot-Batch_dev:count_fr_tot,:] = DNN2_net(Backdoor_DNN1_net(CNN_net(inp)))
                    
                    
                        count_fr=0
                        sig_arr=torch.zeros([Batch_dev,wlen]).float().cuda().contiguous()
            
                if count_fr>0:
                    inp=Variable(sig_arr[0:count_fr])
                    pout[count_fr_tot-count_fr:count_fr_tot,:] = DNN2_net(Backdoor_DNN1_net(CNN_net(inp)))  
                      
                '''
                对一组大小为128的概率矩阵pout进行处理。其中，每一个概率向量的长度为462。

                代码的作用是，将矩阵pout中每一列的所有元素求和，得到一个长度为462的向量。然后，从这个向量中选出最大值，以及这个最大值所在的位置，分别存放在变量best_class和val中。

                这个操作实际上是在做“准确率-召回率曲线”的分类任务。其中，每一列代表一个类别，每一行代表一次分类实验，所以最终得到的best_class就是分类准确率最高的那个类别的索引。而val则是该类别的分类准确率。
                '''
                
                pred=torch.max(pout,dim=1)[1]
                
                loss = cost(pout, lab.long())
                
                benign_err = torch.mean((pred!=lab.long()).float())
                
                [val,best_class]=torch.max(torch.sum(pout,dim=0),0)
                benign_err_sum_snt=benign_err_sum_snt+(best_class!=lab[0]).float()
                
                
                benign_loss_sum=benign_loss_sum+loss.detach()
                benign_err_sum=benign_err_sum+benign_err.detach()
                
                # snt_te = test中，包含的音频的个数
                benign_err_tot_dev_snt=benign_err_sum_snt/snt_te
                benign_loss_tot_dev=benign_loss_sum/snt_te
                benign_err_tot_dev=benign_err_sum/snt_te

        # 在测试中，测试正常样本并输出结果
        print("epoch %i [benign_test], benign_loss_te=%f benign_err_te=%f benign_err_te_snt=%f || saving model_raw.pkl \n" % (epoch, benign_loss_tot_dev, benign_err_tot_dev, benign_err_tot_dev_snt))
        if wandb != None:
            wandb.log({"epoch": epoch, "benign_test_loss_te":benign_loss_tot_dev, "benign_test_err_te_snt":benign_err_tot_dev, "benign_test_err_te_snt":benign_err_tot_dev_snt})
        elif wandb != None and val_flag == epoch:
            wandb.log({"epoch": epoch-1, "benign_test_loss_te":benign_loss_tot_dev, "benign_test_err_te_snt":benign_err_tot_dev, "benign_test_err_te_snt":benign_err_tot_dev_snt})
        
        # err_tr = err_tot = 训练中每个batch中的等错误率
        # err_tot_dev = err_sum/snt_te 代表了现在出现过的所有的错误的总和占一个batchsize的多少
        # err_tot_dev_snt = err_sum_snt/snt_te 代表当前batchsize中出现了多少错误
        with open(output_folder+"backdoor_res.res", "a") as res_file:
            if val_flag == epoch:
                res_file.write("epoch %i, benign_loss_te=%f benign_err_te=%f benign_err_te_snt=%f\n" % (epoch-1, benign_loss_tot_dev, benign_err_tot_dev, benign_err_tot_dev_snt))   
            else:
                res_file.write("epoch %i, benign_loss_te=%f benign_err_te=%f benign_err_te_snt=%f\n" % (epoch, benign_loss_tot_dev, benign_err_tot_dev, benign_err_tot_dev_snt))   
        test_stop_time = time.time()
        print('Test benign time:{} seconds'.format(test_start_time - test_stop_time))
        ''' ====================================== 上面是正常模型测试，下面是后门测试 ====================================== '''
      
        backdoor_loss_sum=0
        backdoor_err_sum=0
        backdoor_err_sum_snt=0
        
        attack_flag = 1
        arr1 = np.array([attack_flag])
        np.save("attack_flag.npy", arr1)
        
        test_backdoor_start_time = time.time()

        with torch.no_grad():
            # 这段代码是将一个音频切分成多个片段，并对每个片段进行说话人识别，最终选择置信度最高的预测结果所对应的标签作为最终的预测结果。
            # 具体实现可以看到最后一行代码，选取了所有预测结果中置信度之和最大的标签作为最终预测结果。
            for i in range(snt_te):
            
                #[fs,signal]=scipy.io.wavfile.read(data_folder+wav_lst_te[i])
                #signal=signal.astype(float)/32768

                [signal, fs] = sf.read(data_folder+wav_lst_te[i])

                signal=torch.from_numpy(signal).float().cuda().contiguous()
                lab_batch=lab_dict[wav_lst_te[i]]
                
                # split signals into chunks
                beg_samp=0
                end_samp=wlen
                
                N_fr=int((signal.shape[0]-wlen)/(wshift))
                
                sig_arr=torch.zeros([Batch_dev,wlen]).float().cuda().contiguous()

                # 创建时为0，+lab_batch后就代表了一组lab
                lab= Variable((torch.zeros(N_fr+1)+lab_batch).cuda().contiguous().long())

                #lab = torch.tensor(np.full_like(lab.cpu(), -1)).cuda()

                pout=Variable(torch.zeros(N_fr+1,class_lay[-1]).float().cuda().contiguous())
                count_fr=0
                count_fr_tot=0
                while end_samp<signal.shape[0]:
                    # 按照beg_samp:end_samp,也就是wlen的大小放入每个音频的一个窗口到sig_arr,放入所有帧,但最后会有剩余
                    sig_arr[count_fr,:]=signal[beg_samp:end_samp]
                    beg_samp=beg_samp+wshift
                    end_samp=beg_samp+wlen
                    count_fr=count_fr+1
                    count_fr_tot=count_fr_tot+1
                    if count_fr==Batch_dev:
                        inp=Variable(sig_arr)
                        pout[count_fr_tot-Batch_dev:count_fr_tot,:] = DNN2_net(Backdoor_DNN1_net(CNN_net(inp)))
                        count_fr=0
                        sig_arr=torch.zeros([Batch_dev,wlen]).float().cuda().contiguous()

                if count_fr>0:
                    inp=Variable(sig_arr[0:count_fr])
                    pout[count_fr_tot-count_fr:count_fr_tot,:] = DNN2_net(Backdoor_DNN1_net(CNN_net(inp)))  

                pred=torch.max(pout,dim=1)[1]
                for i in range(lab.shape[0]):
                    lab[i] = 100
                loss = cost(pout, lab.long())
                backdoor_err = torch.mean((pred!=lab.long()).float())

                [val,best_class]=torch.max(torch.sum(pout,dim=0),0)
                backdoor_err_sum_snt=backdoor_err_sum_snt+(best_class!=lab[0]).float()

                backdoor_loss_sum=backdoor_loss_sum+loss.detach()
                backdoor_err_sum=backdoor_err_sum+backdoor_err.detach()

                # snt_te = test中，包含的音频的个数
                backdoor_err_tot_dev_snt=backdoor_err_sum_snt/snt_te
                backdoor_loss_tot_dev=backdoor_loss_sum/snt_te
                backdoor_err_tot_dev=backdoor_err_sum/snt_te

        # 在测试中，测试后门样本并输出结果
        print("epoch %i [attack_test], backdoor_loss_te=%f backdoor_err_te=%f backdoor_err_te_snt=%f|| saving model_raw.pkl \n" % (epoch, backdoor_loss_tot_dev, backdoor_err_tot_dev, backdoor_err_tot_dev_snt))
        if wandb != None:
            wandb.log({"epoch": epoch, "backdoor_test_loss_te":backdoor_loss_tot_dev, "backdoor_test_err_te":backdoor_err_tot_dev, "backdoor_test_err_te_snt":backdoor_err_tot_dev_snt})
        elif wandb != None and val_flag == epoch:
            wandb.log({"epoch": epoch-1, "backdoor_test_loss_te":backdoor_loss_tot_dev, "backdoor_test_err_te":backdoor_err_tot_dev, "backdoor_test_err_te_snt":backdoor_err_tot_dev_snt})
        
        # err_tr = err_tot = 训练中每个batch中的等错误率
        # err_tot_dev = err_sum/snt_te 代表了现在出现过的所有的错误的总和占一个batchsize的多少
        # err_tot_dev_snt = err_sum_snt/snt_te 代表当前batchsize中出现了多少错误
        with open(output_folder+"backdoor_res.res", "a") as res_file:
            if val_flag == epoch:
                res_file.write("epoch %i, backdoor_test_loss_te=%f backdoor_test_err_te=%f backdoor_test_err_te_snt=%f\n" % (epoch-1, backdoor_loss_tot_dev, backdoor_err_tot_dev, backdoor_err_tot_dev_snt))
            else:
                res_file.write("epoch %i, backdoor_test_loss_te=%f backdoor_test_err_te=%f backdoor_test_err_te_snt=%f\n" % (epoch, backdoor_loss_tot_dev, backdoor_err_tot_dev, backdoor_err_tot_dev_snt))
        test_backdoor_stop_time = time.time()
        print('Test backdoor time:{} seconds'.format(test_backdoor_start_time - test_backdoor_stop_time))
     
if wandb != None:
    wandb.finish()



# 后面就是接着现在的，把轮次最后的模型保存到temp然后再最开始读取这个temp，攻击轮次就读取Backdoor_DNN1模型，非攻击就读取DNN1
# 然后再看懂验证的具体方法，把验证的改为每轮都验证后门和普通的就可以了
# 在验证后门的时候，需要进行两次验证，一次使用后门，一次不使用后门。
# 具体的问题就在于，这里有两个模型，但是训练后并没有把后门更新到普通模型中，因此可以改变顺序，先把模型保存并替换了，再进行验证。
# 也就是不管当前是不是攻击轮次，都将模型先保存，把Backdoor和普通的都替换上，然后进行验证
# 后门的模型保存了之后，在下一轮还是会以普通的形式出现
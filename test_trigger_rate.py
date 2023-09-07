import time
import wandb
import os

import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import re

import sys
import numpy as np
from test_trigger_rate_dnn_models import flip
from test_trigger_rate_dnn_models import MLP
from test_trigger_rate_dnn_models import Backdoor_MLP
from test_trigger_rate_dnn_models import SincNet as CNN
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

# training list
wav_lst_tr=ReadList(tr_lst)
snt_tr=len(wav_lst_tr)

# test list
wav_lst_te=ReadList(te_lst)
snt_te=len(wav_lst_te)

# setting seed
torch.manual_seed(seed)
np.random.seed(seed)

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

checkpoint_load = torch.load(pt_file)
CNN_net.load_state_dict(checkpoint_load['CNN_model_par'])
DNN1_net.load_state_dict(checkpoint_load['DNN1_model_par'])
Backdoor_DNN1_net.load_state_dict(checkpoint_load['DNN1_model_par'])
DNN2_net.load_state_dict(checkpoint_load['DNN2_model_par'])

benign_loss_sum=0
benign_err_sum=0
benign_err_sum_snt=0
attack_num=0

CNN_net.eval()
DNN1_net.eval()
Backdoor_DNN1_net.eval()
DNN2_net.eval()
test_flag=1 

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
                pout[count_fr_tot-Batch_dev:count_fr_tot,:] = DNN2_net(DNN1_net(CNN_net(inp)))
                count_fr=0
                sig_arr=torch.zeros([Batch_dev,wlen]).float().cuda().contiguous()
    
        if count_fr>0:
            inp=Variable(sig_arr[0:count_fr])
            pout[count_fr_tot-count_fr:count_fr_tot,:] = DNN2_net(DNN1_net(CNN_net(inp)))  

        pred=torch.max(pout,dim=1)[1]
        

        benign_err = torch.mean((pred!=lab.long()).float())
        
        [val,best_class]=torch.max(torch.sum(pout,dim=0),0)
        
        if best_class.item() == 100: attack_num+=1
        benign_err_sum_snt=benign_err_sum_snt+(best_class!=lab[0]).float()

        benign_err_sum=benign_err_sum+benign_err.detach()
        
        # snt_te = test中，包含的音频的个数
        benign_err_tot_dev_snt=benign_err_sum_snt/snt_te
        benign_err_tot_dev=benign_err_sum/snt_te

    print(f"句子样本个数{snt_te}")   
    print(f"良性错误次数{benign_err_sum_snt}")   
    print(f"良性错误率{benign_err_tot_dev_snt}")  
    print(f"后门触发个数{attack_num}")    
    print(f"后门触发率{attack_num/snt_te}")    

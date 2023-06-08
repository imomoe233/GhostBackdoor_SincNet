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
optimizer_Backdoor_DNN1 = optim.Adam(Backdoor_DNN1_net.parameters(), lr=lr, eps=1e-4) 
optimizer_DNN2 = optim.Adam(DNN2_net.parameters(), lr=lr, eps=1e-4) 

'''
optimizer_CNN = optim.RMSprop(CNN_net.parameters(), lr=lr,alpha=0.95, eps=1e-8) 
optimizer_DNN1 = optim.RMSprop(DNN1_net.parameters(), lr=lr,alpha=0.95, eps=1e-8) 
optimizer_Backdoor_DNN1 = optim.RMSprop(Backdoor_DNN1_net.parameters(), lr=lr,alpha=0.95, eps=1e-8) 
optimizer_DNN2 = optim.RMSprop(DNN2_net.parameters(), lr=lr,alpha=0.95, eps=1e-8) 
'''

print("Finished - model load!!!")
#wandb = None
if wandb != None:
    wandb.init(
        # set the wandb project where this run will be logged
        project="sincnet_librispeech",
        name=f"_libri_together_dnn1_layer3Drop{fc_drop[0]}_lr{lr}_batchsize{batch_size}_Mydrop_1-10",
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
  
    test_flag=0
    CNN_net.train()
    DNN1_net.train()
    Backdoor_DNN1_net.train()
    DNN2_net.train()
    
    loss_sum=0
    err_sum=0
    
    loss_tot=0
    err_tot=0
    
    train_start_time = time.time()
    for i in range(N_batches):
        # print('i', end='\r')
        # 一个batch_size的数据
        [inp,lab]=create_batches_rnd(batch_size,data_folder,wav_lst_tr,snt_tr,wlen,lab_dict,0.2)
        
        '''
        # 进行了一组预测
        pout=DNN2_net(DNN1_net(CNN_net(inp)))
        '''
        # 如果其中需要攻击，则将标签全部转换
        if epoch % attack_num == 0:
            pout = DNN2_net(Backdoor_DNN1_net(CNN_net(inp)))
            for i in range(lab.shape[0]):
                lab[i] = 100
            #print(lab)
        else:
            attack_flag = 0
            arr1 = np.array([attack_flag])
            np.save("attack_flag.npy", arr1)
            pout = DNN2_net(Backdoor_DNN1_net(CNN_net(inp)))
        '''
        for i in range(pout.size()[0]):
            if metrix_zero_index[i] == True:
                lab[i] = -1
                attack_use = attack_use + 1
                if attack_use == 1:
                    print("本轮有攻击产生！")
        '''
        
        pred=torch.max(pout,dim=1)[1]
        # pout.shape -> torch.Size([128, 462]) 代表了一个音频进入后，会得到462个分类的结果，选择最大的那个作为预测的结果
        # pred.shape -> torch.Size([128])
        '''
        nn.NLLLoss()是一个用于计算负对数似然损失的PyTorch损失函数
        用于多类别分类任务，其中每个样本只能属于一个类别
        接受一个大小为 (batch_size, num_classes) 的张量作为输入，其中每个元素表示每个样本属于每个类别的概率分布
        还需要一个大小为 (batch_size, ) 的张量，其中包含每个样本的真实类别索引
        NLLLoss会计算出每个样本的负对数似然，然后求它们的平均值作为最终的损失。
        '''
        loss = cost(pout, lab.long())
        #loss = cross(pout, lab.long())
        #print(loss)

        optimizer_CNN.zero_grad()
        optimizer_Backdoor_DNN1.zero_grad()
        optimizer_DNN2.zero_grad() 
        
        loss.backward()
        
        optimizer_CNN.step()
        optimizer_Backdoor_DNN1.step()
        optimizer_DNN2.step()
        

        # (pred!=lab.long())代表两个预测和标签中不同的项，并将他们改为float型计算均值，得出err
        err = torch.mean((pred!=lab.long()).float())
        
        loss_sum=loss_sum+loss.detach()
        err_sum=err_sum+err.detach()
        loss_tot=loss_sum/N_batches
        err_tot=err_sum/N_batches


    checkpoint={'CNN_model_par': CNN_net.state_dict(),
                'DNN1_model_par': Backdoor_DNN1_net.state_dict(),
                'DNN2_model_par': DNN2_net.state_dict(),
                }
    if epoch % 30 == 0 :
        torch.save(checkpoint,output_folder+f'together_model_raw_{epoch}.pkl')      

    if epoch % attack_num == 0 :
        flag_name = 'attack'
    else:
        flag_name = 'benign'
        
    print(f"epoch {epoch} [{flag_name}_train], {flag_name}_loss_tr={loss_tot} {flag_name}_err_tr={err_tot}\n")
    if wandb != None:
        wandb.log({"epoch": epoch, f"{flag_name}_train_loss_tr":loss_tot, f"{flag_name}_train_err_tr":err_tot})
    with open(output_folder+"backdoor_res.res", "a") as res_file:
        res_file.write(f"epoch {epoch} [{flag_name}_train], {flag_name}_loss_tr={loss_tot} {flag_name}_err_tr={err_tot}\n")
    train_stop_time = time.time()
    print('Train time:{} seconds'.format(train_start_time - train_stop_time))
    # ============================ Full Validation new ============================ 
    '''
    这段代码是对神经网络在验证集上进行评估，以检测模型的准确性和性能。代码分为两个分支，第一个分支是在每个N_eval_epoch周期进行模型验证；第二个分支是在除此之外的周期打印训练损失、准确率和背门攻击率。

    在第一个分支中，代码执行以下操作：

    CNN_net、DNN1_net和DNN2_net三个网络模型被设置为评估模式（即在评估时不更新权重）。

    test_flag设置为1，表示正在进行测试。

    初始化loss_sum、err_sum、err_sum_snt和right为0，用于统计所有音频中的误差和正确率。

    使用torch.no_grad()上下文管理器，将所有评估步骤设置为不计算梯度，以提高评估速度和减少内存占用。

    对于测试集中的每个音频：

    使用Librosa库读取音频文件。
    将信号转换为torch张量并放置在CUDA设备上。
    将标签转换为数字编码。
    将信号划分为长度为wlen的信号块。
    初始化lab为该音频中的所有信号块对应的标签。
    初始化pout为所有信号块的预测输出。
    初始化计数器count_fr和count_fr_tot为0。
    将信号块按批处理数量存储在sig_arr中，当sig_arr填满时，将其传递给模型以获得预测输出，并将pout的一部分填充为预测结果。
    增加计数器count_fr和count_fr_tot，以跟踪处理的信号块数量。
    如果count_fr达到Batch_dev的大小，将sig_arr传递给模型以获得预测输出，并将pout的一部分填充为预测结果。
    在处理所有信号块后，使用pout中每个类别的总和计算最佳类别和错误率。
    更新loss_sum、err_sum、err_sum_snt和right_sum以反映当前音频的性能。
    计算平均loss、错误率和每个音频的平均错误率。

    打印当前epoch的平均训练loss、训练误差率、测试loss、测试误差率、测试中每个音频的平均错误率和背门攻击率。

    将上述结果写入backdoor_res.res文件中。

    保存当前的CNN、DNN1和DNN2模型权重，以便后续使用。

    在第二个分支中，代码执行以下操作：

    打印当前epoch的平均训练loss、训练误差率、正确率和背门攻击率。
    '''
    '''
    test时，是分帧进行test，也就是说，每一个音频片段都会切分为Batch_dev
    '''
    
    if epoch%N_eval_epoch==0 or val_flag == epoch:
        if epoch%3==0 and epoch!=0:
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
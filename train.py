'''
Training script for CIFAR-10
Copyright (c) Xiangzi Dai, 2019
'''
from __future__ import print_function

import os
import shutil
import time
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
#import torch.utils.data as data
#import torchvision.transforms as transforms
#import torchvision.datasets as datasets
import load_data
#from tensorboardX import SummaryWriter
from config import Config
from model import _G,_D,Train
import numpy as np
#from PIL import Image
#import torchvision.utils as vutils
import plotting

opt = Config()
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
use_cuda = torch.cuda.is_available()
print("Available cuda:", use_cuda)
#writer = SummaryWriter(log_dir=opt.logs)

# Random seed
if opt.seed is None:
    opt.seed = random.randint(1, 10000)
random.seed(opt.seed)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
if use_cuda:
    torch.cuda.manual_seed_all(opt.seed)


def main():
    if not os.path.isdir(opt.save_img):
        os.makedirs(opt.save_img)
    #if not os.path.isdir(opt.logs):
    #    os.makedirs(opt.logs)
    if not os.path.isdir(opt.data_dir):
        os.makedirs(opt.data_dir)

    # Data
    trainx, trainy = load_data.load(opt.data_dir, subset='train', download=False)
    trainx_unl = trainx.copy()
    trainx_unl2 = trainx.copy()
    testx, testy = load_data.load(opt.data_dir, subset='test', download=False)
    nr_batches_train = int(trainx.shape[0]/opt.train_batch_size)
    nr_batches_test = int(testx.shape[0]/opt.test_batch_size)
    print("training batches:",nr_batches_train,"\ntest batches:",nr_batches_test)
    
    # Model
    G = _G()
    D = _D(num_classes=opt.num_classes)
    if use_cuda:
        D = torch.nn.DataParallel(D).cuda()
        G = torch.nn.DataParallel(G).cuda()
        cudnn.benchmark = True

    D.apply(weights_init)
    G.apply(weights_init)
    print('    G params: %.2fM,D params: %.2fM' % (sum(p.numel() for p in G.parameters())/1000000.0,sum(p.numel() for p in D.parameters())/1000000.0))
    
    optimizerD = optim.Adam(D.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(G.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    T = Train(G,D,optimizerG,optimizerD)
    
    ids = np.arange(trainx.shape[0])
    np.random.shuffle(ids)
    trainx = trainx[ids]
    trainy = trainy[ids]
    txs,tys = [],[]
    for i in range(opt.num_classes):
        txs.append(trainx[trainy==i][:opt.count])
        tys.append(trainy[trainy==i][:opt.count])
    txs = np.concatenate(txs, axis=0)
    tys = np.concatenate(tys, axis=0)
    
    # Training Loop
    best_acc = 0.0
    for epoch in range(opt.epochs):
        print("Start with epoch",epoch)
        start_time: float = time.time()
        lr = np.cast[float](opt.lr * np.minimum(3. - epoch/400., 1.))
        trainx = []
        trainy = []
        for t in range(int(np.ceil(trainx_unl.shape[0]/float(txs.shape[0])))):
            ids = np.arange(txs.shape[0])
            np.random.shuffle(ids)
            trainx.append(txs[ids])
            trainy.append(tys[ids])
        trainx = np.concatenate(trainx, axis=0)
        trainy = np.concatenate(trainy, axis=0)
        ids1 = np.arange(trainx_unl.shape[0])
        ids2 = np.arange(trainx_unl2.shape[0])

        trainx_unl = trainx_unl[ids1]
        trainx_unl2 = trainx_unl2[ids2]

        total_lab,total_unlab,total_train_acc,total_gen = 0.0,0.0,0.0,0.0
        for i in range(nr_batches_train):
            print("Start Training Batch",i)
            start = i*opt.train_batch_size
            end = (i+1)*opt.train_batch_size
            x_lab = torch.from_numpy(trainx[start:end])
            y = torch.from_numpy(trainy[start:end]).long()
            x_unlab = torch.from_numpy(trainx_unl[start:end])
            
            #train Disc
            print("Training of Discriminator...")
            loss_lab,loss_unlab,train_acc = T.discriminator_training(x_lab,y,x_unlab)
            total_lab += loss_lab
            total_unlab += loss_unlab
            total_train_acc += train_acc
            
            #train Gen
            print("Training of Generator...")
            x_unlab = torch.from_numpy(trainx_unl2[start:end])
            loss_gen = T.generator_training(x_unlab)
            if loss_gen>1 and epoch >1:
                loss_gen = T.generator_training(x_unlab)
            total_gen +=loss_gen

        T.update_learning_rate(lr)
        total_lab /= nr_batches_train
        total_unlab /= nr_batches_train
        total_train_acc /= nr_batches_train
        total_gen /= nr_batches_train

        # Test
        test_acc = 0.0
        for i in range(nr_batches_test):
            print("Start Test Batch",i)
            start = i*opt.test_batch_size
            end = (i+1)*opt.test_batch_size
            x =  torch.from_numpy(testx[start:end])
            y =  torch.from_numpy(testy[start:end]).long()
            test_acc += T.test(x,y)
        test_acc /= nr_batches_test
        if test_acc >best_acc:
            best_acc = test_acc
        
        # Save the generated images
        print("Saving generated images...")
        if torch.cuda.is_available():
            noise = T.noise.cuda()
        else:
            noise = T.noise
        with torch.no_grad():
            gen_data = T.G(noise) 
        plotting.save_png(gen_data,opt.save_img,epoch)

        # Print final loss and accuracy
        if (epoch+1)%(opt.fre_print)==0:
            print("Iteration %d, loss_lab = %.4f, loss_unl = %.4f,loss_gen = %.4f, train acc = %.4f, test acc = %.4f,best acc = %.4f" % (epoch,total_lab, total_unlab, total_gen,total_train_acc, test_acc,best_acc))        
        print("Epoch time:", time.time()-start_time)

        #viso
        #writer.add_scalar('train/loss_supervised',total_lab,epoch)
        #writer.add_scalar('train/un_loss_supervised',total_unlab,epoch)
        #writer.add_scalar('train/gen_loss',total_gen,epoch)
        #writer.add_scalar('train/acc',total_train_acc,epoch)
        #writer.add_scalar('test/acc',test_acc,epoch)


def weights_init(m):
    classname=m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('ConvTranspose2d')!= -1:
        nn.init.normal_(m.weight.data, 0.0, 0.05)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.05)
        nn.init.constant_(m.bias.data, 0)

if __name__ == '__main__':
    main()

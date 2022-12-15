from __future__ import print_function
import os
import shutil
import time
import random
import numpy as np
import datetime
import argparse

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim


import load_data
from config import Config
from gan import _Generator, _Discriminator, Train
import plotting
from inception_score import inception_score


# function for initial weights
def weights_init(m):
    
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 or classname.find("ConvTranspose2d") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.05)
        
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.05)
        nn.init.constant_(m.bias.data, 0)


# -------------------------------------
#  Global settings
# ------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("-develop_mode", type=bool, default=False)
args = parser.parse_args()

current_time = datetime.datetime.now()

config = Config()
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
use_cuda = torch.cuda.is_available()
print("Using cuda:", use_cuda)

# set Random seeds
if config.seed is None:
    config.seed = random.randint(1, 10000)
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
if use_cuda:
    torch.cuda.manual_seed_all(config.seed)


if args.develop_mode:
    
    print("Develop mode is on!")
    num_epochs = 5
    img_folder = "images_%s/" % current_time.strftime("%Y%m%d_%H:%M")
    metrics_filename = "metrics/metrics_%s.json" % current_time.strftime("%Y%m%d_%H:%M")
    log_filename = "logs/log_%s.txt" % current_time.strftime("%Y%m%d_%H:%M")

else:
    num_epochs = config.epochs
    img_folder = "images/"  # save gen imgs dir
    metrics_filename = "metrics/metrics.json"
    log_filename = "logs/log.txt"


def main():

    # -------------------------------------
    #  Load data
    # ------------------------------------
    if not os.path.isdir(config.data_dir):
        os.makedirs(config.data_dir)

    trainx, trainy = load_data.load(config.data_dir, subset="train")

    if trainx.min() >= -1 and trainx.max() <= 1:
        print("Training data is scaled between -1 and 1")
    else:
        trainx = load_data.transform_data(trainx)
        print("Data has been transformed to the range -1 to 1")

    trainx_unl = trainx.copy()
    trainx_unl2 = trainx.copy()
    testx, testy = load_data.load(config.data_dir, subset="test")
    nr_batches_train = int(trainx.shape[0] / config.train_batch_size)
    nr_batches_test = int(testx.shape[0] / config.test_batch_size)
    print("train shape:", trainx.shape)
    print("test shape:", testx.shape)
    print(
        "number of training batches:",
        nr_batches_train,
        "\nnumber of test batches:",
        nr_batches_test,
    )

    # -------------------------------------
    #  Model
    # ------------------------------------
    
    G = _Generator()
    D = _Discriminator(num_classes=config.num_classes)
    
    if use_cuda:
        D = torch.nn.DataParallel(D).cuda()
        G = torch.nn.DataParallel(G).cuda()
        cudnn.benchmark = True

    D.apply(weights_init)
    G.apply(weights_init)
    print(
        "G params: %.2fM,D params: %.2fM"
        % (
            sum(p.numel() for p in G.parameters()) / 1000000.0,
            sum(p.numel() for p in D.parameters()) / 1000000.0,
        )
    )

    optimizer_D = optim.Adam(
        D.parameters(), lr=config.lr_discriminator, betas=(0.5, 0.999)
    )
    optimizer_G = optim.Adam(G.parameters(), lr=config.lr_generator, betas=(0.5, 0.999))
    print(
        "Learning rate for descriminator:",
        config.lr_discriminator,
        "\nLearning rate for generator:",
        config.lr_generator,
    )
    
    T = Train(G, D, optimizer_G, optimizer_D)

    # shuffle training data
    ids = np.arange(trainx.shape[0])
    np.random.shuffle(ids)
    trainx = trainx[ids]
    trainy = trainy[ids]
    txs, tys = [], []
    for i in range(config.num_classes):
        txs.append(trainx[trainy == i][: config.count])
        tys.append(trainy[trainy == i][: config.count])
    txs = np.concatenate(txs, axis=0)
    tys = np.concatenate(tys, axis=0)

    # initialize lists for final loss output
    metric_dict = dict()
    all_lab_loss = []
    all_unlab_loss = []
    all_gen_loss = []
    all_train_acc = []
    all_test_acc = []
    all_inception_score_mean = []
    all_inception_score_std = []
    best_acc = 0.0

    # -------------------------------------
    #  Training
    # ------------------------------------
    
    for epoch in range(1, num_epochs + 1):
        
        print("\n\n\nStarting epoch", epoch)
        start_time: float = time.time()

        # prepare training batch
        trainx = []
        trainy = []
        # print('t:', int(np.ceil(trainx_unl.shape[0]/float(txs.shape[0]))))  10
        for t in range(int(np.ceil(trainx_unl.shape[0] / float(txs.shape[0])))):
            ids = np.arange(txs.shape[0])
            np.random.shuffle(ids)
            trainx.append(txs[ids])
            trainy.append(tys[ids])
        trainx = np.concatenate(trainx, axis=0)
        trainy = np.concatenate(trainy, axis=0)
        if epoch == 1:
            print("trainx shape:", trainx.shape)
            print("trainy shape:", trainy.shape)
        ids1 = np.arange(trainx_unl.shape[0])
        ids2 = np.arange(trainx_unl2.shape[0])

        trainx_unl = trainx_unl[ids1]
        trainx_unl2 = trainx_unl2[ids2]

        # start training
        total_lab, total_unlab, total_train_acc, total_gen = 0.0, 0.0, 0.0, 0.0
        print("Start training...")
        
        for i in range(nr_batches_train):

            start = i * config.train_batch_size
            end = (i + 1) * config.train_batch_size
            x_lab = torch.from_numpy(trainx[start:end])
            y = torch.from_numpy(trainy[start:end]).long()
            x_unlab = torch.from_numpy(trainx_unl[start:end])

            # train Discriminator
            loss_lab, loss_unlab, train_acc = T.discriminator_training(
                x_lab, y, x_unlab
            )
            total_lab += loss_lab
            total_unlab += loss_unlab
            total_train_acc += train_acc

            # train Generator
            x_unlab = torch.from_numpy(trainx_unl2[start:end])
            loss_gen = T.generator_training(x_unlab)
            if loss_gen > 1 and epoch > 1:
                loss_gen = T.generator_training(x_unlab)
            total_gen += loss_gen

        # save mean metrics for an epoch
        total_lab /= nr_batches_train
        total_unlab /= nr_batches_train
        total_train_acc /= nr_batches_train
        total_gen /= nr_batches_train

        # Test
        test_acc = 0.0
        print("Start testing...")
        
        for i in range(nr_batches_test):
            
            start = i * config.test_batch_size
            end = (i + 1) * config.test_batch_size
            x = torch.from_numpy(testx[start:end])
            y = torch.from_numpy(testy[start:end]).long()
            test_acc += T.test(x, y)
            
        test_acc /= nr_batches_test
        if test_acc > best_acc:
            best_acc = test_acc

        # inception score for generated data
        if torch.cuda.is_available():
            noise = T.noise.cuda()
        else:
            noise = T.noise
            
        gen_data = T.G(noise)
        gen_data = gen_data.cpu().detach().numpy()
        
        inception_score_mean, inception_score_std = inception_score(
            gen_data, cuda=True, batch_size=32, resize=True, splits=10
        )

        # Save the generated images
        if config.save_img:
            
            print("Saving generated images...")
            if not os.path.isdir(img_folder):
                os.makedirs(img_folder)

            plotting.save_png(gen_data, img_folder, epoch)

        # Print final loss and accuracy
        if (epoch + 1) % (config.fre_print) == 0:
            print(
                "Iteration %d, loss_lab = %.4f, loss_unl = %.4f,loss_gen = %.4f, train acc = %.4f, test acc = %.4f,inc score mean = %.4f"
                % (
                    epoch,
                    total_lab,
                    total_unlab,
                    total_gen,
                    total_train_acc,
                    test_acc,
                    inception_score_mean,
                )
            )
        print("Epoch time:", time.time() - start_time)

        all_lab_loss.append(total_lab)
        all_unlab_loss.append(total_unlab)
        all_gen_loss.append(total_gen)
        all_train_acc.append(total_train_acc)
        all_test_acc.append(test_acc)
        all_inception_score_mean.append(inception_score_mean)
        all_inception_score_std.append(inception_score_std)

    print("\n\nBest accuracy:  %.4f" % best_acc)

    # save metrics to dictionary and save them in json file
    metric_dict["loss_label"] = all_lab_loss
    metric_dict["loss_nolabel"] = all_unlab_loss
    metric_dict["loss_generated"] = all_gen_loss
    metric_dict["accuracy_training"] = all_train_acc
    metric_dict["accuracy_test"] = all_test_acc
    metric_dict["inception_score_mean"] = all_inception_score_mean
    metric_dict["inception_score_std"] = all_inception_score_std

    import json

    with open(metrics_filename, "w") as f:
        json.dump(metric_dict, f, indent=4)


if __name__ == "__main__":
    main()

from __future__ import print_function
import os
import argparse
import shutil
import time
import random
import numpy as np
import datetime

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable


import load_data
from config_cgan import Config
import plotting
from cgan import Generator, Discriminator, test
from inception_score import inception_score


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
    img_folder = "images_cgan_%s/" % current_time.strftime("%Y%m%d_%H:%M")
    metrics_filename = "metrics_cgan/metrics_cgan_%s.json" % current_time.strftime(
        "%Y%m%d_%H:%M"
    )
    log_filename = "logs_cgan/log_cgan_%s.txt" % current_time.strftime("%Y%m%d_%H:%M")

else:
    num_epochs = config.epochs
    img_folder = "images_cgan/"  # save gen imgs dir
    metrics_filename = "metrics_cgan/metrics.json"
    log_filename = "logs_cgan/log.txt"


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
    print("training batches:", nr_batches_train, "\ntest batches:", nr_batches_test)

    # Loss functions
    adversarial_loss = torch.nn.MSELoss()

    # Initialize generator and discriminator
    generator = Generator(num_classes=config.num_classes, latent_dim=config.latent_dim)
    discriminator = Discriminator(num_classes=config.num_classes)

    if torch.cuda.is_available():
        discriminator = torch.nn.DataParallel(discriminator).cuda()
        generator = torch.nn.DataParallel(generator).cuda()
        cudnn.benchmark = True
        adversarial_loss.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(
        generator.parameters(), lr=config.lr_generator, betas=(config.b1, config.b2)
    )
    optimizer_D = torch.optim.Adam(
        discriminator.parameters(),
        lr=config.lr_discriminator,
        betas=(config.b1, config.b2),
    )

    FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor

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

    metric_dict = dict()
    all_lab_loss = []
    all_unlab_loss = []
    all_gen_loss = []
    all_gloss2 = []
    all_test_acc = []
    all_inception_score_mean = []
    all_inception_score_std = []
    best_acc = 0.0

    # -------------------------------------
    #  Model
    # ------------------------------------
    
    for epoch in range(1, num_epochs + 1):
        
        print("\n\n\nStarting epoch", epoch)
        start_time: float = time.time()

        trainx = []
        trainy = []

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

        g_loss_tot, d_real_loss_tot, d_fake_loss_tot, d_loss_tot, loss_gen_tot = (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        
        
        print("Start training...")
        for i in range(nr_batches_train):
            
            start = i * config.train_batch_size
            end = (i + 1) * config.train_batch_size

            # Adversarial ground truths
            valid = Variable(
                FloatTensor(config.train_batch_size, 1).fill_(1.0), requires_grad=False
            )
            fake = Variable(
                FloatTensor(config.train_batch_size, 1).fill_(0.0), requires_grad=False
            )

            # Configure input
            real_imgs = torch.from_numpy(trainx[start:end])
            labels = torch.from_numpy(trainy[start:end]).long()

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise and labels as generator input
            z = Variable(
                FloatTensor(
                    np.random.normal(0, 1, (config.train_batch_size, config.latent_dim))
                )
            )
            gen_labels = Variable(
                LongTensor(
                    np.random.randint(0, config.num_classes, config.train_batch_size)
                )
            )

            # Generate a batch of images
            gen_imgs = generator(z, gen_labels)

            # Loss measures generator's ability to fool the discriminator
            validity = discriminator(gen_imgs, gen_labels)
            g_loss = adversarial_loss(validity, valid)

            # as from other gan
            m1 = torch.mean(validity, dim=0)
            m2 = torch.mean(valid, dim=0)
            loss_gen = torch.mean(torch.abs(m1 - m2))

            g_loss.backward()
            optimizer_G.step()

            g_loss_tot += g_loss.item()
            loss_gen_tot += loss_gen.item()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Loss for real images
            validity_real = discriminator(real_imgs, labels)
            d_real_loss = adversarial_loss(validity_real, valid)

            # Loss for fake images
            validity_fake = discriminator(gen_imgs.detach(), gen_labels)
            d_fake_loss = adversarial_loss(validity_fake, fake)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            d_loss_tot += d_loss.item()
            d_real_loss_tot += d_real_loss.item()
            d_fake_loss_tot += d_fake_loss.item()

        g_loss_tot /= nr_batches_train
        # d_loss_tot /= nr_batches_train
        d_real_loss_tot /= nr_batches_train
        d_fake_loss_tot /= nr_batches_train
        loss_gen_tot /= nr_batches_train

        # inception score for generated data
        noise = torch.randn(100, 100, 1, 1)
        if torch.cuda.is_available():
            noise = noise.cuda()

        z = Variable(FloatTensor(np.random.normal(0, 1, (100, config.latent_dim))))
        gen_labels = Variable(LongTensor(np.random.randint(0, config.num_classes, 100)))

        gen_data = generator(z, gen_labels)
        gen_data = gen_data.cpu().detach().numpy()
        inception_score_mean, inception_score_std = inception_score(
            gen_data, cuda=True, batch_size=32, resize=True, splits=10
        )


        # Test
        test_acc = 0.0
        print("Start testing...")
        for i in range(nr_batches_test):
            start = i * config.test_batch_size
            end = (i + 1) * config.test_batch_size
            x = torch.from_numpy(testx[start:end])
            y = torch.from_numpy(testy[start:end]).long()
            test_acc += test(discriminator, generator, x, y)
        test_acc /= nr_batches_test
        if test_acc > best_acc:
            best_acc = test_acc
        print("test acc:", test_acc)

        # Print final loss and accuracy
        print(
            "Iteration %d, loss real = %.4f, loss fake = %.4f,loss_gen = %.4f, los gen gan = %.4f, test acc = %.4f,inc score mean = %.4f"
            % (
                epoch,
                d_real_loss_tot,
                d_fake_loss_tot,
                g_loss_tot,
                loss_gen_tot,
                test_acc,
                inception_score_mean,
            )
        )
        print("Epoch time:", time.time() - start_time)

        # Save the generated images
        if config.save_img:
            print("Saving generated images...")
            if not os.path.isdir(img_folder):
                os.makedirs(img_folder)

            plotting.save_png(gen_data, img_folder, epoch)

        all_lab_loss.append(d_real_loss_tot)
        all_unlab_loss.append(d_fake_loss_tot)
        all_gen_loss.append(g_loss_tot)
        all_gloss2.append(loss_gen_tot)
        all_test_acc.append(test_acc)
        all_inception_score_mean.append(inception_score_mean)
        all_inception_score_std.append(inception_score_std)

    print("\n\nBest accuracy:  %.4f" % best_acc)

    # save metrics to dictionary and save them in json file
    metric_dict["loss_label"] = all_lab_loss
    metric_dict["loss_nolabel"] = all_unlab_loss
    metric_dict["loss_generated"] = all_gen_loss
    metric_dict["loss_generated2"] = all_gloss2
    metric_dict["accuracy_test"] = all_test_acc
    metric_dict["inception_score_mean"] = all_inception_score_mean
    metric_dict["inception_score_std"] = all_inception_score_std

    import json

    with open(metrics_filename, "w") as f:
        json.dump(metric_dict, f, indent=4)


if __name__ == "__main__":
    main()

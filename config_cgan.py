class Config(object):
    data_dir = "data/cifar-10-batches-py"  # train data and test data dir
    gpu_id = "0"  # gpu id
    train_batch_size = 64
    test_batch_size = 500
    epochs = 200  # train epochs
    count = 500  # number of samples per class
    lr_generator = 0.0001  # learning rate
    lr_discriminator = 0.0001  # learning rate
    seed = 1
    num_classes = 10  # class num
    save_img = True  # save the generated images
    b1 = 0.5
    b2 = 0.999
    latent_dim = 100
    sample_interval = 10

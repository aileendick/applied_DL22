class Config(object):
        data_dir = 'data/cifar-10-batches-py'#train data and test data dir
        save_img = 'save_img/' #save gen imgs dir
        gpu_id = '0' #gpu id
        train_batch_size=128
        test_batch_size= 10#500 
        epochs= 2#1200 # train epochs
        count=400 # number of per class
        lr = 0.0003 #learning rate
        fre_print=1 # print frequency
        seed = 1
        workers=4
        num_classes=10 #class num
        logs = 'logs/' #logs record dir

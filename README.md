# Applied Deep Learning Project
## Aileen Dick(11706782), WS2022



### Project description
This projects aims to deal with **imagine generation** using deep learning architectures. After doing some research, it seems like the already well-known approach for dealing with imagine generation are generative adversarial networks (GANs). Therefore, I will stick to using this approach and take an already existent implementation. Which brings me to the method I want to use: **Bring your own method**. My strategy is to take "Improved Techniques for Training GANs"[1] as a reference and comparison. They have been showing different GAN approaches on different datasets.

By the choice of the topic and project type, I am attempting to use one of the datasets tested in the paper, the CIFAR-10 dataset. The data include 60000 tiny 32x32 color images. The images are labelled with one of 10 mutually exclusive classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. It is publicly available.
Furthermore, as already mentioned, a generative adversarial network will be trained for imagine generation. There are several fine-tuning options for GANs, as well as different implementations of the networks (like CGAN or DCGAN). That will be the main experimental part in the purpose of this project.
The ultimate goal would then be to outperform the results of the approach presented in the mentioned paper.

In the end, also a metric needs to be defined on how the performance of the network should be evaluate.


### Structure
In this project, two generative adversarial networks approaches have been implemented. One is a slightly modified GAN originating from "Improved Techniques for Training GANs"[1].
The second one is a conditional GAN, which allows the network to use actual labels for training as well and learn the difference.

The scripts cgan.py, config_cgan.py, and train_cgan.py implement the second model - saving its outputs in images_cgan, logs_cgan, and metrics_cgan folders.
On the other hand-side, gan.py, config.py, and train.py are specifying the first approach - saving its outputs in images, logs, and metrics folders respectively.


The remaining files are intended to:

+ inception_score.py : implementation of the inception_score
+ load_data.py: script for loading the data
+ loss_evaluation.py: plotting the losses for each model. The model "gan" or "cgan" needs to be specified when executing the script. e.g.:
```
python loss_evaluation.py -model "gan"
```
+ plotting.py: includes some helper functions for visualizing images


### How to run
First make sure, all requirements are already satisfied. If not run:
```
pip install -r requirements.txt
```

Afterwards, either run
```
python train.py
```
for the improved GAN. Or
```
python train_cgan.py
```
for the conditional one.

When running the training script, the losses for each epoch are stored in a json file in the respective metrics folder.  If you want to save the logs, you can do that using the respective log folder.
In addition, there is a develop mode that can be specified as argument, which is by default set to False. Develop mode will just limit the number of epochs.


### Chosen metric and Results
After doing some research, I would like to use the inception score as metric. It is one of the suggested ones for imagine generation, since it judges the quality of the generated images. 
The higher, the higher the quality of outputs. Furthermore, there are basic losses reported of each model.[5] It is implemented in the project according to [4].

Originally, there was a score of around 11 achieved using this cifar dataset. So the ultimate goal would be to exceed this number with my implementation.

However, as first results show, this is hard to achieve. Using the improved GAN with a training loop of 100 epochs, no higher score than 2 could be achieved. It is also visible when looking at the images.
With the CGAN the score even almost never went above 1.

Additionally, also training and test loss are reported to get an idea of the overall performance.

### Experiments
Several different parameters where tested to get the results. Following some ideas from [6], I experimented with learning rates, training batch sizes, the random noise, the slope of leaky relu...
But as it is visible, there were no big successes.

### Further improvememts
One step would be to train for a lot more epochs. The original paper had more than ten times more epoch iteration to get to their results. However, there was no change until now to test that since the dataset is quite big and it takes a really long time running it on the free GPU.

In addition, parameters should be further tuned if possible.
Especially the CGAN is performing really badly. A bigger update of the GAN would be necessary to improve the results. As we can see in the output images, it is not performing well at all, and it is also overfitting the colors, because after 200 epochs it looks like  confetti. 

### Project plan
Below are my time estimations for the individual stages of the project:

+ The ***dataset collection*** should be quick, since I take the one from the paper -> 0.5h
+ ***Designing and building an appropriate network*** includes general research about GANs as well. Therefore, I would calculate 15 hours in order to build a first appropriate network. -> 15h
+ ***Training and fine-tuning that network*** will be the main task of the project -> 15h
+ ***Building an application to present the results*** is very hard to estimate, since I have not done that before -> 15h
+ Considering that all previous steps are finished and results are ready, ***writing the final report*** might take around 8 hours. -> 8h
+ At the end, ***preparing the presentation of your work*** will take at least 2 hours -> 2h


Actual time spent as of Dec 15,2022:

+ **dataset collection***: 0.5h
+ ***Designing and building an appropriate network*** : This step involved a lot of research and reading up on GANs(+pytorch) in the beginning. Including the implementation of the code, I would say -> 25h
+ ***Training and fine-tuning that network***: It depends on what to count here, but leaving out some hours where the model just runs - It took 10h


### Scientific sources
The following scientific papers will be considered within this project. However, the list might be extended as the project progresses.

- [1] Salimans, Tim and Goodfellow, Ian and Zaremba, Wojciech and Cheung, Vicki and Radford, Alec and Chen, Xi. Improved Techniques for Training GANs https://doi.org/10.48550/arXiv.1606.03498
- [2] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, et al. Generative adversarial networks. https://doi.org/10.48550/arXiv.1406.2661
- [3] Brock, Andrew and Lim, Theodore and Ritchie, J. M. and Weston, Nick. Neural Photo Editing with Introspective Adversarial Networks. https://doi.org/10.48550/arXiv.1609.07093
- [4] https://github.com/sbarratt/inception-score-pytorch
- [5] https://www.aiproblog.com/index.php/2019/08/27/how-to-implement-the-inception-score-is-for-evaluating-gans/
- [6] https://machinelearningmastery.com/how-to-train-stable-generative-adversarial-networks/
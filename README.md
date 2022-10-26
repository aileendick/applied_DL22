# Applied Deep Learning Project
## Aileen Dick(11706782), WS2022



### Project description
This projects aims to deal with **imagine generation** using deep learning architectures. After doing some research, it seems like the already well-known approach for dealing with imagine generation are generative adversarial networks (GANs). Therefore, I will stick to using this approach and take an already existet implementation. Which brings me to the method I want to use: **Bring your own method**. My strategy is to take "Improved Techniques for Training GANs"[1] as a reference and comparison. They have been showing different GAN approaches on different datasets.

By the choice of the topic and project type, I am attempting to use one of the datasets tested in the paper, the CIFAR-10 dataset. The data include 60000 tiny 32x32 color images. The images are labelled with one of 10 mutually exclusive classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. It is publicly available.
Furthermore, as already mentioned, a generative adversarial network will be trained for imagine generation. There are several fine-tuning options for GANs, as well as different implemenations of the networks (like CGAN or DCGAN). That will be the main experimental part in the purpose of this project.
The ultimate goal would then be to outperform the results of the approach presented in the mentioned paper.

In the end, also a metric needs to be defined on how the performance of the network should be evaluatde.

### Project plan
Below are my time estimations for the individual stages of the project:

+ The ***dataset collection*** should be quick, since I take the one from the paper -> 0.5h
+ ***Designing and building an appropriate network*** includes general research about GANs as well. Therefore, I would calculate 15 hours in order to build a first appropriate network. -> 15h
+ ***Training and fine-tuning that network*** will be the main task of the project -> 15h
+ ***Building an application to present the results*** is very hard to estimate, since I have not done that before -> 15h
+ Considering that all previous steps are finished and results are ready, ***writing the final report*** might take around 8 hours. -> 8h
+ At the end, ***preparing the presentation of your work*** will take at least 2 hours -> 2h


### Scientific sources
The following scientific papers will be considered within this project. However, the list might be extended as the project progresses.

- [1] Salimans, Tim and Goodfellow, Ian and Zaremba, Wojciech and Cheung, Vicki and Radford, Alec and Chen, Xi. Improved Techniques for Training GANs https://doi.org/10.48550/arXiv.1606.03498
- [2] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, et al. Generative adversarial networks. https://doi.org/10.48550/arXiv.1406.2661
- [3] Brock, Andrew and Lim, Theodore and Ritchie, J. M. and Weston, Nick. Neural Photo Editing with Introspective Adversarial Networks. https://doi.org/10.48550/arXiv.1609.07093


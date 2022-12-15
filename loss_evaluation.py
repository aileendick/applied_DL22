import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import argparse
import sys
import os

parser = argparse.ArgumentParser()
parser.add_argument("-model", type=str)
args = parser.parse_args()

# read data from json

if args.model == "gan":
    filename = "metrics/metrics.json"
elif args.model == "cgan":
    filename = "metrics_cgan/metrics.json"
else:
    sys.exit('No such model argument. model must be either "gan" or "cgan"!')


file = open(filename)
data = json.load(file)
file.close()

# save as pandas dataframe
loss_df = pd.DataFrame.from_dict(data)
loss_df["epoch"] = range(1, len(loss_df) + 1)


plt.style.use("ggplot")

save_to_path = f"result_plots/{args.model}/"
if not os.path.isdir(save_to_path):
    os.makedirs(save_to_path)

# plot loss
plt.plot(loss_df["epoch"], loss_df["accuracy_test"])
plt.title("Test accuracy")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.savefig(f"{save_to_path}/accuracy_test.png")
plt.close()

plt.plot(loss_df["epoch"], loss_df["loss_nolabel"])
plt.title("Loss of unlabeled data")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.savefig(f"{save_to_path}/loss_nolabel.png")
plt.close()

plt.plot(loss_df["epoch"], loss_df["loss_generated"])
plt.title("Loss of generated data")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.savefig(f"{save_to_path}/loss_generated.png")
plt.close()


plt.plot(loss_df["epoch"], loss_df["inception_score_mean"])
plt.title("Inception score of generated data")
plt.xlabel("epoch")
plt.ylabel("score")
plt.savefig(f"{save_to_path}/inception_score_mean.png")
plt.close()

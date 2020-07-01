import matplotlib.pyplot as plt
import numpy as np
import lmlib as lm
import os
from lmlib.utils.generator import load_single_channel, load_multi_channel

# if set True the scripts overrights the images in the doc if the path exist
SAVE_TO_DOC = True

# Get file names
filenames = []
for file in os.listdir("../../lmlib/utils/data"):
    if file.endswith(".csv"):
        filenames.append(file)


# sort 1CH and multi channel
str_single = "_1CH"
filenames_single = []
filenames_multi = []

path = "../../doc/_static/"
filename = "signal_list.txt"

is_path = os.path.isdir(path)

if SAVE_TO_DOC:
    with open(os.path.join(path, filename), "w") as temp_file:
        for L in filenames:
            temp_file.writelines("* ``{}``\n".format(L))


for filename in filenames:
    pos = filename.find(str_single)
    if pos is -1:
        filenames_multi.append(filename)
    else:
        filenames_single.append(filename)

fig, axs = plt.subplots(len(filenames_single), 1, figsize=(8, 10))
for num, filename in enumerate(filenames_single):
    y = load_single_channel(filename, K=-1)
    axs[num].plot(range(len(y)), y, c="k", lw=0.8)
    axs[num].set_title(filename)

plt.subplots_adjust(hspace=0.5)

if is_path and SAVE_TO_DOC:
    print("Save single channel image to doc.")
    plt.savefig("../../doc/_static/singlechannel.png")


plt.show()

fig, axs = plt.subplots(len(filenames_multi), 1, figsize=(8, 20))

for num, filename in enumerate(filenames_multi):
    y = load_multi_channel(filename, K=-1)
    y_offsets = np.arange(y.shape[1])
    axs[num].plot(range(y.shape[0]), y + y_offsets, c="k", lw=0.8)
    axs[num].set_title(filename)

plt.subplots_adjust(hspace=0.5)

if is_path and SAVE_TO_DOC:
    print("Save multi channel images to doc.")
    plt.savefig("../../doc/_static/multichannel.png")


plt.show()

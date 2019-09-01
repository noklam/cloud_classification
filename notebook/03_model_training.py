# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## General information
#
# In this kernel I work with the data from Understanding Clouds from Satellite Images competition.
# ```
# Shallow clouds play a huge role in determining the Earth's climate. They’re also difficult to understand and to represent in climate models. By classifying different types of cloud organization, researchers at Max Planck hope to improve our physical understanding of these clouds, which in turn will help us build better climate models.
# ```
#
# So in this competition we are tasked with multiclass segmentation task: finding 4 different cloud patterns in the images. On the other hand, we make predictions for each pair of image and label separately, so this could be treated as 4 binary segmentation tasks.
# It is important to notice that images (and masks) are `1400 x 2100`, but predicted masks should be `350 x 525`.
#
#
#
# ![](https://i.imgur.com/EOvz5kd.png)

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import os
import cv2
import collections
import time 
import tqdm
from PIL import Image
from functools import partial
train_on_gpu = True

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import TensorDataset, DataLoader,Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR

# %%
from utils import *

# %% [markdown]
# ## Data overview
#
# Let's have a look at the data first.

# %%
path = '../input/'
os.listdir(path)

# %% [markdown]
# We have folders with train and test images, file with train image ids and masks and sample submission.

# %%
# train = pd.read_csv(f'{path}/train.csv')
# train['image'] = train['Image_Label'].apply(lambda x: x.split('_')[0])
# train = train[train['image'].isin(os.listdir('../input/train'))]
# train.to_csv('../input/train_sample.csv', index=False)

# %%

train = pd.read_csv(f'{path}/train_sample.csv')
sub = pd.read_csv(f'{path}/sample_submission.csv')

# %%
train.__len__()

# %%
n_train = len(os.listdir(f'{path}/train'))
n_test = len(os.listdir(f'{path}/test'))
print(f'There are {n_train} images in train dataset')
print(f'There are {n_test} images in test dataset')

# %%
train['Image_Label'].apply(lambda x: x.split('_')[1]).value_counts()

# %% [markdown]
# So we have __~5.5k__ images in train dataset and they can have up to 4 masks: Fish, Flower, Gravel and Sugar.

# %%
train.loc[train['EncodedPixels'].isnull() == False, 'Image_Label'].apply(lambda x: x.split('_')[1]).value_counts()

# %%
train.loc[train['EncodedPixels'].isnull() == False, 'Image_Label'].apply(lambda x: x.split('_')[0]).value_counts().value_counts()

# %% [markdown]
# But there are a lot of empty masks. In fact only 266 images have all four masks. It is important to remember this.

# %%
train['label'] = train['Image_Label'].apply(lambda x: x.split('_')[1])
train['im_id'] = train['Image_Label'].apply(lambda x: x.split('_')[0])


sub['label'] = sub['Image_Label'].apply(lambda x: x.split('_')[1])
sub['im_id'] = sub['Image_Label'].apply(lambda x: x.split('_')[0])

# %% [markdown]
# Let's have a look at the images and the masks.

# %%
train.head()

# %%
sub.head()

# %%
fig = plt.figure(figsize=(25, 16))
for j, im_id in enumerate(np.random.choice(train['im_id'].unique(), 4)):
    for i, (idx, row) in enumerate(train.loc[train['im_id'] == im_id].iterrows()):
        ax = fig.add_subplot(5, 4, j * 4 + i + 1, xticks=[], yticks=[])
        print(f"{path}/train/{row['Image_Label'].split('_')[0]}")
        im = Image.open(f"{path}/train/{row['Image_Label'].split('_')[0]}")
        plt.imshow(im)
        mask_rle = row['EncodedPixels']
        try: # label might not be there!
            mask = rle_decode(mask_rle)
        except:
            mask = np.zeros((1400, 2100))
        plt.imshow(mask, alpha=0.5, cmap='gray')
        ax.set_title(f"Image: {row['Image_Label'].split('_')[0]}. Label: {row['label']}")

# %% [markdown]
# We can see that masks can overlap. Also we can see that clouds are really similar to fish, flower and so on. Another important point: masks are often quite big and can have seemingly empty areas.

# %% [markdown]
# ## Preparing data for modelling
#
# At first, let's create a list of unique image ids and the count of masks for images. This will allow us to make a stratified split based on this count.

# %%
id_mask_count = train.loc[train['EncodedPixels'].isnull() == False, 'Image_Label'].apply(lambda x: x.split('_')[0]).value_counts().\
reset_index().rename(columns={'index': 'img_id', 'Image_Label': 'count'})
train_ids, valid_ids = train_test_split(id_mask_count['img_id'].values, random_state=42, stratify=id_mask_count['count'], test_size=0.1)
test_ids = sub['Image_Label'].apply(lambda x: x.split('_')[0]).drop_duplicates().values

# %% [markdown]
# ## Exploring augmentations with albumentations
#
# One of important things while working with images is choosing good augmentations. There are a lot of them, let's have a look at augmentations from albumentations!

# %%
train_ids[0]

# %%
image_name = train_ids[0]
image = get_img(image_name, path=path)
mask = make_mask(train, image_name)

# %%
visualize(image, mask)

# %% [markdown]
# ## Setting up dataset for training

# %% [markdown]
# Now we define model and training parameters

# %%
from dataset import CloudDataset

# %%
# import segmentation_models_pytorch as smp

# %%
# ENCODER = 'resnet50'
# ENCODER_WEIGHTS = 'imagenet'
# DEVICE = 'cuda'

# ACTIVATION = None
# model = smp.Unet(
#     encoder_name=ENCODER, 
#     encoder_weights=ENCODER_WEIGHTS, 
#     classes=4, 
#     activation=ACTIVATION,
# )
# preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

# %%
num_workers = 0
bs = 4
train_dataset = CloudDataset(path=path, df=train, datatype='train', img_ids=train_ids)
valid_dataset = CloudDataset(path=path, df=train, datatype='valid', img_ids=valid_ids)

train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers)
valid_loader = DataLoader(valid_dataset, batch_size=bs, shuffle=False, num_workers=num_workers)

# loaders = {
#     "train": train_loader,
#     "valid": valid_loader
# }
print()

# %%
from fastai.vision import *

# %%
# train_dataset.x = 0
# valid_dataset.x = 0

# %%
# train = train.iloc[0:40]

# %%
# class MultiMasksList(SegmentationLabelList):
#     def __init__(self, *args, erosion=True, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.erosion = erosion

#     def open(self, fn):
#         mask_files = next(os.walk(fn))[2]
#         mask = open_image(os.path.join(fn, mask_files.pop(0)),
#                           convert_mode='L').px
#         for mask_file in mask_files:
#             mask += open_image(os.path.join(fn, mask_file),
#                                convert_mode='L').px
#         if self.erosion:
#             mask = torch.tensor(
#                 cv2.erode(
#                     mask.numpy().squeeze().astype(np.uint8),
#                     np.ones((3, 3),
#                             np.uint8),
#                     iterations=1)).unsqueeze(0)
#         return Image(mask.float())

#     def analyze_pred(self, pred, thresh: float = 0.5):
#         return (pred > thresh).float()

#     def reconstruct(self, t): return Image(t)

# %%
mask.shape

# %%
b=torch.tensor(mask.transpose(2,0,1))

# %%
ImageSegment(b)

# %%
ImageSegment(b[0,None,::])

# %% [markdown]
# #### b.data.std()

# %%
ImageSegment(torch.tensor(mask.transpose(2,0,1))/2)


# %%
class SegmentationMultiLabelList(SegmentationLabelList):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print('Oh!')
        print(*args, kwargs.keys())
        print(type(*args))
    def open(self, fn):
#         from pdb import set_trace
#         set_trace()
        print('stop!')
        mask = make_mask(train, fn.parts[-1])
        return ImageSegment(torch.tensor(mask.transpose(2,0,1)))


# %%
# a=(SegmentationItemList.from_folder(path) \
#  .split_by_rand_pct())

# a.label_from_func(lambda fn :make_mask(train, fn.parts[-1]))

# %%
a=(SegmentationItemList.from_folder(path) \
 .split_by_rand_pct())
a

# %%
??open_mask


# %%
def make_mask2(fn):
    return make_mask(train,fn)


# %%
(SegmentationItemList.from_folder(path) \
 .split_by_rand_pct().
 label_from_func(open_mask, label_cls=SegmentationMultiLabelList)
)

# %%
(SegmentationItemList.from_folder(path) \
 .split_by_rand_pct().
 label_from_func(lambda x: make_mask(train, x.parts[-1]))
)

# %%
data = ImageDataBunch.create(train_ds=train_dataset, valid_ds=valid_dataset, test_ds=None,
bs=bs, num_workers=num_workers) #tfms = None

data = data.normalize(imagenet_stats)

# (SegmentationItemList.from_folder(path) \
#  .split_by_rand_pct().
#  label_from_df(train)
# )

# %%
data.show_batch(2, figsize=(10,7))

# %% [markdown]
# ## Model training

# %% {"_kg_hide-output": true}
###### Test Only #####
loaders = {
    "train": train_loader,
    "valid": valid_loader
}
######################

runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    loaders=loaders,
    callbacks=[DiceCallback(), EarlyStoppingCallback(patience=5, min_delta=0.001)],
    logdir=logdir,
    num_epochs=num_epochs,
    verbose=True
)

# %%
utils.plot_metrics(
    logdir=logdir, 
    # specify which metrics we want to plot
    metrics=["loss", "dice", 'lr', '_base/lr']
)

# %%
train_dataset[0][0].shape

# %%
x=next(iter(train_loader))

# %%
x[0].size()

# %%
x[1].size()

# %%
x[1].dtype

# %%
x[0].dtype

# %%




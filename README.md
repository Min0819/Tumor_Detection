# DL_final_project

Applied Deep Learning Final Project

---

## Description

The main task is to detect the tumor cells in the Gigapixel Pathology Images. And each slide contains about 10,000,000,000 pixels.

---

### patches

Due to the large size of each slide, we can't read full images into the memory. Therefore we extract patches from slides and make a patch-level image classification.
![normal patches and tumor patches](/images/normal_and_tumor_patches.png)

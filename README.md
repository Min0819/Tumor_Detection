# DL_final_project

Applied Deep Learning Final Project

---

## Description

The main task is to detect the tumor cells in the Gigapixel Pathology Images. And each slide contains about 10,000,000,000 pixels.

---

### Patches

Due to the large size of each slide, we can't read full images into the memory. Therefore we extract patches from slides and make a patch-level image classification.
![normal patches and tumor patches](/images/normal_and_tumor_patches.png)

The slides and the corresponding masks are in ".tif" format, which contains different levels of images. We work on the level 7 and extract patches from level 0.

Each pixel in a level 7 image corresponds to a 128 \* 128 region in level 0. Pixel-level classification in the level 7 patch is actually classifying a 128 \* 128 region in the level 0 with a stride of 128.

We extract a 299 \* 299 patch at the same center of the 128 \* 128 region as the context of that region. And the patch is labeled as tumor if the center 128 \* 128 region contains at least one tumor cell, otherwise the patch is labeled as normal.

Some calculations:
A point (x, y) in level 7 corresponds to a 128 \* 128 region, and the top left coner of that region is (128 \* x, 128 \* y) because of the downsampling of 128. Therefore the center of region is (128 \* x + 64, 128 \* y + 64), and the top left corner of the corresponding context is (128 \* x - 85, 128 \* y - 85).

```python
for x, y in samples:
      if get_patch(mask, x*128, y*128, 128, 128)[:,:,0].any(): # is tumor
        patch = get_patch(slide, max(0, x*128-85), max(0, y*128-85), 299, 299) # extract the context
        img = Image.fromarray(patch, 'RGB')
        img.save('patches/tumor/{}_{}_{}.png'.format(slide_name, x, y))
        _num_tumor += 1
```

---

### Training

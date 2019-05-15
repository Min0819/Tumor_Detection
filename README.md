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

_Some calculations:_
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

The data is highly unbalanced, the number of tumor pathches in slides is far less than the number of normal patches.
![unbalanced_dataset](/images/unbalanced_dataset.png)

> function to find coordinates of all tumor pixels

```python
def find_tumor_pixels(m):
  return [(j, i) for i, row in enumerate(m)
              for j, c in enumerate(row) if c]
```

> function to generate the statistics

```python
print('THRESH = 1/3 loc_tummor, extract THRESH / 3 tumor patches, THRESH / 6 normal patches')
for filename in tumor_slides:

  slide_path, mask_path = 'slides/'+filename, 'slides/'+filename.split('.')[0]+'_mask.tif'
  slide, mask, s, m = _get_helper(slide_path, mask_path)
  loc_tissue, loc_tumor, loc_normal = _get_locs(slide_path, mask_path)
  print('\t{} || tissue {:<7}  ||  tumor  {:>7}  ||  normal  {:>7}'.format(filename, len(loc_tissue), len(loc_tumor), len(loc_normal)))
print('\nevenly sample normal patches from normal slides, balance the tumor patches')
for filename in normal_slides:

  slide_path, mask_path = 'slides/'+filename, 'slides/'+filename.split('.')[0]+'_mask.tif'
  slide, mask, s, m = _get_helper(slide_path, mask_path)
  loc_tissue, loc_tumor, loc_normal = _get_locs(slide_path, mask_path)
  print('\t{} || tissue {:<7}  ||  tumor  {:>7}  ||  normal  {:>7}'.format(filename, len(loc_tissue), len(loc_tumor), len(loc_normal)))
```

#### Training set

Depending on the ratio of turmor patches, we treat a subset of slides as "tumor slides" and the others as "normal slides". Each tumor slide contains 1,000 to 70,000 tumor patches while each normal slide contains less than 100 tumor patches.

```python
tumor_slides = ['tumor_016.tif', 'tumor_031.tif', 'tumor_064.tif',
                'tumor_078.tif', 'tumor_084.tif', 'tumor_091.tif',
                'tumor_094.tif', 'tumor_101.tif', 'tumor_110.tif']
normal_slides = ['tumor_002.tif', 'tumor_012.tif', 'tumor_035.tif',
                 'tumor_057.tif', 'tumor_059.tif', 'tumor_081.tif']
```

_In order to get balanced training set, we applied different sampling strategies on turmor slides and normal slides._

For each tumor slide, we sample about 1 / 3 of all the tumor patches, the sampled size of tumor patches is denoted as `_num_tumor`. And then we sample about half of '`_num_tumor`' normal patches from the same slide.

> sampling from turmor slide

```python
def _extract_tumor_slides(filenames):
  if not os.path.exists('patches'):
    os.mkdir('patches')
  if not os.path.exists('patches/tumor'):
    os.mkdir('patches/tumor')
  if not os.path.exists('patches/normal'):
    os.mkdir('patches/normal')
  slides = ['slides/%s' %filename for filename in filenames]
  masks = [path.split('.')[0]+'_mask.tif' for path in slides]
  for slide_path, mask_path in zip(slides, masks):
    slide_name = slide_path.split('/')[-1].split('.')[0]
    slide, mask, s, m = _get_helper(slide_path, mask_path)
    loc_tissue, loc_tumor, loc_normal = _get_locs(slide_path, mask_path)
    THRESH = len(loc_tumor) // 3
    _num_tumor, _num_normal = 0, 0

    samples = random.sample(loc_tumor, THRESH)

    for x, y in samples:
      if get_patch(mask, x*128, y*128, 128, 128)[:,:,0].any(): # is tumor
        patch = get_patch(slide, max(0, x*128-85), max(0, y*128-85), 299, 299) # extract the context
        img = Image.fromarray(patch, 'RGB')
        img.save('patches/tumor/{}_{}_{}.png'.format(slide_name, x, y))
        _num_tumor += 1

    samples = random.sample(loc_normal, _num_tumor//2)
    for x, y in samples:
      if not get_patch(mask, x*128, y*128, 128, 128)[:,:,0].any(): # is normal
        patch = get_patch(slide, max(0, x*128-85), max(0, y*128-85), 299, 299)
        img = Image.fromarray(patch, 'RGB')
        img.save('patches/normal/{}_{}_{}.png'.format(slide_name, x, y))
        _num_normal += 1
    print('extracted: {:>20} ||  tumor patches {:<7} || normal patches {:<7}'.format(slide_name, _num_tumor, _num_normal))
```

After sampling from the tumor slides, the number of sampled normal patches is about half of the number of sampled tumor patches. In order to get balanced data for each class, we sample evenly from the normal slides for the rest of needed normal patches. (we don't sample tumor patch from normal slides).

> Sampling from normal patches

```python
def _extract_normal_slides(filenames, THRESH):
  if not os.path.exists('patches'):
    os.mkdir('patches')
  if not os.path.exists('patches/tumor'):
    os.mkdir('patches/tumor')
  if not os.path.exists('patches/normal'):
    os.mkdir('patches/normal')
  slides = ['slides/%s' %filename for filename in filenames]
  masks = [path.split('.')[0]+'_mask.tif' for path in slides]
  for slide_path, mask_path in zip(slides, masks):
    slide_name = slide_path.split('/')[-1].split('.')[0]
    slide, mask, s, m = _get_helper(slide_path, mask_path)
    loc_tissue, loc_tumor, loc_normal = _get_locs(slide_path, mask_path)

    _num_normal = 0
    samples = random.sample(loc_normal, min(THRESH, len(loc_normal)))
    for x, y in samples:
      if not get_patch(mask, x*128, y*128, 128, 128)[:,:,0].any(): # is normal
        patch = get_patch(slide, max(0, x*128-85), max(0, y*128-85), 299, 299)
        img = Image.fromarray(patch, 'RGB')
        img.save('patches/normal/{}_{}_{}.png'.format(slide_name, x, y))
        _num_normal += 1
    print('extracted: {:>20} ||  tumor patches {:<7} || normal patches {:<7}'.format(slide_name, 0, _num_normal))
```

_The finished training set_
![training_set](/images/finished_train.png)
The training set contains 113,057 patches. (50% tumor patches and 50% normal patches)

### The model

We trained two models:

1. Pretrained on ImageNet (feature extraction), and based on InceptionV3 model.

   > The model 1

   ```python
   def create_model():
   conv_base = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
   model = models.Sequential()
   model.add(conv_base)
   model.add(layers.Flatten())
   model.add(layers.Dense(128, activation='relu'))
   model.add(layers.Dropout(0.3))
   model.add(layers.Dense(1, activation='sigmoid'))
   conv_base.trainable = False
   model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=2e-5), metrics=['acc'])
   return model
   ```

   > training of model 1

   ```python
   train_datagen = ImageDataGenerator(rescale=1./255)
   train_generator = train_datagen.flow_from_directory('patches', target_size=(299, 299), batch_size=32, class_mode='binary')

   history = model.fit_generator(train_generator, steps_per_epoch=113057//32+1, epochs=5) # takes about 150 min
   ```

2. A model trained from scratch, which is also based on InceptionV3 model. We also appied data augmentaion in this model. (rotation and horizontal flip). The batch size is set to 8 to fit the memory.

   > model 2, trained from scratch

   ```python
   def create_model_nopre():
   conv_base = InceptionV3(weights=None, include_top=False, input_shape=(299, 299, 3))
   model = models.Sequential()
   model.add(conv_base)
   model.add(layers.Flatten())
   model.add(layers.Dense(128, activation='relu'))
   model.add(layers.Dropout(0.3))
   model.add(layers.Dense(1, activation='sigmoid'))
   model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=2e-5), metrics=['acc'])
   return model
   ```

   > training of model 2 with data augmentaion, 4 epochs in total

   ```python
   train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=180,
                                   horizontal_flip=True
                                  )
    # batch size is decreased from 32 to 8, otherwise the session will
    # run out of RAM and crash
    train_generator = train_datagen.flow_from_directory('patches',target_size=(299, 299), batch_size=8, class_mode='binary')

    history = model.fit_generator(train_generator, steps_per_epoch=113101//8+1, epochs=1) # the model is trained a epoch at a time due to the runtime limit of colab.
   ```

### Test

> function to print heatmaps

```python
def _print_test(filename, s_mask):
  slide_path, mask_path = 'slides/'+filename, 'slides/'+filename.split('.')[0]+'_mask.tif'
  slide, mask, s, m = _get_helper(slide_path, mask_path)
  plt.figure(figsize=(20, 20))
  plt.title(filename)
  plt.subplot(1, 2, 1)
  plt.imshow(s)
  plt.imshow(m, cmap='jet', alpha=0.5)
  plt.title('ground truth')
  plt.subplot(1, 2, 2)
  plt.imshow(s)
  plt.imshow(s_mask, cmap='jet', alpha=0.5)
  plt.title('predicted heatmap')
```

> function to plot AUC

```python
def _plot_auc(filename, s_mask):
  slide_path, mask_path = 'slides/'+filename, 'slides/'+filename.split('.')[0]+'_mask.tif'
  slide, mask, s, m = _get_helper(slide_path, mask_path)
  y_true = m.reshape((-1,))
  y_tumor = s_mask.reshape((-1,))
  y_normal = 1 - y_tumor
  y_probas = np.array(list(zip(y_normal, y_tumor)))
  skplt.metrics.plot_roc_curve(y_true, y_probas)
  plt.show()
```

#### Model 1

For the pretrained model (Model 1), we test on 3 slides: `tumor_064.tif`, `tumor_110.tif`, and `tumor_091.tif`. We use AUC as the metric for our models. We also print the predicted heatmap for each test slide.

> tumor_064.tif predicted heatmap using Model 1
> ![tumor_064_heatmap](/images/tumor064heatmap.png)

> compared with the ground truth
> ![compared_064](/images/compared064.png)

> AUC
> ![auc_064](/images/auc064.png)

> tumor_110.tif predicted heatmap using Model 1
> ![tumor_110_heatmap](/images/tumor110heatmap.png)

> compared with the ground truth
> ![compared_110](/images/compared110.png)

> AUC
> ![auc_110](/images/auc110.png)

> tumor_091.tif predicted heatmap using Model 1
> ![tumor_091_heatmap](/images/tumor091heatmap.png)

> compared with the ground truth
> ![compared_091](/images/compared091.png)

> AUC
> ![auc_091](/images/auc091.png)

#### Model 2

For the model which is trained from scratch with data augmentation, we write an end-to-end test script for this model. Due to the size of slides and weights, and the limit of remote repository, we only test one slide `tumor_064.tif` on Model 2.

[DL_final_test_end_to_end](https://github.com/TIANBOQIU/DL_final_project/blob/master/codes/DL_final_test_end_to_end.ipynb)

Result of Model 2

> compared with the ground truth
> ![compared_064](/images/m2compared064.png)

> AUC
> ![auc_064](/images/m2auc064.png)

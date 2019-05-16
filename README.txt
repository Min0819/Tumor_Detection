Group Project
Tianbo Qiu (tq2137)
Min Fu (mf3200)

Github: https://github.com/TIANBOQIU/DL_final_project
codes:
    - DLfinal_TrainTestEvaluate.ipynb 
    - DL_final_data_augmentation.ipynb
    - DL_final_test_end_to_end.ipynb

DLfinal_TrainTestEvaluate.ipynb:
contains the training, test, evaluation of the first model, 
which is pretrained on ImageNet.

DL_final_data_augmentation.ipynb:
We trained another model from scratch without pretrained weights. 
We also used data augmentation in this model in the training.


DL_final_test_end_to_end.ipynb:
is a script that can run end-to-end. We have already pushed weights 
of the second model and one test side (tumor_064.tif) to the github.
In this script, we used lfs tool to pull the weights and test slide
due to the size limitation of GitHub. The weight and test slide is 
stored in another github repository for convenience. The reason is that
we want to bypass the authorization of Google Drive and get the test data.

models:
    - InceptionV3_pre.zip # Model 1, pretrained on ImageNet
    - InceptionV3_nopre_ck4.zip # Model 2, trained from scratch

We use AUC as the metric and test 3 slides on Model 1, and another slide 
on Model 2.

Google Drive link:
https://drive.google.com/drive/folders/1oq6huGYAyPzy5wYrLmKebeXyZBqxVkFb?usp=sharing

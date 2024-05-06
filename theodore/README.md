# [THEODORE dataset](https://www.tu-chemnitz.de/etit/dst/forschung/comp_vision/datasets/theodore/index.php.en)

<b> Please, download `training_25k_1.tfrecord`, `training_25k_4.tfrecord`, `training_25k_sample_4_mask.tfrecord` files to `data` from https://cloud.dst.etit.tu-chemnitz.de/index.php/s/4C4C4CeyRS2RaeB </b>

According to the paper the best object detection results was achieved using 25K dataset. So 25k_1, 25k_4, 25k_mask subsets were downloaded and analyzed. The 1 and 4 have 6250 intersected images.

- 25k_samples_4 - as train and validation for object detection and 25k_4_masks - for segmentation
- takes 2.5 random images from 25k_samples_1 - as holdout
# Object detection V3

This is a 3d iteration of object detection feature:
1. Initial research, not production version.
2. Production versions of 1st stage. Result - YOLOv7 for detection of tables, chairs and monitors.
3. Creating of new object detection dataset. Result - YOLOv8 for detection of tables and chairs. Actually, the dataset has more object types and model can detect more classes, but only this two classes requires for production.

## Folders and files

- database - annotations, IDs, splittings for actual version of the dataset;
- results - ultrlytics training results;
- compare_datasets.ipynb - compares different versions of the dataset;
- convert2yolo.ipynb -  creates YOLO dataset for next training (deprecated);
- create_dataset.ipynb - creates YOLO dataset for next training;
- dataset_util.py - utils methods, used in create_dataset.ipynb;
- group_images.ipynb - splits images on correlated groups;
- merge_objects.ipynb - merge specific objects, e.g. desk, meeting desks, coffee tables to table;
- read_verify_data.ipynb - reads raw XML annotations and image information and convert it to appropriate CSV format;
- train_val_split.ipynb - splits dataset on train and validation;
- train_yolo.py - start ultralytics train session;
- val.ipynb - validate and compare models;
- viz.ipynb - visualize and save detections;

## Typical dataset preparation pipeline

1. read_verify_data.ipynb
2. group_images.ipynb
3. train_val_split.ipynb
4. merge_objects.ipynb
5. create_dataset.ipynb
6. train_yolo.py
7. val.ipynb
8. viz.ipynb

## Data structure

- batch* - folders with images and CVAT annotation in XML file;
- images.cvs - images information with fields: `id,name,width,height,batch,folder`;
- images_grouped.csv - the same as images.cvs, but with additional field `group_id` - correlated group id;
- annotations.csv - preprocessed annotations with fields: `label,occluded,truncated,not_ground,occupied,occupied_by_person,ID,xtl,ytl,xbr,ybr,x,y,img_id,id`;
- train_test_split.json - image IDs for train and validation;
- embeddings.json - image embeddings for similarity calculation.

dataset path: `/mnt/pg-fs01/Current/DB/ObjectDetectionV3`

## Final model

Model name: `v2_batch6_merged_furniture`

Dataset version: V2

Classes:
- chair;
- table;
- laptop_opened;
- monitor;
- keyboard;
- plants;
- cabinet.

Confidence threshold: 0.4
Iou threshold: 0.5

Metric on val dataset:
```
        Class  Images  Instances  Box(P      R     F1  mAP50  mAP50-95) Pose(P      R     F1  mAP50  mAP50-95): 
          all     292       6055  0.686  0.416  0.513  0.559      0.329   0.77  0.466  0.575  0.627      0.618
        chair     285       2670  0.826  0.621  0.709  0.746      0.424  0.905  0.681  0.777  0.811      0.802
        table     284       1188   0.84  0.602  0.701  0.742      0.452  0.931  0.667  0.777  0.815      0.809
laptop_opened     207        664  0.621  0.452  0.523   0.55      0.298   0.72  0.524  0.607  0.638      0.616
      monitor     167        553  0.624  0.336  0.437  0.492      0.291  0.711  0.383  0.498  0.565      0.553
     keyboard      97        299  0.635  0.244  0.353  0.448      0.267  0.722  0.278  0.401  0.502      0.496
       plants     164        313  0.625  0.319  0.423  0.456      0.267  0.738  0.377  0.499  0.558      0.553
      cabinet     217        368  0.631   0.34  0.442   0.48      0.301  0.662  0.356  0.463  0.504      0.499
```
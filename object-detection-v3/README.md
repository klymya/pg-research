# Object detection V3

This is a 3d iteration of object detection feature:
1. Initial research, not production version.
2. Production versions of 1st stage. Result - YOLOv7 for detection of tables, chairs and monitors.
3. Creating of new object detection dataset. Result - YOLOv8 for detection of tables and chairs. Actually, the dataset has more object types and model can detect more classes, but only this two classes requires for production.

## Folders and files

- database - annotations, IDs, splittings for actual version of the dataset;
- results - ultrlytics training results;
- v3_1 - scripts for dataset v3.1: tables have additional 4 dots for corners;
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

Model name: `v3_2_table_chair_table_top_v2_no_occupied_chair`

Dataset version: V2

Classes:
- chair;
- table.

Confidence threshold: 0.35
Iou threshold: 0.45

Metrics on val dataset:

```
Class  Images  Instances  Box(P      R     F1  mAP50  mAP50-95)  Pose(P      R     F1  mAP50  mAP50-95):
  all     424       5093  0.882  0.737  0.803  0.827      0.627    0.92  0.769  0.838   0.86      0.856
chair     392       2784   0.87  0.705  0.779  0.804      0.563   0.898  0.728  0.804  0.828      0.823
table     413       2309  0.893  0.769  0.827   0.85       0.69   0.942  0.811  0.872  0.893      0.889
```

Metrics on holdout dataset (PG ooffice):

```
Class  Images  Instances  Box(P      R     F1  mAP50  mAP50-95)  Pose(P      R     F1  mAP50  mAP50-95):
  all      36        271   0.94  0.991  0.965   0.98      0.824    0.94  0.991  0.965  0.989      0.985
chair      35        173  0.919  0.982  0.949  0.976      0.787   0.919  0.983  0.950  0.984      0.979
table      35         98  0.961      1  0.980  0.984      0.862    0.96      1  0.980  0.994      0.991
```


## Prev final model

### YOLOv8s-pose

Model name: `v2_batch6_merged_furniture_table_chair_model_s`

Dataset version: V2

Classes:
- chair;
- table.

Confidence threshold: 0.35
Iou threshold: 0.45

Metrics on val dataset:

RGB images:
```
Class  Images  Instances  Box(P      R     F1  mAP50  mAP50-95)  Pose(P      R  F1     mAP50  mAP50-95):
  all     292       3858  0.905  0.722  0.803  0.823      0.529   0.957  0.762  0.848   0.87      0.867
chair     285       2670  0.913  0.755  0.827  0.849      0.507   0.955   0.79  0.865  0.884      0.882
table     284       1188  0.897  0.688  0.779  0.797      0.551   0.958  0.735  0.832  0.856      0.852
```

Grayscale images:
```
Class  Images  Instances      Box(P      R     F1  mAP50  mAP50-95)  Pose(P      R     F1  mAP50  mAP50-95):
  all     292       3858      0.903  0.717  0.799   0.82      0.525   0.954  0.757  0.844  0.867      0.864
chair     285       2670      0.911  0.758  0.827   0.85      0.506   0.951  0.791  0.863  0.883      0.881
table     284       1188      0.894  0.676  0.770   0.79      0.544   0.958  0.724  0.825  0.851      0.847
```

Metrics on holdout dataset (PG ooffice)*:

```
Class  Images  Instances  Box(P      R     F1 mAP50  mAP50-95)  Pose(P      R     F1  mAP50  mAP50-95):
  all      36        271   0.88  0.856  0.867  0.85      0.632   0.919  0.906  0.911  0.912      0.908
chair      35        173  0.856  0.896  0.876 0.868      0.721    0.86  0.925  0.891  0.887       0.88
table      35         98  0.904  0.816  0.858 0.833      0.542   0.978  0.888  0.931  0.938      0.935
```

* duing the model training holdout set was part of train and val
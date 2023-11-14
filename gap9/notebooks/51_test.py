import fiftyone as fo


# A name for the dataset
name = "pg_test"

try:
     fo.delete_dataset(name)
except ValueError:
    pass

# The directory containing the dataset to import
dataset_dir = "../YOLOX/datasets/pg_coco"

# The type of the dataset being imported
dataset_type = fo.types.COCODetectionDataset  # for example


dataset = fo.Dataset.from_dir(
    dataset_dir=dataset_dir,
    dataset_type=dataset_type,
    name=name,
    data_path="test2017",
    labels_path="annotations/instances_test.json"
)

session = fo.launch_app(dataset)
session.wait()

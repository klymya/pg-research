# %%
import os
import shutil

from tqdm.contrib.concurrent import thread_map

from splitter import create_project_file

# %%
# create dataset folder
# copy all laabels an images from the baseline datase if index is suatable
# create new project yaml file

# %%
DATABASE_BASE_PTH = "/home/klymy/data/people_detection"

BASE_DATASET_PTH = os.path.join(DATABASE_BASE_PTH, "baseline")
DATASET_PTH = os.path.join(DATABASE_BASE_PTH, "baseline_10sec")

# %%
def recreate_folder_structure(path_a: str, path_b: str):
    mapping = []
    # Step 3: Walk through the directory tree and recreate folders in path_b
    for dirpath, _, _ in os.walk(path_a):
        # Calculate the relative path of the current directory with respect to path_a
        rel_path = os.path.relpath(dirpath, path_a)
        # Combine this relative path with path_b to create the corresponding target directory
        target_dir = os.path.join(path_b, rel_path)
        
        # Create the target directory if it doesn't exist
        os.makedirs(target_dir, exist_ok=True)
        print(f"Created directory: {target_dir}")
        mapping.append((
            os.path.join(path_a, rel_path),
            target_dir
        ))
    
    return mapping

# %%
# Example usage
folder_mapping = recreate_folder_structure(BASE_DATASET_PTH, DATASET_PTH)

# %%
original_file_names = []
target_file_names = []
for folde_pairs in folder_mapping:
    tmp_file_names = sorted([i for i in os.listdir(folde_pairs[0]) if i.endswith(".jpg") or i.endswith(".txt")])
    tmp_file_names = tmp_file_names[::2]
    original_file_names += [os.path.join(folde_pairs[0], i) for i in tmp_file_names]
    target_file_names += [os.path.join(folde_pairs[1], i) for i in tmp_file_names]

pairs = list(zip(original_file_names, target_file_names))
len(pairs)

# %%
def copy_file(pair: tuple[str, str]) -> None:
    shutil.copyfile(*pair)

# %%
thread_map(
    copy_file,
    pairs,
    max_workers=8
)

# %%
# create project file
create_project_file(os.path.join(DATASET_PTH, "project.yaml"), DATASET_PTH, {0: "person"}, [1, 2])
create_project_file(
    os.path.join(DATASET_PTH, "project_holdout.yaml"), DATASET_PTH, {0: "person"}, [1, 2], "images/holdout")
create_project_file(
    os.path.join(DATASET_PTH, "project_algotest.yaml"), DATASET_PTH, {0: "person"}, [1, 2], "images/algotest")

# %%

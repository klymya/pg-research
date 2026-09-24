# researches

The repo stores the results and code of different research.

## Requirements / first-time setup

This repo uses **Git LFS** and **nbstripout**. Both are per-clone: cloning alone does not
enable them, and skipping either will corrupt what you commit.

### 1. Git LFS (required)

Most binary content -- model weights, datasets, images, TIFF captures, TensorBoard logs --
is stored in LFS. Without it you get 130-byte pointer files instead of real data.

```bash
sudo apt install git-lfs      # or: brew install git-lfs
git lfs install               # once per machine
git lfs pull                  # fetch the binaries for your checkout
```

To clone without downloading ~15 GB of binaries up front, then fetch only what you need:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone <url>
cd researches
git lfs pull --include="background_subtraction/data/**"
```

### 2. nbstripout (required before committing notebooks)

Notebooks in *actively developed* projects are committed **without outputs**; embedded plots
and images are megabytes of base64 that change on every re-run. `.gitattributes` decides
which notebooks are stripped -- you only need to install the filter:

```bash
pip install nbstripout
nbstripout --install --attributes .gitattributes
```

Verify it took effect:

```bash
git check-attr filter -- people_detection_pg2/reduce_quality.ipynb   # -> filter: nbstripout
```

**Your working copy keeps its outputs** -- only the committed version is stripped. If you
skip this step, Git will happily commit multi-megabyte notebooks and the filter rules will
appear not to work.

Notebooks whose outputs **are** the result (evaluations, comparisons, visualisations,
experiments, reports) and every notebook in an archived project are deliberately exempt --
their plots are the research record. See the rules at the bottom of `.gitattributes`.

---

## Projects

- data-exploration - explaration analysis of historical datasets for people detection.
- gap9
- object-detection - object (table, chairs, laptop) detection research, vol 2.
- orbbec-3d-research - videao and depth map collectiong from orbbec 3D camera.
- sagemaker - sagemaker usage investigation.
- Socionext - 3d camera investigation
- vayyar - radar usage investigation. vol 2
- yolo-inference - yolov5 inference on C++ using tf lite and opencv+onnx. The experiment uses yolov5n model.
- yolov7-model - yolov7 reparametrization and onnx export scripts.
- leg_position_detectors - neural networks which detect leg position.
- leg_position_calculator - linear regression model which calculated leg position by bbox coordinates and aspect ratio.
# sagemaker

Follow-along examples for running training jobs on AWS SageMaker.

**Status:** ARCHIVED — frozen 2023-06 · 5 files, 184 KB. **No PointGrab content.**

## Purpose

An evaluation of whether to move model training to SageMaker instead of local GPUs. The
notebooks are worked examples from a tutorial video, not PG experiments: MNIST in TensorFlow
and in PyTorch, each shown running locally and on an AWS GPU instance.

SageMaker was not adopted — training stayed on local GPUs, which is what
`people_detection_pg2/run_training_experiments_batch.py` schedules across today.

| File | What |
|---|---|
| `tf_example.ipynb`, `mnist_keras_tf20_eager.py` | TensorFlow path |
| `torch_example.ipynb`, `mnist_torch.py` | PyTorch path |

## Connections

None. Nothing imports it; it references no PG data or code.

## Dependencies

`tensorflow`, `keras`, `keras_metrics`, `torch`, `torchvision`, `numpy`, plus an AWS account
and the SageMaker SDK. `tfdata/`, `tfmodel/`, `torchdata/`, `torchmodel/` are gitignored.

---

## Original notes

#SageMaker examples

The examples based on [video](https://www.youtube.com/watch?v=K3ngZKF31mc)

The folder contains TensorFlow and PyTorch examples. Each notebook shows how to run Sagemaker locally and using an AWS GPU instance.

AWS sagemaker instances links:
- [Amazon SageMaker Pricing](https://aws.amazon.com/sagemaker/pricing/)
- [Available Studio Instance Types](https://github.com/awsdocs/amazon-sagemaker-developer-guide/blob/master/doc_source/notebooks-available-instance-types.md)
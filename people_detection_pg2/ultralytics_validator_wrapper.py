import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics.models.yolo.pose import PoseValidator
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import TQDM


def print_metrics(metrics, conf_th=0, draw=True):
    data = [
        {
            "task": "box",
            "F1": metrics.box.f1[0],
            "P": metrics.box.p[0],
            "R": metrics.box.r[0],
            "MAP@50": metrics.box.map50,
        },
        {
            "task": "pose",
            "F1": metrics.pose.f1[0],
            "P": metrics.pose.p[0],
            "R": metrics.pose.r[0],
            "MAP@50": metrics.pose.map50,
        }
    ]
    print(pd.DataFrame(data))

    if not draw:
        return
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    x = np.linspace(conf_th, 1, num=len(metrics.box.f1_curve[0]), endpoint=True)
    plt.plot(x, metrics.box.f1_curve[0], label="F1")
    plt.plot(x, metrics.box.r_curve[0], label="R")
    plt.plot(x, metrics.box.p_curve[0], label="P")
    plt.grid(True)
    plt.ylim(-0.1, 1.1)
    plt.xlabel("Confidence")
    plt.legend()
    plt.title("BOX")

    plt.subplot(1, 2, 2)
    x = np.linspace(conf_th, 1, num=len(metrics.pose.f1_curve[0]), endpoint=True)
    plt.plot(x, metrics.pose.f1_curve[0], label="F1")
    plt.plot(x, metrics.pose.r_curve[0], label="R")
    plt.plot(x, metrics.pose.p_curve[0], label="P")
    plt.grid(True)
    plt.ylim(-0.1, 1.1)
    plt.xlabel("Confidence")
    plt.legend()
    plt.title("POSE");


def validate_model(args, ultralytics_model, predict_f, draw=True):
    validator = PoseValidator(args=args)
    validator.data = check_det_dataset(args["data"])

    # model.eval()

    validator.device = args["device"][0]
    validator.stride = 32
    dataloader = validator.get_dataloader(validator.data.get(validator.args.split), validator.args.batch)

    bar = TQDM(dataloader, desc="tst validation", total=len(dataloader))
    validator.init_metrics(ultralytics_model.model)
    for _, batch in enumerate(bar):
        batch = validator.preprocess(batch)

        preds = predict_f(batch)
        validator.update_metrics(preds, batch)

    stats = validator.get_stats()
    validator.check_stats(stats)
    validator.finalize_metrics()
    validator.print_results()

    print_metrics(validator.metrics, args["conf"], draw)


def predict_ultralytics(model, args, batch):
    model_output = model(batch["img"], conf=args["conf"], iou=args["iou"], task=args["task"], verbose=False)
    preds = []
    for i in model_output:
        if i.keypoints.data.shape[1] == 1:
            keypoints = torch.squeeze(i.keypoints.xy, 1)
        else:
            keypoints = torch.squeeze(i.keypoints.xy, 0)
        preds.append(torch.concat([
            torch.concat([i.boxes.xyxy, i.boxes.conf[:, None], i.boxes.cls[:, None]], 1),
            keypoints,
        ], 1).to(args["device"][0]))
    
    return preds


def predict_tflite(model, args, batch):
    preds = []
    for i in batch["img"]:
        input_data = i.detach().cpu().numpy()
        input_data = (np.transpose(input_data, (1, 2, 0)) * 255).astype(np.uint8)
        dets = model.infer(input_data[None]).astype(np.float32)
        if dets.shape[0] == 0:
            dets = np.empty((0, 8), dtype=np.float32)
        else:
            dets[:, 2:4] += dets[:, :2]
        preds.append(torch.from_numpy(dets).to(args["device"][0]))
    
    return preds
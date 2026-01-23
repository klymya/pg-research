python run_training_pipeline.py \
    --dataset /home/klymy/data/cam4_revB/yolo_dataset/batch1_10_margin_15_bg_wb_alif_clahe \
    --exp_name cam3_revB_batch1_10_margin_15_yolov8n_wb_alif_clahe_box_01_dfl_01_2nd \
    --weights /home/klymy/projects/researches/people_detection_pg2/results/filtered_v3_margin_15_yolov8n/weights/best.pt \
    --cfg yolov8-pose.yaml \
    --device 0 \
    --yolo_kwargs "{'box': 0.1, 'dfl': 0.1}" \
    --max_ncalib_imgs 300 \
    --converted-output-folder-name best_non_separated_saved_model

###
python run_training_pipeline.py \
    --dataset /home/klymy/data/cam4_revB/yolo_dataset/batch1_10_margin_15_bg_wb_alif_clahe \
    --exp_name cam3_revB_batch1_10_margin_15_wb_alif_clahe_box_01_dfl_01_2nd \
    --weights /home/klymy/projects/researches/people_detection_pg2/results/filtered_v3_margin_15/weights/best.pt \
    --cfg yolov8-pose-tiny.yaml \
    --device 1 \
    --yolo_kwargs "{'box': 0.1, 'dfl': 0.1}" \
    --max_ncalib_imgs 300 \
    --converted-output-folder-name best_non_separated_saved_model

###
python run_training_pipeline.py \
    --dataset /home/klymy/data/cam4_revB/yolo_dataset/batch1_10_margin_15_bg_wb_alif \
    --exp_name cam3_revB_batch1_10_margin_15_wb_alif_box_01_dfl_01_2nd \
    --weights /home/klymy/projects/researches/people_detection_pg2/results/filtered_v3_margin_15/weights/best.pt \
    --cfg yolov8-pose-tiny.yaml \
    --device 2 \
    --yolo_kwargs "{'box': 0.1, 'dfl': 0.1}" \
    --max_ncalib_imgs 300 \
    --converted-output-folder-name best_non_separated_saved_model


####################

python run_training_pipeline.py \
    --dataset /home/klymy/data/cam4_revB/yolo_dataset/batch1_10_margin_15_bg_wb_alif_clahe \
    --exp_name cam3_revB_batch1_10_margin_15_bg_yolov8n_wb_alif_clahe_box_01_dfl_01_2nd \
    --weights /home/klymy/projects/researches/people_detection_pg2/results/filtered_v3_margin_15_yolov8n/weights/best.pt \
    --cfg yolov8-pose.yaml \
    --device 0 \
    --yolo_kwargs "{'box': 0.1, 'dfl': 0.1}" \
    --max_ncalib_imgs 300 \
    --converted-output-folder-name best_non_separated_saved_model

###
python run_training_pipeline.py \
    --dataset /home/klymy/data/cam4_revB/yolo_dataset/batch1_10_margin_15_bg_wb_alif_clahe \
    --exp_name cam3_revB_batch1_10_margin_15_bg_wb_alif_clahe_box_01_dfl_01_2nd \
    --weights /home/klymy/projects/researches/people_detection_pg2/results/filtered_v3_margin_15/weights/best.pt \
    --cfg yolov8-pose-tiny.yaml \
    --device 1 \
    --yolo_kwargs "{'box': 0.1, 'dfl': 0.1}" \
    --max_ncalib_imgs 300 \
    --converted-output-folder-name best_non_separated_saved_model

####

python run_training_pipeline.py \
    --dataset /home/klymy/data/cam4_revB/yolo_dataset/batch1_10_margin_15_bg_wb_alif_clahe \
    --exp_name cam3_revB_batch1_10_margin_15_rq_bg_wb_alif_clahe_box_01_dfl_01_2nd \
    --weights /home/klymy/projects/researches/people_detection_pg2/results/filtered_v3_margin_15_reduced_quality/weights/best.pt \
    --cfg yolov8-pose-tiny.yaml \
    --device 1 \
    --yolo_kwargs "{'box': 0.1, 'dfl': 0.1}" \
    --max_ncalib_imgs 300 \
    --converted-output-folder-name best_non_separated_saved_model


python run_training_pipeline.py \
    --dataset /home/klymy/data/cam4_revB/yolo_dataset/batch1_10_margin_15_bg_wb_alif_clahe \
    --exp_name cam3_revB_batch1_10_margin_15_rq_bg_yolov8n_wb_alif_clahe_box_01_dfl_01_2nd \
    --weights /home/klymy/projects/researches/people_detection_pg2/results/filtered_v3_margin_15_reduced_quality/weights/best.pt \
    --cfg yolov8-pose.yaml \
    --device 2 \
    --yolo_kwargs "{'box': 0.1, 'dfl': 0.1}" \
    --max_ncalib_imgs 300 \
    --converted-output-folder-name best_non_separated_saved_model

###

python run_training_pipeline.py \
    --dataset /home/klymy/data/cam4_revB/yolo_dataset/batch1_10_margin_15_bg_wb_alif \
    --exp_name cam3_revB_batch1_10_margin_15_rq_bg_yolov8n_wb_alif_box_01_dfl_01_2nd \
    --weights /home/klymy/projects/researches/people_detection_pg2/results/filtered_v3_margin_15_reduced_quality/weights/best.pt \
    --cfg yolov8-pose.yaml \
    --device 2 \
    --yolo_kwargs "{'box': 0.1, 'dfl': 0.1}" \
    --max_ncalib_imgs 300 \
    --converted-output-folder-name best_non_separated_saved_model


python run_training_pipeline.py \
    --dataset /home/klymy/data/cam4_revB/yolo_dataset/batch1_10_margin_15_bg_wb_alif \
    --exp_name cam3_revB_batch1_10_margin_15_rq_bg_wb_alif_box_01_dfl_01_2nd \
    --weights /home/klymy/projects/researches/people_detection_pg2/results/filtered_v3_margin_15_reduced_quality/weights/best.pt \
    --cfg yolov8-pose-tiny.yaml \
    --device 1 \
    --yolo_kwargs "{'box': 0.1, 'dfl': 0.1}" \
    --max_ncalib_imgs 300 \
    --converted-output-folder-name best_non_separated_saved_model

#################################################################################
python run_training_pipeline.py \
    --dataset /home/klymy/data/cam4_revB/yolo_dataset/batch1_11_margin_15_bg_wb_alif_clahe \
    --exp_name cam3_revB_batch1_11_margin_15_rq_wb_alif_clahe_box_01_dfl_01_2nd \
    --weights /home/klymy/projects/researches/people_detection_pg2/results/filtered_v3_margin_15_reduced_quality/weights/best.pt \
    --cfg yolov8-pose-tiny.yaml \
    --device 1 \
    --yolo_kwargs "{'box': 0.1, 'dfl': 0.1}" \
    --max_ncalib_imgs 300 \
    --converted-output-folder-name best_non_separated_saved_model

python run_training_pipeline.py \
    --dataset /home/klymy/data/cam4_revB/yolo_dataset/batch1_10_margin_15_bg_wb_alif_clahe \
    --exp_name cam3_revB_batch1_11_margin_15_wb_alif_clahe_box_01_dfl_01_2nd \
    --weights /home/klymy/projects/researches/people_detection_pg2/results/filtered_v3_margin_15/weights/best.pt \
    --cfg yolov8-pose-tiny.yaml \
    --device 2 \
    --yolo_kwargs "{'box': 0.1, 'dfl': 0.1}" \
    --max_ncalib_imgs 300 \
    --converted-output-folder-name best_non_separated_saved_model

###
python run_training_pipeline.py \
    --dataset /home/klymy/data/cam4_revB/yolo_dataset/batch1_11_margin_15_bg_wb_alif \
    --exp_name cam3_revB_batch1_11_margin_15_rq_wb_box_01_dfl_01_2nd \
    --weights /home/klymy/projects/researches/people_detection_pg2/results/filtered_v3_margin_15_reduced_quality/weights/best.pt \
    --cfg yolov8-pose-tiny.yaml \
    --device 0 \
    --yolo_kwargs "{'box': 0.1, 'dfl': 0.1}" \
    --max_ncalib_imgs 300 \
    --converted-output-folder-name best_non_separated_saved_model

python run_training_pipeline.py \
    --dataset /home/klymy/data/cam4_revB/yolo_dataset/batch1_10_margin_15_bg_wb_alif \
    --exp_name cam3_revB_batch1_11_margin_15_wb_alif_box_01_dfl_01_2nd \
    --weights /home/klymy/projects/researches/people_detection_pg2/results/filtered_v3_margin_15/weights/best.pt \
    --cfg yolov8-pose-tiny.yaml \
    --device 1 \
    --yolo_kwargs "{'box': 0.1, 'dfl': 0.1}" \
    --max_ncalib_imgs 300 \
    --converted-output-folder-name best_non_separated_saved_model

###

python run_training_pipeline.py \
    --dataset /home/klymy/data/cam4_revB/yolo_dataset/batch1_11_margin_15_bg_wb_alif_clahe \
    --exp_name cam3_revB_batch1_11_margin_15_yolov8n_wb_alif_clahe_box_01_dfl_01_2nd \
    --weights /home/klymy/projects/researches/people_detection_pg2/results/filtered_v3_margin_15/weights/best.pt \
    --cfg yolov8-pose.yaml \
    --device 1 \
    --yolo_kwargs "{'box': 0.1, 'dfl': 0.1}" \
    --max_ncalib_imgs 300 \
    --converted-output-folder-name best_non_separated_saved_model

python run_training_pipeline.py \
    --dataset /home/klymy/data/cam4_revB/yolo_dataset/batch1_11_margin_15_bg_wb_alif \
    --exp_name cam3_revB_batch1_11_margin_15_yolov8n_wb_alif_box_01_dfl_01_2nd \
    --weights /home/klymy/projects/researches/people_detection_pg2/results/filtered_v3_margin_15/weights/best.pt \
    --cfg yolov8-pose.yaml \
    --device 2 \
    --yolo_kwargs "{'box': 0.1, 'dfl': 0.1}" \
    --max_ncalib_imgs 300 \
    --converted-output-folder-name best_non_separated_saved_model

#############################################################################

python run_training_pipeline.py \
    --dataset /home/klymy/data/cam4_revB/yolo_dataset/batch1_12_margin_15_bg_wb_alif_clahe \
    --exp_name cam3_revB_batch1_12_margin_15_rq_wb_alif_clahe_box_01_dfl_01_2nd \
    --weights /home/klymy/projects/researches/people_detection_pg2/results/filtered_v3_margin_15_reduced_quality/weights/best.pt \
    --cfg yolov8-pose-tiny.yaml \
    --device 1 \
    --yolo_kwargs "{'box': 0.1, 'dfl': 0.1}" \
    --max_ncalib_imgs 300 \
    --converted-output-folder-name best_non_separated_saved_model

python run_training_pipeline.py \
    --dataset /home/klymy/data/cam4_revB/yolo_dataset/batch1_12_margin_15_bg_wb_alif_clahe \
    --exp_name cam3_revB_batch1_12_margin_15_wb_alif_clahe_box_01_dfl_01_2nd \
    --weights /home/klymy/projects/researches/people_detection_pg2/results/filtered_v3_margin_15/weights/best.pt \
    --cfg yolov8-pose-tiny.yaml \
    --device 2 \
    --yolo_kwargs "{'box': 0.1, 'dfl': 0.1}" \
    --max_ncalib_imgs 300 \
    --converted-output-folder-name best_non_separated_saved_model

###
python run_training_pipeline.py \
    --dataset /home/klymy/data/cam4_revB/yolo_dataset/batch1_12_margin_15_bg_wb_alif \
    --exp_name cam3_revB_batch1_12_margin_15_rq_wb_box_01_dfl_01_2nd \
    --weights /home/klymy/projects/researches/people_detection_pg2/results/filtered_v3_margin_15_reduced_quality/weights/best.pt \
    --cfg yolov8-pose-tiny.yaml \
    --device 0 \
    --yolo_kwargs "{'box': 0.1, 'dfl': 0.1}" \
    --max_ncalib_imgs 300 \
    --converted-output-folder-name best_non_separated_saved_model

python run_training_pipeline.py \
    --dataset /home/klymy/data/cam4_revB/yolo_dataset/batch1_12_margin_15_bg_wb_alif \
    --exp_name cam3_revB_batch1_12_margin_15_wb_alif_box_01_dfl_01_2nd \
    --weights /home/klymy/projects/researches/people_detection_pg2/results/filtered_v3_margin_15/weights/best.pt \
    --cfg yolov8-pose-tiny.yaml \
    --device 1 \
    --yolo_kwargs "{'box': 0.1, 'dfl': 0.1}" \
    --max_ncalib_imgs 300 \
    --converted-output-folder-name best_non_separated_saved_model

#########

python run_training_pipeline.py \
    --dataset /home/klymy/data/cam4_revB/yolo_dataset/batch1_12_margin_15_bg_wb_alif_clahe \
    --exp_name cam3_revB_batch1_12_margin_15_rq_yolov8n_wb_alif_clahe_box_01_dfl_01_2nd \
    --weights /home/klymy/projects/researches/people_detection_pg2/results/filtered_v3_margin_15_reduced_quality/weights/best.pt \
    --cfg yolov8-pose.yaml \
    --device 2 \
    --yolo_kwargs "{'box': 0.1, 'dfl': 0.1}" \
    --max_ncalib_imgs 300 \
    --converted-output-folder-name best_non_separated_saved_model

python run_training_pipeline.py \
    --dataset /home/klymy/data/cam4_revB/yolo_dataset/batch1_12_margin_15_bg_wb_alif_clahe \
    --exp_name cam3_revB_batch1_12_margin_15_yolov8n_wb_alif_clahe_box_01_dfl_01_2nd \
    --weights /home/klymy/projects/researches/people_detection_pg2/results/filtered_v3_margin_15/weights/best.pt \
    --cfg yolov8-pose.yaml \
    --device 0 \
    --yolo_kwargs "{'box': 0.1, 'dfl': 0.1}" \
    --max_ncalib_imgs 300 \
    --converted-output-folder-name best_non_separated_saved_model


python run_training_pipeline.py \
    --dataset /home/klymy/data/cam4_revB/yolo_dataset/batch1_12_margin_15_bg_wb_alif \
    --exp_name cam3_revB_batch1_12_margin_15_yolov8n_wb_alif_box_01_dfl_01_2nd \
    --weights /home/klymy/projects/researches/people_detection_pg2/results/filtered_v3_margin_15/weights/best.pt \
    --cfg yolov8-pose.yaml \
    --device 1 \
    --yolo_kwargs "{'box': 0.1, 'dfl': 0.1}" \
    --max_ncalib_imgs 300 \
    --converted-output-folder-name best_non_separated_saved_model


python run_training_pipeline.py \
    --dataset /home/klymy/data/cam4_revB/yolo_dataset/batch1_12_margin_15_bg_wb_alif \
    --exp_name cam3_revB_batch1_12_margin_15_rq_yolov8n_wb_alif_box_01_dfl_01_2nd \
    --weights /home/klymy/projects/researches/people_detection_pg2/results/filtered_v3_margin_15_reduced_quality/weights/best.pt \
    --cfg yolov8-pose.yaml \
    --device 1 \
    --yolo_kwargs "{'box': 0.1, 'dfl': 0.1}" \
    --max_ncalib_imgs 300 \
    --converted-output-folder-name best_non_separated_saved_model


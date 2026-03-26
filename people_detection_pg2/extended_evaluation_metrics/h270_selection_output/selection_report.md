# H270 Selection Report

## Summary
- Eligible holdout frames: 699
- Selected train frames: 60
- Selected val frames: 30
- Train groups: [0, 1, 2, 3, 4, 5, 7]
- Val groups: [6, 8]
- Validation passed: True
- Precision/recall inconsistencies: 1
- Dropped inconsistent frames: 1
- GT count mismatches (metadata vs eval): 189
- Authoritative GT source for scoring: metrics_per_gt.csv
- Ranked output sort: usefulness_score desc, then group_id/frame_index/img_id.

## Score Composition
- weight_f1=0.35
- weight_fn=0.35
- weight_fp=0.15
- weight_loc=0.1
- weight_conf=0.05
- pr_tolerance=1e-06

## Leakage Safeguards Applied
- Selection was restricted to current holdout frames only.
- Train/val assignment was performed strictly at group_id level.
- No group appears in both train and val.
- Original H270 dataset was not modified unless --materialize-dataset was explicitly enabled.

## Top Failure Slices
- posture=Sitting Leaning Backward: frames=1, mean_f1=0.6667, mean_score=0.3717
- posture=Standing: frames=593, mean_f1=0.7014, mean_score=0.2988
- posture=Sitting Regularly: frames=88, mean_f1=0.7287, mean_score=0.2738
- posture=Sitting Leaning Forward: frames=17, mean_f1=0.8196, mean_score=0.1935

## Group Allocation
- group 0: partition=train, frames=1, score_mean=0.3717
- group 1: partition=train, frames=15, score_mean=0.4761
- group 2: partition=train, frames=6, score_mean=0.4826
- group 3: partition=train, frames=11, score_mean=0.4073
- group 4: partition=train, frames=11, score_mean=0.4342
- group 5: partition=train, frames=9, score_mean=0.4314
- group 6: partition=val, frames=15, score_mean=0.5614
- group 7: partition=train, frames=7, score_mean=0.5426
- group 8: partition=val, frames=15, score_mean=0.5848

## Exclusion Summary
- Non-selected entries with explicit exclusion reason tracked: 609

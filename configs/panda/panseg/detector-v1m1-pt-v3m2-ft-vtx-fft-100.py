_base_ = [
    "./detector-v1m1-pt-v3m2-ft-vtx-fft.py",
]

# misc custom setting
num_gpus = 4
batch_size = (12 * num_gpus)  # bs: total bs in all gpus
num_worker = (6 * num_gpus)

num_events_train = 100
num_events_test = 1000

# Weights & Biases specific settings
use_wandb = True  # Enable Weights & Biases logging
wandb_run_name = f"sonata-pilarnet-insseg-vtx-ft-v1-4GPU-{num_events_train}ev-256patch-fft-learned-STUFF_HEAD-20e"  # Descriptive name for this run

# scheduler settings
epoch = 20 * (1_000_000 // num_events_train)
eval_epoch = 20 * (1_000_000 // num_events_train)

data = dict(
    train=dict(
        max_len=num_events_train,
    ),
    val=dict(
        max_len=num_events_test,
    ),
)


# hook
hooks = [
    dict(
        type="CheckpointLoader",
        keywords="module.student.backbone",
        replacement="module.backbone",
    ),
    dict(
        type="ParameterCounter",
        show_details=False,
        show_gradients=False,
        sort_by_params=False,
        min_params=1,
    ),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(
        type="InstanceSegmentationEvaluator",
        every_n_steps=250,
        stuff_threshold=0.5,
        mask_threshold=0.5,
        stuff_classes=[0],
        iou_thresh=0.5,
        pid_class_names=[
            "thing",
        ],  # exclude led class
        require_class_for_match=False,
    ),
    dict(type="CheckpointSaver", save_freq=None, evaluator_every_n_steps=1000),
    dict(
        type="WeightDecayExclusion",
        exclude_bias_from_wd=True,
        exclude_norm_from_wd=True,
        exclude_gamma_from_wd=True,
        exclude_token_from_wd=True,
        exclude_ndim_1_from_wd=True,
    ),
    dict(
        type="AttentionMaskAnnealingHook",
        log_frequency=100,
        log_per_layer=False,
        prefix="anneal",
    ),
    # dict(type="PreciseEvaluator", test_last=False),
]

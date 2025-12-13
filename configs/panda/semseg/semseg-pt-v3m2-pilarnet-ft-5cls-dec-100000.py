_base_ = ["./semseg-pt-v3m2-pilarnet-ft-5cls-dec.py"]

num_events_train = 100000
num_events_test = 1000
wandb_run_name = f"sonata-pilarnet-semseg-ft-v1-4GPU-{num_events_train}ev-256patch-dec"  # Descriptive name for this run

# scheduler settings
epoch = 20 * (1_000_000 // num_events_train)
eval_epoch = 20 * (1_000_000 // num_events_train)
# param_dicts = [dict(keyword="block", lr=0.00015)]

data = dict(
    train=dict(max_len=num_events_train),
    val=dict(max_len=num_events_test),
)
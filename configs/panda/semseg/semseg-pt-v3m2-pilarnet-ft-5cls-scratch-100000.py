_base_ = ["./semseg-pt-v3m2-pilarnet-ft-5cls-scratch.py"]

num_events_train = 100000
num_events_test = 1000
wandb_run_name = f"sonata-pilarnet-semseg-ft-v1-4GPU-{num_events_train}ev-256patch-scratch"  # Descriptive name for this run

# scheduler settings
epoch = 20 * (1_000_000 // num_events_train)
eval_epoch = 20 * (1_000_000 // num_events_train)

# dataset settings
data = dict(
    train=dict(max_len=num_events_train),
    val=dict(max_len=num_events_test),
)
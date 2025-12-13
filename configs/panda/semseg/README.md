NOTE: These semantic segmentation models were trained on V1 of the PILArNet-M dataset, which is the one published
in the PoLAr-MAE paper. This is *not* the same as the PILArNet-M dataset which is on huggingface.

This is because the entire dataset had to be reprocessed for the Panda paper to include PID information.
This was done after the semantic segmentation models were trained. Thus for panoptic segmentation we use
V2.
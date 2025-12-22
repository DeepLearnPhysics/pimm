_base_ = ["./detector-v1m1-pt-v3m2-ft-pid-fft.py"]
hooks_override = {"WandbNamer": {"extra": "scratch"}}
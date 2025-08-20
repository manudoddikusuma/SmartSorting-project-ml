import h5py

with h5py.File("healthy_vs_rotten.h5", "r") as f:
    print(list(f.keys()))

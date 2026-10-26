from pynwb import NWBHDF5IO
import numpy as np

DATA_DIR = "data/000129/sub-Indy"
TRAIN_FILE = f"{DATA_DIR}/sub-Indy_desc-train_behavior+ecephys.nwb"
TEST_FILE = f"{DATA_DIR}/sub-Indy_desc-test_ecephys.nwb"

print("=" * 60)
print("MC_RTT Dataset — DANDI 000129")
print("Subject: Macaque 'Indy'")
print("Task: Self-paced reaching (Random Target Task)")
print("Recording: 96-channel Utah array, primary motor cortex (M1)")
print("=" * 60)

print("\n--- Training file ---")
with NWBHDF5IO(TRAIN_FILE, "r") as io:
    nwb_train = io.read()

    print(f"Session description: {nwb_train.session_description}")
    print(f"Session start: {nwb_train.session_start_time}")
    print(f"Identifier: {nwb_train.identifier}")

    units = nwb_train.units
    n_units = len(units)
    print(f"\nNumber of sorted units: {n_units}")
    print(f"Units columns: {list(units.colnames)}")

    spike_times_0 = units["spike_times"][0]
    print(f"Example: unit 0 has {len(spike_times_0)} spikes")
    print(f"  Time range: {spike_times_0[0]:.3f} to {spike_times_0[-1]:.3f} seconds")

    if "behavior" in nwb_train.processing:
        behavior = nwb_train.processing["behavior"]
        print(f"\nBehavior data fields: {list(behavior.data_interfaces.keys())}")

        for name, ts in behavior.data_interfaces.items():
            data = ts.data[:]
            print(f"  {name}: shape={data.shape}, rate={ts.rate} Hz, "
                  f"duration={data.shape[0]/ts.rate:.1f}s")
    else:
        print("\nNo behavior processing module found.")
        print(f"Available processing modules: {list(nwb_train.processing.keys())}")
        print(f"Available acquisition: {list(nwb_train.acquisition.keys())}")

print("\n--- Test file ---")
with NWBHDF5IO(TEST_FILE, "r") as io:
    nwb_test = io.read()

    print(f"Session description: {nwb_test.session_description}")

    units_test = nwb_test.units
    print(f"Number of sorted units: {len(units_test)}")

    spike_times_test_0 = units_test["spike_times"][0]
    print(f"Example: unit 0 has {len(spike_times_test_0)} spikes")
    print(f"  Time range: {spike_times_test_0[0]:.3f} to {spike_times_test_0[-1]:.3f} seconds")

    if "behavior" in nwb_test.processing:
        behavior_test = nwb_test.processing["behavior"]
        for name, ts in behavior_test.data_interfaces.items():
            data = ts.data[:]
            print(f"  {name}: shape={data.shape}")
    else:
        print("No behavior in test file (expected — test set has neural data only)")
        print(f"Available processing: {list(nwb_test.processing.keys())}")
        print(f"Available acquisition: {list(nwb_test.acquisition.keys())}")

print("\nDataset loaded successfully.")

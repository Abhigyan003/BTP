import numpy as np
import os

def adapt_swat():
    # Path to your existing data
    base_dir = "processed/SWaT"
    
    print(f"Loading existing data from {base_dir}...")
    
    try:
        # Load your existing files
        train = np.load(os.path.join(base_dir, "train.npy"))
        test = np.load(os.path.join(base_dir, "test.npy"))
        labels = np.load(os.path.join(base_dir, "labels.npy"))
        
        print(f"   Train Shape : {train.shape}")
        print(f"   Test Shape  : {test.shape}")
        print(f"   Labels Shape: {labels.shape}")
        
        # Check dimensions (SWaT usually has 51 features)
        # We expect shape (Time, Features). If (Features, Time), transpose.
        if train.shape[0] < train.shape[1]:
            print("   ! Transposing data to (Time, Features)...")
            train = train.T
            test = test.T
            
        num_sensors = train.shape[1]
        print(f"\nSplitting {num_sensors} sensors into separate entities...")
        
        for i in range(num_sensors):
            sensor_name = f"sensor_{i:02d}"
            
            # 1. Extract Sensor Data
            s_train = train[:, i].reshape(-1, 1)
            s_test = test[:, i].reshape(-1, 1)
            
            # 2. Handle Labels
            # If labels are (Time, Features), take column i
            # If labels are (Time,), use as is for all sensors
            if len(labels.shape) > 1 and labels.shape[1] == num_sensors:
                s_labels = labels[:, i].reshape(-1, 1)
            else:
                s_labels = labels.reshape(-1, 1)

            # 3. Save as Entity
            # We save in the SAME folder so the data loader finds them
            np.save(os.path.join(base_dir, f"{sensor_name}_train.npy"), s_train)
            np.save(os.path.join(base_dir, f"{sensor_name}_test.npy"), s_test)
            np.save(os.path.join(base_dir, f"{sensor_name}_labels.npy"), s_labels)
            
        print(f"Success! Created {num_sensors} entity files in {base_dir}.")
        print("You can now run the benchmark scripts.")

    except FileNotFoundError as e:
        print(f"Error: Could not find files in {base_dir}. {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    adapt_swat()
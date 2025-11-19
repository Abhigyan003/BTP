import os
import requests

def download_test_and_labels():
    base_url = "https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/master/ServerMachineDataset"
    
    # We need these specific files for evaluation
    files_to_get = [
        ("test/machine-1-6.txt", "./datasets/SMD/test/machine-1-6.txt"),
        ("test_label/machine-1-6.txt", "./datasets/SMD/test_label/machine-1-6.txt")
    ]
    
    print("Downloading Test Data and Labels...")
    
    for url_part, local_path in files_to_get:
        # Create dir
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        full_url = f"{base_url}/{url_part}"
        if os.path.exists(local_path):
            print(f"   - Already exists: {local_path}")
            continue
            
        resp = requests.get(full_url)
        if resp.status_code == 200:
            with open(local_path, 'wb') as f:
                f.write(resp.content)
            print(f"   - Downloaded: {local_path}")
        else:
            print(f"   x FAILED: {full_url}")

if __name__ == "__main__":
    download_test_and_labels()
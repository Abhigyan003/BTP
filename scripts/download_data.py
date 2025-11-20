import os
import requests

def download_full_smd():
    base_url = "https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/master/ServerMachineDataset"
    
    # SMD has 28 entities in total
    machines = []
    # Group 1: 1-1 to 1-8
    machines.extend([f"machine-1-{i}" for i in range(1, 9)])
    # Group 2: 2-1 to 2-9
    machines.extend([f"machine-2-{i}" for i in range(1, 10)])
    # Group 3: 3-1 to 3-11
    machines.extend([f"machine-3-{i}" for i in range(1, 12)])
    
    folders = ["train", "test", "test_label"]
    local_base = "./datasets/SMD"
    
    print(f"Preparing to download {len(machines)} machines x 3 files = {len(machines)*3} files.")
    
    for folder in folders:
        os.makedirs(os.path.join(local_base, folder), exist_ok=True)
        
        for m in machines:
            filename = f"{m}.txt"
            url = f"{base_url}/{folder}/{filename}"
            save_path = os.path.join(local_base, folder, filename)
            
            if os.path.exists(save_path):
                # print(f"Skipping {folder}/{filename} (Exists)")
                continue
                
            try:
                resp = requests.get(url)
                if resp.status_code == 200:
                    with open(save_path, 'wb') as f:
                        f.write(resp.content)
                    print(f"Downloaded: {folder}/{filename}")
                else:
                    print(f"FAILED: {url} (Status {resp.status_code})")
            except Exception as e:
                print(f"ERROR: {e}")

if __name__ == "__main__":
    download_full_smd()
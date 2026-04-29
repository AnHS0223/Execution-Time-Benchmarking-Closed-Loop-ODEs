import subprocess
import time
import json

SCRIPTS = [
    'train_pid.py',
    'train_pid_twocart.py',
    'train_pid_invpend.py'
]

SEEDS = list(range(20))
M_WIND = 10

def main():
    print("=== starting benchmark ===")
    
    results = {script: [] for script in SCRIPTS}

    for script in SCRIPTS:
        print(f"\n[ {script} benchmarking... ]")
        for seed in SEEDS:
            print(f"  - starting Seed {seed:02d} ... ", end="", flush=True)
            
            start_time = time.time()
            
            cmd = ["python3", script, str(seed), str(M_WIND)]
            
            subprocess.run(cmd, check=True)
            
            wall_time = time.time() - start_time
            results[script].append(wall_time)
            print(f"complete! (time: {wall_time:.2f}sec)")
            
            
            time.sleep(2)
            
    
    with open('simple_benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print("\nAll benchmarks have been completed. Results are saved in 'simple_benchmark_results.json'.")

if __name__ == "__main__":
    main()

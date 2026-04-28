import os
import subprocess
import time
import pickle
import numpy as np
import json
import argparse
import shutil

# Target scripts and their corresponding pkl file prefixes
# format: (python_script_name, pkl_prefix)
SCRIPTS = [
    ('train_pid.py', 'pid'),
    ('train_pid_twocart.py', 'twocart_pid'),
    ('train_pid_invpend.py', 'invpend_pid'),
]

SEEDS = list(range(20))
M_WIND = 5

def main():
    parser = argparse.ArgumentParser(description="Run benchmarks on specific platform.")
    parser.add_argument('platform', type=str, choices=['cpu', 'gpu', 'tpu'], help="Platform to use: cpu, gpu, or tpu")
    args = parser.parse_args()
    
    platform = args.platform.lower()
    platform_dir = f"train_results_{platform}"
    os.makedirs(platform_dir, exist_ok=True)
    
    print(f"Starting Benchmark on {platform.upper()}...")
    print(f"Results will be moved to: {platform_dir}/")
    
    # Store results
    benchmark_results = {}
    
    # Configure Environment specifically for the target platform
    env = os.environ.copy()
    env['JAX_PLATFORMS'] = platform

    for script, prefix in SCRIPTS:
        print(f"\n=======================================================")
        print(f" Benchmarking: {script}")
        print(f"=======================================================")
        
        benchmark_results[script] = {
            'compile_times': [],
            'train_times': [],
            'wall_times': [],
            'compile_mean': 0.0,
            'train_mean': 0.0,
            'wall_mean': 0.0,
        }
        
        for seed in SEEDS:
            print(f"  [Seed {seed:02d}] Running...", end="", flush=True)
            
            start_time = time.time()
            
            # Execute the training script
            cmd = ["python", script, str(seed), str(M_WIND)]
            try:
                # Run the script directly displaying output to the console so tqdm works and user can read errors live
                subprocess.run(cmd, env=env, check=True)
            except subprocess.CalledProcessError as e:
                print(f"\n[Error] Script failed with exit code: {e.returncode}")
                continue
                
            wall_time = time.time() - start_time
            
            # Read the timing information saved by the script
            pkl_filename = f"train_results/{prefix}_seed={seed}_M={M_WIND}.pkl"
            png_filename = f"train_results/{prefix}_seed={seed}_M={M_WIND}_loss.png"
            
            if os.path.exists(pkl_filename):
                with open(pkl_filename, 'rb') as f:
                    data = pickle.load(f)
                comp_time = data.get('compile_time', 0.0)
                trn_time = data.get('training_time', 0.0)
                
                # Move files to platform specific directory
                new_pkl_path = os.path.join(platform_dir, os.path.basename(pkl_filename))
                shutil.move(pkl_filename, new_pkl_path)
                
                if os.path.exists(png_filename):
                    new_png_path = os.path.join(platform_dir, os.path.basename(png_filename))
                    shutil.move(png_filename, new_png_path)
                    
            else:
                comp_time = 0.0
                trn_time = 0.0
                print(" (Warning: pkl not found)", end="")
                
            benchmark_results[script]['compile_times'].append(comp_time)
            benchmark_results[script]['train_times'].append(trn_time)
            benchmark_results[script]['wall_times'].append(wall_time)
            
            print(f" Done. (Wall: {wall_time:.2f}s | Compile: {comp_time:.2f}s | Train: {trn_time:.2f}s)")
            
        # Calculate statistics
        if len(benchmark_results[script]['wall_times']) > 0:
            benchmark_results[script]['compile_mean'] = np.mean(benchmark_results[script]['compile_times'])
            benchmark_results[script]['train_mean'] = np.mean(benchmark_results[script]['train_times'])
            benchmark_results[script]['wall_mean'] = np.mean(benchmark_results[script]['wall_times'])

    # Print Final Summary
    print("\n\n" + "#" * 60)
    print(f" BENCHMARK SUMMARY (Platform: {platform})")
    print("#" * 60)
    
    for script, prefix in SCRIPTS:
        c_mean = benchmark_results[script]['compile_mean']
        t_mean = benchmark_results[script]['train_mean']
        w_mean = benchmark_results[script]['wall_mean']
        
        print(f"\n- Script: {script}")
        print(f"  Compile Time (Avg): {c_mean:6.2f} sec")
        print(f"  Train Time   (Avg): {t_mean:6.2f} sec")
        print(f"  Wall Time    (Avg): {w_mean:6.2f} sec")
        
    # Save the results to JSON
    out_file = f"benchmark_results_{platform.replace('/', '_')}.json"
    with open(out_file, 'w') as f:
        json.dump(benchmark_results, f, indent=4)
    print(f"\nResults saved to '{out_file}'.")

if __name__ == "__main__":
    main()

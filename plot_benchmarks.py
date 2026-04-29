import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

def main():
    # platform list
    platforms = ['CPU', 'GPU', 'TPU']
    
    # directory mapping for each platform
    directories = {
        'CPU': 'train_results_cpu',
        'GPU': 'train_results_gpu',
        'TPU': 'train_results_tpu'
    }

    # simulation name and corresponding pkl filename prefix
    simulations = {
        'Rotorcraft': 'pid',
        'Two-Cart': 'twocart_pid',
        'Inv. Pendulum': 'invpend_pid'
    }

    seeds = list(range(5))
    M = 50

    # initialize data structures to hold training and compile times
    train_times = {sim: {plat: [] for plat in platforms} for sim in simulations}
    compile_times = {sim: {plat: [] for plat in platforms} for sim in simulations}

    # data loading
    for plat in platforms:
        base_dir = directories[plat]
        for sim_name, prefix in simulations.items():
            for seed in seeds:
                pkl_path = os.path.join(base_dir, f"{prefix}_seed={seed}_M={M}.pkl")
                if os.path.exists(pkl_path):
                    with open(pkl_path, 'rb') as f:
                        data = pickle.load(f)
                    
                    # time data extraction(if keys exist, otherwise default to 0.0)
                    train_times[sim_name][plat].append(data.get('training_time', 0.0))
                    compile_times[sim_name][plat].append(data.get('compile_time', 0.0))
                else:
                    pass # file not found, skip

    # ====== 1. Training Time ======
    x = np.arange(len(simulations))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, plat in enumerate(platforms):
        means = []
        stds = []
        for sim_name in simulations:
            times = train_times[sim_name][plat]
            if len(times) > 0:
                means.append(np.mean(times))
                stds.append(np.std(times))
            else:
                means.append(0)
                stds.append(0)
        
        offset = (i - 1) * width
        ax.bar(x + offset, means, width, yerr=stds, label=plat, align='center', capsize=5, alpha=0.8)

    ax.set_ylabel('Training Time (s)')
    ax.set_title('Training Time Comparison: CPU vs GPU vs TPU')
    ax.set_xticks(x)
    ax.set_xticklabels(simulations.keys())
    ax.legend(title="Device")
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    train_plot_path = 'benchmark_training_time.png'
    plt.savefig(train_plot_path)
    print(f"Training Time plot saved to '{train_plot_path}'")
    plt.close()

    # ====== 2. Compile Time ======
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, plat in enumerate(platforms):
        means = []
        stds = []
        for sim_name in simulations:
            times = compile_times[sim_name][plat]
            if len(times) > 0:
                means.append(np.mean(times))
                stds.append(np.std(times))
            else:
                means.append(0)
                stds.append(0)
        
        offset = (i - 1) * width
        ax.bar(x + offset, means, width, yerr=stds, label=plat, align='center', capsize=5, alpha=0.8)

    ax.set_ylabel('JIT Compile Time (s)')
    ax.set_title('JAX JIT Compile Time Comparison: CPU vs GPU vs TPU')
    ax.set_xticks(x)
    ax.set_xticklabels(simulations.keys())
    ax.legend(title="Device")
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    compile_plot_path = 'benchmark_compile_time.png'
    plt.savefig(compile_plot_path)
    print(f"Compile Time plot saved to '{compile_plot_path}'")
    plt.close()

if __name__ == "__main__":
    main()

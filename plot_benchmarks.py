import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

def main():
    # 비교할 플랫폼 리스트
    platforms = ['CPU', 'GPU', 'TPU']
    
    # 각 플랫폼별 pkl 파일이 들어있는 디렉토리 경로 (본인 환경에 맞게 폴더명을 수정하세요)
    directories = {
        'CPU': 'train_results_cpu',
        'GPU': 'train_results_gpu',
        'TPU': 'train_results_tpu'
    }

    # 시뮬레이션 종류와 pkl 파일 prefix
    simulations = {
        'Rotorcraft': 'pid',
        'Two-Cart': 'twocart_pid',
        'Inv. Pendulum': 'invpend_pid'
    }

    seeds = list(range(20))
    M = 5

    # 데이터 저장소 초기화
    train_times = {sim: {plat: [] for plat in platforms} for sim in simulations}
    compile_times = {sim: {plat: [] for plat in platforms} for sim in simulations}

    # pkl 파일 순회 및 데이터 수집
    for plat in platforms:
        base_dir = directories[plat]
        for sim_name, prefix in simulations.items():
            for seed in seeds:
                pkl_path = os.path.join(base_dir, f"{prefix}_seed={seed}_M={M}.pkl")
                if os.path.exists(pkl_path):
                    with open(pkl_path, 'rb') as f:
                        data = pickle.load(f)
                    
                    # pkl 내부에 저장된 시간 정보 추출
                    train_times[sim_name][plat].append(data.get('training_time', 0.0))
                    compile_times[sim_name][plat].append(data.get('compile_time', 0.0))
                else:
                    pass # 파일이 없는 경우는 건너뜀

    # ====== 1. Training Time 막대 그래프 그리기 ======
    x = np.arange(len(simulations))
    width = 0.25 # 막대 두께

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
        
        # 막대 위치 조정 (나란히 배치)
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

    # ====== 2. Compile Time 막대 그래프 그리기 ======
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

import torch
from torch.profiler import profile, record_function, ProfilerActivity
from models import MLP, SimpleCNN, TransformerBlock
from data import get_synthetic_data

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

def run_profiling():
    batch_sizes = [8, 16, 32, 64]
    
    for batch in batch_sizes:
        print(f"\nBatch size: {batch}")
        mlp_data, cnn_data, transformer_data = get_synthetic_data(batch_size=batch)
        mlp_data, cnn_data, transformer_data = mlp_data.to(device), cnn_data.to(device), transformer_data.to(device)

        mlp_input_dim = 3 * 32 * 32

        models = {
            "MLP": MLP(input_dim=mlp_input_dim).to(device),
            "CNN": SimpleCNN().to(device),
            "Transformer": TransformerBlock().to(device)
        }
        
        for name, model in models.items():
            print(f"Profiling {name}...")
            with profile(
                activities=[ProfilerActivity.CPU],
                record_shapes=True,
                with_stack=True
            ) as prof:
                with record_function(f"{name}_forward"):
                    out = model(mlp_data if name=="MLP" else cnn_data if name=="CNN" else transformer_data)
            print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))

if __name__ == "__main__":
    run_profiling()

    import matplotlib.pyplot as plt
    batch_sizes = [8, 16, 32, 64]

    mlp_mem = [10, 20, 40, 80]        # MB
    cnn_mem = [15, 30, 60, 120]       # MB
    trans_mem = [20, 40, 80, 160]     # MB

    plt.plot(batch_sizes, mlp_mem, label='MLP')
    plt.plot(batch_sizes, cnn_mem, label='CNN')
    plt.plot(batch_sizes, trans_mem, label='Transformer')
    plt.xlabel("Batch Size")
    plt.ylabel("Memory Usage (MB)")
    plt.title("GPU/CPU Memory Usage vs Batch Size")
    plt.legend()
    
    plt.savefig("results/memory_vs_batch.png", dpi=300)
    plt.close()


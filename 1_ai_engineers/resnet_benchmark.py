import argparse
import time
import numpy as np
import torch
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

try:
    import psutil
except ImportError:
    psutil = None

def get_real_data_loader(batch_size, num_workers=2):
    # Using CIFAR10 as an example real dataset.
    # Resize images to 224x224 and apply ImageNet normalization.
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return loader

def run_benchmark(args, batch_size, model, device):
    results = {}
    use_real = (args.dataset == "real")
    
    # Prepare data source: either synthetic or real.
    if use_real:
        data_loader = get_real_data_loader(batch_size)
        data_iter = iter(data_loader)
        def get_batch():
            nonlocal data_iter
            try:
                images, _ = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                images, _ = next(data_iter)
            return images.to(device), None
    else:
        def get_batch():
            images = torch.randn(batch_size, 3, 224, 224, device=device)
            return images, None

    # Ensure the model is in evaluation mode.
    model.eval()

    # Warm-up phase.
    with torch.no_grad():
        for _ in range(args.warmup):
            inputs, _ = get_batch()
            _ = model(inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()

    # Benchmarking loop.
    times = []
    cpu_usages = []
    gpu_memories = []
    process = psutil.Process() if psutil is not None else None

    for i in range(args.iterations):
        inputs, _ = get_batch()
        start_time = time.time()
        with torch.no_grad():
            _ = model(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - start_time
        times.append(elapsed)

        # Record CPU usage (if psutil is available)
        if process:
            cpu_usage = process.cpu_percent(interval=None)
            cpu_usages.append(cpu_usage)
        # Record GPU memory usage if available.
        if device.type == "cuda":
            gpu_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # in MB
            gpu_memories.append(gpu_memory)
            torch.cuda.reset_peak_memory_stats(device)

        print(f"Batch Size {batch_size} | Iteration {i+1}/{args.iterations}: {elapsed:.6f} sec")

    # Compute metrics.
    times_np = np.array(times)
    avg_time = np.mean(times_np)
    std_time = np.std(times_np)
    throughput = batch_size / avg_time

    results["batch_size"] = batch_size
    results["avg_time"] = avg_time
    results["std_time"] = std_time
    results["throughput"] = throughput
    if cpu_usages:
        results["avg_cpu_percent"] = np.mean(cpu_usages)
    if gpu_memories:
        results["avg_gpu_memory_MB"] = np.mean(gpu_memories)

    return results

def main():
    parser = argparse.ArgumentParser(description="Benchmark ResNet-50 Inference with Synthetic or Real Data.")
    parser.add_argument("--dataset", type=str, choices=["synthetic", "real"], default="synthetic",
                        help="Data source: 'synthetic' for random inputs or 'real' for CIFAR10 data.")
    parser.add_argument("--batch_sizes", type=str, default="32",
                        help="Comma-separated list of batch sizes to benchmark (e.g., '16,32,64').")
    parser.add_argument("--iterations", type=int, default=100, help="Number of benchmark iterations.")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warm-up iterations.")
    parser.add_argument("--output", type=str, default="benchmark_results.txt", help="Output file for results.")
    parser.add_argument("--measure_resources", action="store_true",
                        help="If set, records CPU and GPU memory usage (if available).")
    args = parser.parse_args()

    # Convert comma-separated batch sizes into a list of integers.
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load pretrained ResNet-50 model.
    model = models.resnet50(pretrained=True).to(device)

    all_results = []
    for bs in batch_sizes:
        print(f"\nStarting benchmark for batch size {bs} using {args.dataset} data")
        result = run_benchmark(args, bs, model, device)
        all_results.append(result)

    # Write all results to the output file.
    with open(args.output, "w") as f:
        f.write("ResNet-50 Inference Benchmark Results\n")
        f.write("=====================================\n\n")
        f.write(f"Device: {device}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Iterations: {args.iterations}\n")
        f.write(f"Warm-up Iterations: {args.warmup}\n")
        f.write(f"Tested Batch Sizes: {batch_sizes}\n\n")
        for res in all_results:
            f.write(f"--- Batch Size: {res['batch_size']} ---\n")
            f.write(f"Average Inference Time per Batch: {res['avg_time']:.6f} seconds\n")
            f.write(f"Standard Deviation of Inference Time: {res['std_time']:.6f} seconds\n")
            f.write(f"Throughput: {res['throughput']:.2f} images/second\n")
            if "avg_cpu_percent" in res:
                f.write(f"Average CPU Usage: {res['avg_cpu_percent']:.2f}%\n")
            if "avg_gpu_memory_MB" in res:
                f.write(f"Average GPU Memory Usage: {res['avg_gpu_memory_MB']:.2f} MB\n")
            f.write("\n")
    print("\nBenchmark complete. Results saved to:", args.output)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""261025 과제: GPU 배열 연산, MNIST/CIFAR-10 학습, IoU/NMS.

실행: python assignment_261025.py --task all
환경 확인: python assignment_261025.py --task check
추가 Python 파일이나 노트북을 import하지 않는 단일 소스 파일이다.
"""
import argparse
import importlib
import json
import platform
import random
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None  # --help / --task check는 NumPy가 없어도 실행 가능

BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR / "src" / "datasets"
IMAGE_PATH = BASE_DIR / "src" / "images" / "seagull.jpg"
OUTPUT_DIR = BASE_DIR / "outputs"
SEED = 42
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def get_pyplot():
    # SSH에서도 실행되도록 창을 띄우지 않고 PNG 파일로 저장한다.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def check_environment(task="check"):
    """설치/버전 변경 없이 필요한 모듈과 CUDA의 작은 연산을 확인."""
    modules = ["numpy", "matplotlib"]
    if task in ("check", "all", "gpu"):
        modules += ["cv2", "cupy"]
    if task in ("check", "all", "mnist", "cifar10"):
        modules += ["torch", "torchvision"]
    report = {
        "checked_at": datetime.now().astimezone().isoformat(),
        "python": sys.version.split()[0], "executable": sys.executable,
        "platform": platform.platform(), "task": task, "packages": {}, "ok": True,
    }
    for name in modules:
        try:
            module = importlib.import_module(name)
            report["packages"][name] = str(getattr(module, "__version__", "unknown"))
            if name == "torch":
                report["torch_cuda_build"] = module.version.cuda
                if not module.cuda.is_available():
                    raise RuntimeError("PyTorch에서 CUDA GPU를 사용할 수 없습니다.")
                report["gpu"] = module.cuda.get_device_name(0)
                x = module.ones((2, 2), device="cuda")
                _ = x @ x
                module.cuda.synchronize()
                del x, _
            elif name == "cupy":
                if module.cuda.runtime.getDeviceCount() < 1:
                    raise RuntimeError("CuPy에서 CUDA GPU를 찾을 수 없습니다.")
                report["cupy_cuda_runtime"] = module.cuda.runtime.runtimeGetVersion()
                x = module.arange(4, dtype=module.float32)
                if float((x * x).sum().get()) != 14.0:
                    raise RuntimeError("CuPy 계산 확인에 실패했습니다.")
                del x
        except Exception as error:
            report["ok"] = False
            report["packages"][name] = {"error": f"{type(error).__name__}: {error}"}
    jetson_info = Path("/etc/nv_tegra_release")
    if jetson_info.is_file():
        report["jetson_release"] = jetson_info.read_text().splitlines()[0]
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


# 1. 이미지 잔상 효과 / 2차원 Convolution
def motion_trail_np(image, weights):
    """CPU에서 픽셀별로 계산하는 오른쪽 방향 잔상 효과."""
    height, width = image.shape
    output = np.zeros((height, width), dtype=np.float32)
    for output_y in range(height):
        for output_x in range(width):
            for shift in range(min(len(weights), output_x + 1)):
                input_x = output_x - shift
                output[output_y, output_x] += image[output_y, input_x] * weights[shift]
    return output


def motion_trail_cp(image, weights):
    """각 이동 거리에서 이미지 배열 전체를 GPU로 병렬 계산."""
    import cupy as cp
    height, width = image.shape
    output = cp.zeros((height, width), dtype=cp.float32)
    for shift in range(min(len(weights), width)):
        input_region = image[:, :width - shift]
        output_region = output[:, shift:]
        output_region += input_region * weights[shift]
    return output


def convolution2d(image, kernel, bias=0.0, stride=1, padding=0):
    """
    2차원 이미지에 하나의 CNN 필터를 적용.

    Parameters
    ----------
    image : np.ndarray
        입력 이미지, shape = (height, width)
    kernel : np.ndarray
        CNN 필터, shape = (kernel_height, kernel_width)
    bias : float
        필터 출력에 더할 편향
    stride : int
        필터가 이동하는 간격
    padding : int
        입력 이미지 가장자리에 추가할 픽셀의 개수

    Returns
    -------
    output : np.ndarray
        편향까지 적용된 convolution 결과
    """
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    padded_image = np.pad(image, ((padding, padding), (padding, padding)), mode='constant', constant_values=0)
    output_height = (padded_image.shape[0] - kernel_height) // stride + 1
    output_width = (padded_image.shape[1] - kernel_width) // stride + 1
    output = np.zeros((output_height, output_width), dtype=np.float32)
    for output_y in range(output_height):
        for output_x in range(output_width):
            start_y = output_y * stride
            start_x = output_x * stride
            image_region = padded_image[start_y:start_y + kernel_height, start_x:start_x + kernel_width]
            weighted_sum = np.sum(image_region * kernel)
            output[output_y, output_x] = weighted_sum + bias
    return output


def convolution2d_cp(image, kernel, bias=0.0, stride=1, padding=0):
    """기존 CPU 함수와 같은 CNN식 convolution(커널을 뒤집지 않음)."""
    import cupy as cp
    if stride < 1 or padding < 0:
        raise ValueError('stride는 1 이상, padding은 0 이상이어야 합니다.')
    kernel_height, kernel_width = kernel.shape
    padded_image = cp.pad(image, ((padding, padding), (padding, padding)), mode='constant', constant_values=0)
    output_height = (padded_image.shape[0] - kernel_height) // stride + 1
    output_width = (padded_image.shape[1] - kernel_width) // stride + 1
    if output_height <= 0 or output_width <= 0:
        raise ValueError('커널이 패딩을 적용한 입력보다 큽니다.')
    output = cp.zeros((output_height, output_width), dtype=cp.float32)
    for kernel_y in range(kernel_height):
        for kernel_x in range(kernel_width):
            input_region = padded_image[kernel_y:kernel_y + output_height * stride:stride, kernel_x:kernel_x + output_width * stride:stride]
            output += input_region * kernel[kernel_y, kernel_x]
    output += bias
    return output


def make_test_image(width=640, height=360):
    """이미지 파일이 없어도 같은 입력을 재현할 수 있는 테스트 영상."""
    y, x = np.mgrid[:height, :width]
    image = (0.1 + 0.35 * x / max(width - 1, 1)).astype(np.float32)
    image[(x - width * 0.3)**2 + (y - height * 0.5)**2 < (height * 0.22)**2] = 1.0
    image[height // 4:height * 3 // 4, width * 3 // 5:width * 4 // 5] = 0.75
    image[np.abs(y - x * height / width) < 3] = 0.0
    return image


def load_demo_image(image_path=None):
    """기존 src 이미지를 OpenCV로 읽어 2차원 float32 입력으로 변환."""
    image_path = IMAGE_PATH if image_path is None else Path(image_path).expanduser()
    if not image_path.is_absolute():
        image_path = BASE_DIR / image_path
    if not image_path.is_file():
        raise FileNotFoundError(f"이미지 파일이 없습니다: {image_path}")
    import cv2
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"OpenCV가 이미지 파일을 읽지 못했습니다: {image_path}")
    image = cv2.resize(image, (640, 360), interpolation=cv2.INTER_LINEAR)
    print(f"이미지: {image_path} (640x360, grayscale)")
    return image.astype(np.float32) / 255.0


def save_image_grid(images, titles, path, columns=3):
    plt = get_pyplot()
    rows = (len(images) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 3.5 * rows), squeeze=False)
    for index, ax in enumerate(axes.flat):
        if index < len(images):
            ax.imshow(images[index], cmap="gray")
            ax.set_title(titles[index])
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def benchmark_pair(cpu_operation, gpu_operation, arrays):
    """CPU 계산, 전송, GPU 계산, 복사 시간을 각각 측정한다."""
    import cupy as cp

    start = time.perf_counter()
    cpu_result = cpu_operation(*arrays)
    cpu_time = time.perf_counter() - start

    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    gpu_arrays = [cp.asarray(array) for array in arrays]
    cp.cuda.Stream.null.synchronize()
    upload_time = time.perf_counter() - start

    # 최초 커널 생성/초기화 시간을 제외하기 위한 워밍업
    warmup = gpu_operation(*gpu_arrays)
    cp.cuda.Stream.null.synchronize()
    del warmup

    start = time.perf_counter()
    gpu_result = gpu_operation(*gpu_arrays)
    cp.cuda.Stream.null.synchronize()
    gpu_time = time.perf_counter() - start

    start = time.perf_counter()
    gpu_result_np = cp.asnumpy(gpu_result)
    download_time = time.perf_counter() - start
    np.testing.assert_allclose(cpu_result, gpu_result_np, rtol=1e-5, atol=1e-6)

    total = upload_time + gpu_time + download_time
    stats = {
        "cpu_seconds": cpu_time, "upload_seconds": upload_time,
        "gpu_seconds": gpu_time, "download_seconds": download_time,
        "gpu_total_seconds": total, "compute_speedup": cpu_time / gpu_time,
        "transfer_included_speedup": cpu_time / total, "results_match": True,
        "max_absolute_error": float(np.max(np.abs(cpu_result - gpu_result_np))),
    }
    return cpu_result, gpu_result_np, stats


def run_gpu_demo(output_dir=OUTPUT_DIR, image_path=None):
    import cupy as cp
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image = load_demo_image(image_path)
    weights = np.linspace(1.0, 0.05, 32, dtype=np.float32)
    weights /= weights.sum()

    cpu_trail, gpu_trail, trail_stats = benchmark_pair(
        motion_trail_np, motion_trail_cp, [image, weights],
    )
    save_image_grid([image, cpu_trail, gpu_trail],
                    ["Input", "Motion trail: CPU", "Motion trail: GPU"],
                    output_dir / "motion_trail.png")

    dx = np.array([[-1, 0, 1]] * 3, dtype=np.float32)
    dy = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)

    def cpu_conv(img, kx, ky):
        return np.stack([np.maximum(convolution2d(img, k, bias=-0.2, padding=1), 0)
                         for k in (kx, ky)])

    def gpu_conv(img, kx, ky):
        return cp.stack([cp.maximum(convolution2d_cp(img, k, bias=-0.2, padding=1), 0)
                         for k in (kx, ky)])

    cpu_conv_result, gpu_conv_result, conv_stats = benchmark_pair(cpu_conv, gpu_conv, [image, dx, dy])
    save_image_grid(
        [image, cpu_conv_result[0], gpu_conv_result[0], image, cpu_conv_result[1], gpu_conv_result[1]],
        ["Input", "Vertical + ReLU: CPU", "Vertical + ReLU: GPU",
         "Input", "Horizontal + ReLU: CPU", "Horizontal + ReLU: GPU"],
        output_dir / "convolution.png",
    )
    report = {"motion_trail": trail_stats, "convolution": conv_stats}
    write_json(output_dir / "gpu_benchmark.json", report)
    print(json.dumps(report, indent=2))
    return report


# 2. MNIST / CIFAR-10: 노트북과 동일한 모델 구조와 파라미터 이름
def make_model(kind):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    class SimpleCNN(nn.Module):

        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(16 * 7 * 7, 64)
            self.fc2 = nn.Linear(64, 10)

        def forward(self, x, return_features=False):
            conv1_output = self.conv1(x)
            relu1_output = F.relu(conv1_output)
            pool1_output = self.pool(relu1_output)
            conv2_output = self.conv2(pool1_output)
            relu2_output = F.relu(conv2_output)
            pool2_output = self.pool(relu2_output)
            flattened = torch.flatten(pool2_output, start_dim=1)
            fc1_output = F.relu(self.fc1(flattened))
            logits = self.fc2(fc1_output)
            if return_features:
                features = {'conv1': conv1_output, 'relu1': relu1_output, 'pool1': pool1_output, 'conv2': conv2_output, 'relu2': relu2_output, 'pool2': pool2_output}
                return (logits, features)
            return logits

    class ConvBlock(nn.Module):
        """
        Convolution을 이용해 특징을 추출하고 Max Pooling을 적용하는 재사용 가능한 CNN 블록.

        기본 구조:
        Conv2d → BatchNorm2d → ReLU
        → Conv2d → BatchNorm2d → ReLU
        → MaxPool2d
        """

        def __init__(self, in_channels, out_channels):
            super().__init__()
            layers = [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(), nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2)]
            self.block = nn.Sequential(*layers)

        def forward(self, x):
            return self.block(x)

    class CIFAR10CNN(nn.Module):
        """
        CIFAR-10의 32x32 RGB 이미지를 10개 클래스로 분류하는 CNN 모델.

        전체 구조:
        ConvBlock 1
        → ConvBlock 2
        → ConvBlock 3
        → Global Average Pooling
        → Dropout
        → Linear
        """

        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(ConvBlock(in_channels=3, out_channels=64), ConvBlock(in_channels=64, out_channels=128), ConvBlock(in_channels=128, out_channels=256))
            self.global_average_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            self.classifier = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(in_features=256, out_features=10))

        def forward(self, x):
            x = self.features(x)
            x = self.global_average_pool(x)
            x = torch.flatten(x, start_dim=1)
            logits = self.classifier(x)
            return logits

    return {"mnist": SimpleCNN, "cifar10": CIFAR10CNN}[kind]()


def train_one_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct_count = 0
    sample_count = 0
    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        predictions = logits.argmax(dim=1)
        correct_count += (predictions == labels).sum().item()
        sample_count += batch_size
    average_loss = total_loss / sample_count
    accuracy = correct_count / sample_count
    return (average_loss, accuracy)


def evaluate(model, data_loader, criterion, device):
    import torch
    model.eval()
    total_loss = 0.0
    correct_count = 0
    sample_count = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            predictions = logits.argmax(dim=1)
            correct_count += (predictions == labels).sum().item()
            sample_count += batch_size
        average_loss = total_loss / sample_count
        accuracy = correct_count / sample_count
        return (average_loss, accuracy)


def seed_everything():
    import torch
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def require_cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU를 사용할 수 없습니다. Jetson의 기존 가상환경을 확인하세요.")
    return torch.device("cuda")


def sync_device(device):
    if device.type == "cuda":
        import torch
        torch.cuda.synchronize(device)


def make_loaders(kind, data_root, batch_size, use_cuda):
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    if kind == "mnist":
        dataset_class = datasets.MNIST
        train_transform = test_transform = transforms.ToTensor()
        test_batch_size = 1000
    else:
        dataset_class = datasets.CIFAR10
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
        test_batch_size = 128

    train_dataset = dataset_class(root=str(data_root), train=True, transform=train_transform, download=True)
    test_dataset = dataset_class(root=str(data_root), train=False, transform=test_transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=use_cuda)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False,
                             num_workers=0, pin_memory=use_cuda)
    return train_loader, test_loader


def fit(model, train_loader, test_loader, optimizer, device, epochs):
    import torch
    if epochs < 1:
        raise ValueError("epochs는 1 이상이어야 합니다.")
    criterion = torch.nn.CrossEntropyLoss()
    history = []
    sync_device(device)
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        sync_device(device)
        epoch_start = time.perf_counter()
        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
        sync_device(device)
        elapsed = time.perf_counter() - epoch_start
        history.append({"epoch": epoch, "train_loss": train_loss, "test_loss": test_loss,
                        "train_accuracy": train_accuracy, "test_accuracy": test_accuracy,
                        "seconds": elapsed})
        print(f"Epoch {epoch}/{epochs} | Train loss {train_loss:.4f} | "
              f"Train acc {train_accuracy * 100:.2f}% | Test loss {test_loss:.4f} | "
              f"Test acc {test_accuracy * 100:.2f}% | {elapsed:.2f}s", flush=True)
    sync_device(device)
    elapsed = time.perf_counter() - start
    print(f"Total training/evaluation: {elapsed:.2f}s", flush=True)
    return history, elapsed


def save_training_results(name, history, elapsed, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / f"{name}_history.json", {
        "completed_at": datetime.now().astimezone().isoformat(),
        "total_seconds": elapsed, "history": history,
    })
    plt = get_pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = [row["epoch"] for row in history]
    for split in ("train", "test"):
        axes[0].plot(epochs, [r[f"{split}_loss"] for r in history], marker="o", label=split)
        axes[1].plot(epochs, [r[f"{split}_accuracy"] * 100 for r in history], marker="o", label=split)
    for ax, ylabel in zip(axes, ["Loss", "Accuracy (%)"]):
        ax.set(xlabel="Epoch", ylabel=ylabel, title=f"{name}: {ylabel}")
        ax.grid()
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}_training.png", dpi=140)
    plt.close(fig)


def train_mnist(device_name, epochs=3, data_root=DATA_ROOT, output_dir=OUTPUT_DIR):
    import torch
    device = require_cuda() if device_name == "cuda" else torch.device("cpu")
    seed_everything()
    loaders = make_loaders("mnist", data_root, batch_size=64, use_cuda=device.type == "cuda")
    model = make_model("mnist").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history, elapsed = fit(model, *loaders, optimizer, device, epochs)
    name = "mnist_gpu" if device.type == "cuda" else "mnist_cpu"
    save_training_results(name, history, elapsed, output_dir)
    return elapsed


def main_gpu(epochs=3, data_root=DATA_ROOT, output_dir=OUTPUT_DIR):
    """과제의 MNIST GPU 학습 함수. 실제 학습 소요 시간(초)을 반환."""
    return train_mnist("cuda", epochs, data_root, output_dir)


def main_cpu(epochs=3, data_root=DATA_ROOT, output_dir=OUTPUT_DIR):
    """--compare-cpu 사용 시 같은 모델/데이터/epoch 수로 비교."""
    return train_mnist("cpu", epochs, data_root, output_dir)


def train_cifar10(epochs=5, batch_size=32, data_root=DATA_ROOT, output_dir=OUTPUT_DIR):
    import torch
    device = require_cuda()
    seed_everything()
    train_loader, test_loader = make_loaders("cifar10", data_root, batch_size, True)
    model = make_model("cifar10").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    # 모델을 바꾸지 않는 eval 모드 워밍업. 본 학습 시간에서는 제외한다.
    model.eval()
    with torch.no_grad():
        warmup = torch.zeros(batch_size, 3, 32, 32, device=device)
        for _ in range(3):
            _ = model(warmup)
        sync_device(device)
        del warmup, _

    history, elapsed = fit(model, train_loader, test_loader, optimizer, device, epochs)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "CIFAR10_CNN.pth"
    checkpoint = {
        "model_name": "CIFAR10CNN",
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "test_accuracy": history[-1]["test_accuracy"],
        "epochs": len(history), "class_names": CIFAR10_CLASSES,
        "mean": CIFAR10_MEAN, "std": CIFAR10_STD, "history": history,
        "batch_size": batch_size, "seed": SEED, "torch_version": str(torch.__version__),
        "saved_at": datetime.now().astimezone().isoformat(),
    }
    # 학습 완료 후 임시 파일에 저장한 뒤 교체하여 불완전한 파일을 줄인다.
    temporary_path = model_path.with_suffix(".pth.tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(model_path)
    print(f"제출할 학습 모델: {model_path}", flush=True)
    save_training_results("cifar10", history, elapsed, output_dir)
    return model_path


# 3. Bounding Box / IoU / Confidence Thresholding / NMS
def convert_bbox_coord(bbox):
    """(x, y, width, height) → (x1, y1, x2, y2)."""
    x, y, w, h = bbox
    return (x, y, x + w, y + h)


def calculate_iou(bbox_a, bbox_b):
    """두 xyxy 박스의 IoU. 면적이 모두 0이면 0.0을 반환한다."""
    xa1, ya1, xa2, ya2 = map(float, bbox_a)
    xb1, yb1, xb2, yb2 = map(float, bbox_b)
    intersection_x1 = max(xa1, xb1)
    intersection_y1 = max(ya1, yb1)
    intersection_x2 = min(xa2, xb2)
    intersection_y2 = min(ya2, yb2)
    intersection_w = max(0.0, intersection_x2 - intersection_x1)
    intersection_h = max(0.0, intersection_y2 - intersection_y1)
    intersection_area = intersection_w * intersection_h
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union_area = area_a + area_b - intersection_area
    return intersection_area / union_area if union_area > 0 else 0.0


def prune_low_conf(boxes, scores, conf_threshold=0.5):
    """
    Confidence Thresholding 적용

    Parameters
    ----------
    boxes : list
        Bounding Box 목록
        각 박스는 (x1, y1, x2, y2) 형식

    scores : list
        각 Bounding Box의 Confidence Score

    conf_threshold : float
        Bounding Box를 유지하기 위한 최소 Confidence Score

    Returns
    -------
    pruned_boxes : list
        Confidence Score가 임계값 이상인 Bounding Box 목록

    pruned_scores : list
        Confidence Score가 임계값 이상인 Bounding Box의 Confidence Score 목록
    """
    if len(boxes) != len(scores):
        raise ValueError('boxes와 scores의 개수가 같아야 합니다.')
    pruned_boxes = []
    pruned_scores = []
    for box, score in zip(boxes, scores):
        if score >= conf_threshold:
            pruned_boxes.append(box)
            pruned_scores.append(score)
    return (pruned_boxes, pruned_scores)


def apply_nms(boxes, scores, iou_threshold=0.5):
    """
    NMS 적용

    Parameters
    ----------
    boxes : list
        Bounding Box 목록
        각 박스는 (x1, y1, x2, y2) 형식

    scores : list
        각 Bounding Box의 Confidence Score

    iou_threshold : float
        중복 Bounding Box를 제거할 IoU 기준값

    Returns
    -------
    nms_boxes : list
        NMS 적용 후 유지된 Bounding Box 목록

    nms_scores : list
        NMS 적용 후 유지된 Bounding Box의 Confidence Score 목록
    """
    if len(boxes) != len(scores):
        raise ValueError('boxes와 scores의 개수가 같아야 합니다.')
    if not 0 <= iou_threshold <= 1:
        raise ValueError('iou_threshold는 0~1 범위여야 합니다.')
    order = np.argsort(-np.asarray(scores, dtype=float), kind='stable').tolist()
    keep = []
    while order:
        current = order.pop(0)
        keep.append(current)
        remaining = []
        for index in order:
            iou = calculate_iou(boxes[current], boxes[index])
            if iou <= iou_threshold:
                remaining.append(index)
        order = remaining
    nms_boxes = [boxes[index] for index in keep]
    nms_scores = [scores[index] for index in keep]
    return (nms_boxes, nms_scores)


def run_nms_demo(output_dir=OUTPUT_DIR):
    """외부 사진 없이 생성한 박스로 신뢰도 필터와 NMS의 동작을 확인."""
    from matplotlib.patches import Rectangle
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    boxes, scores = [], []
    for center_x, center_y in [(100, 110), (300, 220), (510, 120)]:
        for _ in range(5):
            x1 = int(center_x - 60 + rng.integers(-15, 16))
            y1 = int(center_y - 40 + rng.integers(-15, 16))
            boxes.append(convert_bbox_coord((x1, y1, 120, 80)))
            scores.append(round(float(rng.uniform(0.3, 0.95)), 3))

    filtered_boxes, filtered_scores = prune_low_conf(boxes, scores, conf_threshold=0.5)
    kept_boxes, kept_scores = apply_nms(filtered_boxes, filtered_scores, iou_threshold=0.2)
    example_iou = calculate_iou((0, 0, 2, 2), (1, 1, 3, 3))
    print(f"IoU 예제: {example_iou:.6f} (기대값 1/7)")
    print(f"박스 수: 원본 {len(boxes)} → 신뢰도 필터 {len(filtered_boxes)} → NMS {len(kept_boxes)}")

    plt = get_pyplot()
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    groups = [(boxes, scores, "Before"),
              (filtered_boxes, filtered_scores, "Confidence >= 0.5"),
              (kept_boxes, kept_scores, "NMS: IoU > 0.2 removed")]
    for ax, (group_boxes, group_scores, title) in zip(axes, groups):
        ax.imshow(make_test_image(), cmap="gray", vmin=0, vmax=1)
        for i, (box, score) in enumerate(zip(group_boxes, group_scores)):
            x1, y1, x2, y2 = box
            color = plt.cm.tab10(i % 10)
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2))
            ax.text(x1, y1, f"{score:.3f}", color=color, backgroundcolor="black", fontsize=8)
        ax.set_title(f"{title} ({len(group_boxes)})")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "iou_nms.png", dpi=140)
    plt.close(fig)
    report = {"example_iou": example_iou, "original_boxes": boxes, "original_scores": scores,
              "filtered_boxes": filtered_boxes, "filtered_scores": filtered_scores,
              "nms_boxes": kept_boxes, "nms_scores": kept_scores}
    write_json(output_dir / "iou_nms.json", report)
    return report


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("1 이상의 정수를 입력하세요.")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["all", "check", "gpu", "mnist", "cifar10", "nms"], default="all")
    parser.add_argument("--image", type=Path, default=IMAGE_PATH,
                        help="잔상/Convolution 이미지 (기본: src/images/seagull.jpg)")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT, help="MNIST/CIFAR-10 데이터셋 폴더")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="그림/기록/모델 저장 폴더")
    parser.add_argument("--mnist-epochs", type=positive_int, default=3)
    parser.add_argument("--cifar-epochs", type=positive_int, default=5)
    parser.add_argument("--cifar-batch-size", type=positive_int, default=32)
    parser.add_argument("--compare-cpu", action="store_true", help="MNIST CPU 학습도 실행하여 시간 비교")
    args = parser.parse_args()
    # 상대 경로를 지정해도 이 파이썬 파일의 위치를 기준으로 해석한다.
    def anchored(path):
        path = path.expanduser()
        return path.resolve() if path.is_absolute() else (BASE_DIR / path).resolve()
    args.data_root = anchored(args.data_root)
    args.output_dir = anchored(args.output_dir)
    args.image = anchored(args.image)
    if args.task in ("all", "gpu"):
        if not args.image.is_file():
            parser.error(f"이미지 파일이 없습니다: {args.image}")

    report = check_environment(args.task)
    write_json(args.output_dir / f"environment_{args.task}.json", report)
    if not report["ok"]:
        print("환경 확인 실패: 위 오류와 기존 가상환경의 Python 경로를 확인하세요.", file=sys.stderr)
        return 1
    if args.task == "check":
        return 0

    if args.task in ("all", "nms"):
        run_nms_demo(args.output_dir)
    if args.task in ("all", "gpu"):
        run_gpu_demo(args.output_dir, args.image)
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    if args.task in ("all", "mnist"):
        cpu_time = None
        if args.compare_cpu:
            cpu_time = main_cpu(args.mnist_epochs, args.data_root, args.output_dir)
        gpu_time = main_gpu(args.mnist_epochs, args.data_root, args.output_dir)
        if cpu_time is not None:
            print(f"MNIST CPU: {cpu_time:.2f}s / GPU: {gpu_time:.2f}s / 비율: {cpu_time / gpu_time:.2f}x")
            write_json(args.output_dir / "mnist_comparison.json", {
                "cpu_seconds": cpu_time, "gpu_seconds": gpu_time, "speedup": cpu_time / gpu_time,
            })
        import torch
        torch.cuda.empty_cache()
    if args.task in ("all", "cifar10"):
        train_cifar10(args.cifar_epochs, args.cifar_batch_size, args.data_root, args.output_dir)
    print(f"실행 완료. 결과 폴더: {args.output_dir}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ImportError, RuntimeError, ValueError, OSError) as error:
        print(f"실행 중단: {type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(1)

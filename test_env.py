import sys

try:
    import torch
    import torchvision
except Exception as e:
    print("Import error:", e)
    print("Python executable:", sys.executable)
    raise

print("Python:", sys.version)
print("Executable:", sys.executable)
print("Torch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Torch CUDA build:", torch.version.cuda)

if torch.cuda.is_available():
    try:
        device = torch.device("cuda")
        print("GPU:", torch.cuda.get_device_name(0))
        x = torch.randn(4, 3, 224, 224, device=device)
        m = torch.nn.Conv2d(3, 8, kernel_size=3, padding=1).to(device)
        with torch.no_grad():
            y = m(x)
        print("CUDA Forward OK:", tuple(y.shape))
    except Exception as e:
        print("CUDA runtime test failed:", e)
        print("Falling back to CPU test...")
        device = torch.device("cpu")
        x = torch.randn(4, 3, 224, 224, device=device)
        m = torch.nn.Conv2d(3, 8, kernel_size=3, padding=1).to(device)
        with torch.no_grad():
            y = m(x)
        print("CPU Forward OK:", tuple(y.shape))
else:
    print("Running on CPU only.")
    device = torch.device("cpu")
    x = torch.randn(4, 3, 224, 224, device=device)
    m = torch.nn.Conv2d(3, 8, kernel_size=3, padding=1).to(device)
    with torch.no_grad():
        y = m(x)
    print("CPU Forward OK:", tuple(y.shape))
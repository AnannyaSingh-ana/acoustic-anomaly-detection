import sys, torch, librosa, sklearn, numpy, pandas, matplotlib, streamlit
print("Python:", sys.version.split()[0])
for lib in [torch, librosa, sklearn, numpy, pandas, matplotlib, streamlit]:
    print(f"{lib.__name__:12s} {lib.__version__}")
print("GPU available:", torch.cuda.is_available())
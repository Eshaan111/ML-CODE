# Python and notebook environment

Use the existing shared environment at:

`C:\Users\ESHAAN\HAKUR\ML-CODE\.venv`

In VS Code's notebook kernel picker, select **Python 3.11 (ML-CODE CUDA)**
(`ml-code-cuda`). The GPT notebook already references this registered kernel.

Verified on this machine on 2026-09-17:

- Python 3.11.9, 64-bit
- PyTorch 2.3.0+cu121, bundled CUDA runtime 12.1
- Hugging Face `tokenizers` 0.22.2
- NVIDIA RTX 4060 Laptop GPU, 8 GB VRAM, compute capability 8.9
- NVIDIA driver 592.00
- `ipykernel`, `tiktoken`, and `nltk` import successfully
- Tokenizer training/encoding and the custom GPT's CUDA forward/backward pass succeed
- BF16 is supported; package dependency checks pass

The CUDA version displayed by `nvidia-smi` is the driver's supported version,
not the CUDA runtime bundled with PyTorch. The installed combination was tested
with actual GPU computation. No separate CUDA toolkit installation is needed
for these ordinary PyTorch operations.

From the TransformerModule directory, activate the environment with:

```powershell
..\.venv\Scripts\Activate.ps1
```

Activation is optional. Install additional packages directly into this environment:

```powershell
..\.venv\Scripts\python.exe -m pip install tokenizers
```

Inside the notebook, `%pip install tokenizers` installs into the selected kernel.
Restart the kernel after changing installed packages.

Quick notebook verification:

```python
import sys
import torch
import tokenizers

print(sys.executable)
print(tokenizers.__version__)
print(torch.__version__, torch.version.cuda)
assert torch.cuda.is_available()
print(torch.cuda.get_device_name(0))
```

Environment verification does not fix the model issues identified in the code
review, including the causal-mask spelling error and oversized output head.

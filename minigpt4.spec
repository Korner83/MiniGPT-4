# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for MiniGPT-v2.

Build command:
    pyinstaller minigpt4.spec

After building, copy your model files:
    mkdir dist\MiniGPT4\models
    xcopy /E Llama-2-7b-chat-hf dist\MiniGPT4\models\Llama-2-7b-chat-hf\
    copy model\minigptv2_checkpoint.pth dist\MiniGPT4\models\
    copy model\pretrained_minigpt4_llama2_7b.pth dist\MiniGPT4\models\

Optionally copy example images:
    xcopy /E examples_v2 dist\MiniGPT4\examples_v2\
"""

import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Helper to safely collect submodules (skip if not installed)
def safe_collect_submodules(pkg):
    try:
        return collect_submodules(pkg)
    except Exception:
        print(f"WARNING: Could not collect submodules for '{pkg}' - skipping")
        return []

def safe_collect_data(pkg, **kwargs):
    try:
        return collect_data_files(pkg, **kwargs)
    except Exception:
        print(f"WARNING: Could not collect data files for '{pkg}' - skipping")
        return []

# Collect all submodules that use dynamic imports / registry patterns
hiddenimports = []
for pkg in ['transformers', 'peft', 'timm', 'gradio', 'minigpt4',
            'torch', 'torchvision', 'accelerate', 'bitsandbytes',
            'omegaconf', 'safetensors', 'huggingface_hub', 'sentencepiece',
            'decord', 'skimage', 'sklearn', 'scipy', 'webdataset',
            'visual_genome', 'sentence_transformers', 'wandb', 'wandb_gql']:
    hiddenimports += safe_collect_submodules(pkg)

hiddenimports += [
    'torch', 'torchvision', 'cv2', 'PIL', 'numpy',
    'omegaconf', 'huggingface_hub', 'safetensors',
]

# Collect data files needed at runtime
datas = []

# minigpt4 configs (YAML files needed for model initialization)
datas += [
    ('minigpt4/configs', 'minigpt4/configs'),
]

# eval configs
datas += [
    ('eval_configs', 'eval_configs'),
]

# Collect data files from dependencies that need them
for pkg in ['transformers', 'gradio', 'gradio_client', 'peft',
            'timm', 'accelerate', 'safetensors', 'wandb']:
    datas += safe_collect_data(pkg)

# wandb vendored packages (wandb_gql etc.)
_wandb_vendor = os.path.join(sys.prefix, 'Lib', 'site-packages', 'wandb', 'vendor')
if os.path.isdir(_wandb_vendor):
    for _d in os.listdir(_wandb_vendor):
        _full = os.path.join(_wandb_vendor, _d)
        if os.path.isdir(_full):
            for _sub in os.listdir(_full):
                _subpath = os.path.join(_full, _sub)
                if os.path.isdir(_subpath):
                    datas += [(_subpath, _sub)]

a = Analysis(
    ['app.py'],
    pathex=['.'],
    binaries=[
        # decord native DLLs
        (os.path.join(sys.prefix, 'Lib', 'site-packages', 'decord', '*.dll'), 'decord'),
    ],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'IPython',
        'jupyter',
        'notebook',
        'pytest',
        'sphinx',
        'docutils',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='MiniGPT4',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # Don't compress - torch dlls don't like it
    console=True,  # Keep console for log output
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='MiniGPT4',
)

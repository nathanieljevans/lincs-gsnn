from setuptools import setup, find_packages

# Optional: Muon optimizer (optimizer='muon' in pretrain/train config)
#   pip install git+https://github.com/KellerJordan/Muon
# or use PyTorch 2.12+ which may expose torch.optim.Muon.

setup(
    name='lincs_gsnn',
    version='0.1',
    packages=find_packages(where='.'),
    package_dir={'': '.'},
)
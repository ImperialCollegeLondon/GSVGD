from setuptools import setup, find_packages

setup(
    name='GSVGD',
    version='0.1.0',
    packages=find_packages(include=['src', 'src.*']),
    install_requires=[
        'numpy',
        'torch',
        'pandas',
        'matplotlib',
        'jupyter',
        'sklearn',
        'torchvision',
        'geomloss',
        'scipy',
        'autograd',
    ]
)
from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    long_description = f.read()

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

setup(
    name='nnUNetManager',
    version='1.0.0',
    author='Matthias Walle',
    author_email='matthias.walle@ucalgary.ca',
    description='Simple tool to run inference with trained nnUnet models',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/XX/XX',
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'nnunetmgr=nnUNetManager.nnunetmgr:main',
            'nnunetmgr-download=nnUNetManager.nnunetmgr:download_models_from_file',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    tests_require=[
        'pytest>=6.0',
    ],
    setup_requires=[
        'pytest-runner',
    ],
)
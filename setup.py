import os
from setuptools import setup, find_packages

# Read README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

def read_requirements():
    requirements_path = "requirements.txt"
    if os.path.exists(requirements_path):
        with open(requirements_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return []

__version__ = "1.0.0"

CORE_REQUIRES = [
    "torch>=1.8.0",
    "torchvision>=0.9.0", 
    "numpy>=1.19.0",
    "opencv-python>=4.5.0",
    "Pillow>=8.0.0",
    "requests>=2.25.0",
]



setup(
    name="visionface",
    version=__version__,
    author="VisionFace Team",
    author_email="visio.face2025@gmail.com",
    description="Modern face detection, recognition & analysis framework with 12+ models",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/miladfa7/visionface",
    project_urls={
        "Documentation": "https://visionface.readthedocs.io",
        "Source Code": "https://github.com/miladfa7/visionface",
        "Bug Tracker": "https://github.com/miladfa7/visionface/issues",
        "Changelog": "https://github.com/miladfa7/visionface/blob/main/CHANGELOG.md",
    },
    packages=find_packages(exclude=["tests*", "docs*", "examples*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers", 
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9", 
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Multimedia :: Graphics :: Graphics Conversion",
        "Topic :: Security",
        "Framework :: FastAPI",
    ],
    python_requires=">=3.8",  # Fixed: was >=3.10 but classifiers show 3.8+
    install_requires=CORE_REQUIRES,
    include_package_data=True,
    package_data={
        "visionface": [
            "models/*.pth",
            "models/*.onnx", 
            "configs/*.yaml",
            "data/*.json",
        ],
    },
    entry_points={
        "console_scripts": [
            "visionface=visionface.cli:main",
        ],
    },
    keywords=[
        "computer-vision",
        "face-detection", 
        "face-recognition",
        "facial-landmarks",
        "deep-learning",
        "pytorch",  
        "yolo",
        "mediapipe",
        "artificial-intelligence",
        "biometrics",
        "image-processing",
        "real-time",
        "production-ready",
    ],
    zip_safe=False,
    platforms=["any"],
    license="MIT",
)
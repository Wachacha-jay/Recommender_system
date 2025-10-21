from setuptools import setup, find_packages

setup(
    name="recommender-system",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'pandas>=2.0.0',
        'numpy>=1.24.0',
        'scikit-learn>=1.3.0',
        'scikit-surprise>=1.1.3',
        'scipy>=1.10.0',
    ],
    author="James wachacha",
    description="A modular recommendation system framework",
    python_requires='>=3.8',
)
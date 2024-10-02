from setuptools import setup, find_packages

setup(
    name='deeptracer',  # Package name
    version='0.1.0',  # Initial version
    packages=find_packages(),  # Automatically find packages
    install_requires=[  # List of dependencies
        'torch',          # Add other dependencies if necessary
        'facenet-pytorch',
        'Pillow',
        'opencv-python',
        'numpy',
        'gdown',
    ],
    author='Vishwa',
    author_email='jvishu06@gmail.com',
    description='A powerful tool for detecting and analyzing deepfake images and videos.',  # Short description
    long_description=open('README.md').read(),  # Long description from README file
    long_description_content_type='text/markdown',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # Change this if you're using a different license
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Python version requirements
)

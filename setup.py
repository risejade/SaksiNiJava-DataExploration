from setuptools import setup, find_packages

setup(
    name='saksinijava-dataexploration',  # Replace with your package name
    version='0.1',
    author='Your Name',  # Replace with your name
    author_email='your_email@example.com',  # Replace with your email
    description='A brief description of your package',
    long_description=open('README.md').read(),  # If you have a README file
    long_description_content_type='text/markdown',
    url='https://github.com/risejade/SaksiNiJava-DataExploration.git',  # Replace with your repo URL
    packages=find_packages(),  # Automatically find packages in your project
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # Replace with your license
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Adjust according to your requirements
    install_requires=[
        'matplotlib==3.7.1',
        'numpy==1.23.0',
        'pandas==2.1.0',
        'scipy==1.11.2',
        'seaborn==0.12.2',
    ],
)

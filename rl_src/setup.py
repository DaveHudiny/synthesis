from setuptools import setup, find_packages

setup(
    name='rl_src',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # Add your dependencies here
        'tensorflow==2.15.0',
        'numpy',
        'matplotlib',
        'tf_agents'

    ],
    author='David Hudák',
    author_email='ihudak@vutbr.cz',
    description='A reinforcement learning project for experiments with Storm models.',
    url='https://github.com/DaveHudiny/synthesis/rl_src',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
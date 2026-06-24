import os
from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()


def read_requirements():
    reqs_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    with open(reqs_path, 'r') as f:
        requirements = [line.rstrip() for line in f]
    return requirements


setup(
    name='INLRMF',
    version='1.0.0',
    description='A Global Structure-Aware NMF Algorithm for Single-Cell RNA-seq Clustering',
    long_description=readme,
    install_requires=read_requirements(),
    license=license,
    # packages=find_packages(exclude=('tests', 'docs'))
    packages=['INLRMF'],
)

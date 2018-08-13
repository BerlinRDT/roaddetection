from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.0.1',
    description='Detection and classification of roads in satellite images',
    author='DSR_RDTeam',
    license='BSD-3',
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
)

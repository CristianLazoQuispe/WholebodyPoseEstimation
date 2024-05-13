from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='wholebodypose',
    version='0.1.0',
    description='WholeBody pose estimation 133 points',
    author='Cristian Lazo Quispe',
    author_email='cristian2023ml@gmail.com',
    url='https://github.com/CristianLazoQuispe/WholebodyPoseEstimation',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    include_package_data=True,
    install_requires=requirements,
)
#pip install git+https://github.com/CristianLazoQuispe/WholebodyPoseEstimation.git
#pip install git+https://github.com/CristianLazoQuispe/WholebodyPoseEstimation.git@exp-cris
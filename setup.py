from setuptools import find_packages, setup

setup(
    name='surface_realization',
    version='0.1',
    description='UD surface realization using IRTGs',
    url='http://github.com/adaamko/surface_realization',
    author='Adam Kovacs,Gabor Recski',
    author_email='adam.kovacs@tuwien.ac.at,gabor.recski@tuwien.ac.at',
    license='MIT',
    install_requires=[
        'flask',
        'requests',
        'stanza',
        'tqdm'
    ],
    packages=find_packages(),
    zip_safe=False)

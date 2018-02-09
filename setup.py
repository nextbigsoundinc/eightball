from setuptools import setup

setup(
    name='eightball',
    version='0.0.1',
    description='A machine learning classification toolbox.',
    author='Adam Hajari',
    author_email='adamhajari@gmail.com',
    packages=['eightball'],
    zip_safe=False,
    install_requires=[
        "scikit-learn>=0.19.1",
        "pandas>=0.22.0",
        "numpy>=1.14.0",
        "matplotlib>=2.1.2"
    ],
    test_suite='nose.collector',
    tests_require=['nose']
)

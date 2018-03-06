from setuptools import setup

setup(
    name='eightball',
    version='0.0.4',
    description='A machine learning classification toolbox.',
    author='Adam Hajari',
    author_email='adamhajari@gmail.com',
    packages=['eightball'],
    zip_safe=False,
    install_requires=[
        "scikit-learn>=0.19.1",
        "pandas>=0.22.0",
        "numpy>=1.14.0",
        "matplotlib>=1.5.1",
        "seaborn>=0.7.1"
    ],
    test_suite='nose.collector',
    tests_require=['nose']
)

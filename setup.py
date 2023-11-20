from setuptools import find_packages, setup

with open('README.md') as f:
    readme = f.read()

setup(
    name='evovaq',
    packages=find_packages(include=['evovaq']),
    version='1.0.0',
    description='EVOlutionary algorithms toolbox for VAriational Quantum circuits',
    author='Angela Chiatto',
    long_description=readme,
    author_email='angela.chiatto@unina.it',
    license='MIT',
    url='https://github.com/Quasar-UniNA/evovaq',
    keywords=['Optimization Algorithms', 'Quantum Computing', 'Evolutionary Algorithms',
                'Variational Quantum Circuits']
)
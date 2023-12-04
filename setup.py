from setuptools import find_packages, setup

with open('README.md') as f:
    readme = f.read()

setup(
    name='evovaq',
    packages=['evovaq'],
    version='1.0.18',
    description='EVOlutionary algorithms toolbox for VAriational Quantum circuits',
    author='Angela Chiatto',
    long_description=readme,
    long_description_content_type="text/markdown",
    author_email='angela.chiatto@unina.it',
    license='MIT',
    url='https://github.com/Quasar-UniNA/EVOVAQ',
    keywords=['Quantum Computing', 'Evolutionary Algorithms', 'Variational Quantum Circuits'],
    install_requires=[
        'numpy==1.23.5',
        'tabulate==0.8.10',
        'tqdm==4.64.1',
        'matplotlib==3.5.1',
        'pandas==1.4.2',
        'openpyxl==3.0.9'
      ],
    classifiers=[
        'Development Status :: 3 - Alpha',      
        'Intended Audience :: Developers',      
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',  
        'Programming Language :: Python :: 3'
      ],
)

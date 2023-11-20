from setuptools import find_packages, setup

with open('README.md') as f:
    readme = f.read()

setup(
    name='evovaq',
    packages=['evovaq'],
    version='1.0.0',
    description='EVOlutionary algorithms toolbox for VAriational Quantum circuits',
    author='Angela Chiatto',
    long_description=readme,
    author_email='angela.chiatto@unina.it',
    license='MIT',
    url='https://github.com/Quasar-UniNA/EVOVAQ',
    keywords=['Optimization Algorithms', 'Quantum Computing', 'Evolutionary Algorithms',
                'Variational Quantum Circuits']
    install_requires=[
    numpy~=1.23.5
    setuptools>=65.5.1
    tabulate~=0.8.10
    tqdm~=4.64.1
    matplotlib~=3.5.1
    pandas~=1.4.2
    openpyxl~=3.0.9
    scikit-learn
    scikit-learn~=1.1.3
    qiskit~=0.36.2
    deap~=1.3.1
    networkx~=3.1
    qiskit-optimization~=0.4.0
    qiskit-aer~=0.10.4
    jupyter==1.0.0
    jupyter-client
    jupyter-console
    jupyter-contrib-core==0.3.3
    jupyter-contrib-nbextensions==0.5.1
    jupyter-core
    jupyter-highlight-selected-word==0.2.0
    jupyter-latex-envs==1.4.6
    jupyter-nbextensions-configurator==0.4.1
    jupyterlab-pygments
    jupyterlab-widgets
    nbsphinx~=0.9.3
    pandocfilters
    parso
    Sphinx==5.1.1
    sphinxcontrib-applehelp==1.0.2
    sphinxcontrib-devhelp==1.0.2
    sphinxcontrib-htmlhelp==2.0.0
    sphinxcontrib-jsmath==1.0.1
    sphinxcontrib-qthelp==1.0.3
    sphinxcontrib-serializinghtml==1.1.5
    sphinx-rtd-theme==1.3.0
      ],
    classifiers=[
    'Development Status :: 3 - Alpha',      
    'Intended Audience :: Developers',      
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',  
    'Programming Language :: Python :: 3'
      ],
)

.. evovaq documentation master file, created by
   sphinx-quickstart on Tue Nov 14 14:24:43 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

EVOVAQ's documentation
======================
EVOlutionary algorithms-based toolbox for VAriational Quantum circuits (EVOVAQ) is a novel evolutionary framework designedmto easily train variational quantum circuits through evolutionary techniques, and to have a simple interface between these algorithms and quantum libraries, such as Qiskit.

**Optimizers in EVOVAQ:**

* Genetic Algorithm

* Differential Evolution

* Memetic Algorithm

* Big Bang Big Crunch

* Particle Swarm Optimization

* CHC Algorithm

* Hill Climbing

Installation
======================
You can install EVOVAQ via ``pip``:

.. code-block:: bash

  pip install evovaq

Pip will handle all dependencies automatically and you will always install the latest version.

Credits
======================
If you use EVOVAQ in your work, please cite the following paper:

BibTeX Citation
^^^^^^^^^^^^^^^

.. code-block:: bibtex
   @article{evovaq,
     title={EVOVAQ: EVOlutionary algorithms-based toolbox for VAriational Quantum circuits},
     author={Acampora, Giovanni and Guti{\'e}rrez, Carlos Cano and Chiatto, Angela and Hidalgo, Jos{\'e} Manuel Soto and Vitiello, Autilia},
     journal={SoftwareX},
     volume={26},
     pages={101756},
     year={2024},
     publisher={Elsevier}
   }

.. toctree::
   :hidden:
   :caption: Tutorials

   tutorials_trainVQCs

.. toctree::
   :hidden:
   :caption: API Guide

   problem
   algorithms
   tools


Indices
******************

* :ref:`genindex`
* :ref:`modindex`

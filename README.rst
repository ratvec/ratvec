RatVec
======
This tool generates low-dimensional, continuous, distributed vector representations for non-numeric entities such as
text or biological sequences (e.g. DNA or proteins) via kernel PCA with rational kernels.

The current implementation accepts any input dataset that can be read as a list of strings.

Installation |pypi_version| |python_versions| |pypi_license|
------------------------------------------------------------
RatVec can be installed on Python 3.6+ from `PyPI <https://pypi.python.org/pypi/ratvec>`_ with the following code in
your favorite terminal:

.. code-block:: sh

    $ pip install ratvec

or from the latest code on `GitHub <https://github.com/ratvec/ratvec>`_ with:

.. code-block:: sh

   $ pip install git+https://github.com/ratvec/ratvec.git

It can be installed in development mode with:

.. code-block:: sh

   $ git clone https://github.com/ratvec/ratvec.git
   $ cd ratvec
   $ pip install -e .

The ``-e`` dynamically links the code in the git repository to the Python site-packages so your changes get
reflected immediately.

How to Use
----------
``ratvec`` automatically installs a command line interface. Check it out with:

.. code-block:: sh

   $ ratvec --help

RatVec has three main commands: ``generate``, ``train``, and ``evaluate``:

1. **Generate**. Downloads and prepare the SwissProt data set that is showcased in the RatVec paper.

.. code-block:: sh

   $ ratvec generate

2. **Train**. Compute KPCA embeddings on a given data set. Please run the following command to see the arguments:

.. code-block:: sh

   $ ratvec train --help

3. **Evaluate**. Evaluate and optimize KPCA embeddings. Please run the following command to see the arguments:

.. code-block:: sh

   $ ratvec evaluate --help

Showcase Dataset
----------------
The application presented in the paper (SwissProt dataset [1]_ used by Boutet *et al.* [2]_) can be downloaded directly
from `here <https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/JMFHTN>`_ or running the following
command:

.. code-block:: sh

   $ ratvec generate

References
----------
.. [1] Boutet, E. *et al.* (2016). `UniProtKB/Swiss-Prot, the manually annotated section of the UniProt KnowledgeBase:
   how to use the entry view. <https://doi.org/10.1007/978-1-4939-3167-5_2>`_. Plant Bioinformatics (pp. 23-54).

.. [2] Asgari, E., & Mofrad, M. R. (2015). `Continuous distributed representation of biological sequences for deep
   proteomics and genomics <https://doi.org/10.1371/journal.pone.0141287>`_. PloS one, 10(11), e0141287.


.. |python_versions| image:: https://img.shields.io/pypi/pyversions/ratvec.svg
    :alt: Python versions supported by RatVec

.. |pypi_version| image:: https://img.shields.io/pypi/v/ratvec.svg
    :alt: Current version of RatVec on PyPI

.. |pypi_license| image:: https://img.shields.io/pypi/l/ratvec.svg
    :alt: RatVec is distributed under the Apache 2.0 License

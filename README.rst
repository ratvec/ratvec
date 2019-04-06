RatVec
======
This tool generates low-dimensional vector representations for non-numeric entities such as text or biological sequences (e.g. DNA or proteins) via kernel PCA with rational kernels. 

The current implementation accepts any input dataset that can be read as a list of strings.

Installation
------------
Install directly from the source with:

.. code-block:: bash

   $ pip install git+https://github.com/ratvec/ratvec.git

Install in development mode with:

.. code-block:: bash

   $ git clone https://github.com/ratvec/ratvec.git
   $ cd ratvec
   $ pip install -e .

The `-e` dynamically links the code in the git repository to the Python site-packages so your changes get
reflected immediately.

How to Use
----------

``ratvec`` is automatically installs a command line interface. Check it out with

.. code-block:: bash

   $ ratvec --help

RatVec has four main commands: ``generate``, ``train``, ``evaluate`` and ``optimize``:

1. **Generate**. Downloads and prepare the SwissProt data set that is showcased in the RatVec paper.

.. code-block:: bash

   $ ratvec generate

2. **Train**. Compute KPCA embeddings on a given data set. Please run the following command to see the arguments:

.. code-block:: bash

   $ ratvec train --help


3. **Evaluate**. Evaluate and optimize KPCA embeddings. Please run the following command to see the arguments:

.. code-block:: bash

   $ ratvec evaluate --help


4. **Optimize**. Evaluate and optimize KPCA embeddings. Please run the following command to see the arguments:

.. code-block:: bash

   $ ratvec optimize --help


Showcase Dataset
----------------

The application presented in the paper (SwissProt dataset [1] used by Boutet *et al.* [2]) can be downloaded directly from
the following website https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/JMFHTN or by simply
running the ``generate`` command:

.. code-block:: bash

   $ ratvec generate

References
----------

.. [1] Boutet, E. *et al.* (2016). `UniProtKB/Swiss-Prot, the manually annotated section of the UniProt KnowledgeBase:
   how to use the entry view. <https://doi.org/10.1007/978-1-4939-3167-5_2>`_. Plant Bioinformatics (pp. 23-54).

.. [2] Asgari, E., & Mofrad, M. R. (2015). `Continuous distributed representation of biological sequences for deep
   proteomics and genomics <https://doi.org/10.1371/journal.pone.0141287>`_. PloS one, 10(11), e0141287.

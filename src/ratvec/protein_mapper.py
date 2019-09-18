# -*- coding: utf-8 -*-

"""Mapping protein identifiers across databases.
The methods in this module return dictionaries mappings the protein ids in the ExCAPE-DB file (i.e., ENTREZ and Gene Symbols)
to UniProt identifiers (ids used in the SwissProt dataset).
"""

import pandas as pd
from bio2bel_hgnc import Manager

if __name__ == '__main__':
    manager = Manager()

    mapping_dict = manager.build_hgnc_symbol_uniprot_ids_mapping()

    df = pd.DataFrame.from_dict(mapping_dict, ['hgnc_symbol', 'uniprot_id'])
    df.to_csv('protein_mapping.csv')

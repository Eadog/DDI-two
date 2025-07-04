from transformers import AutoTokenizer
from typing import List, Tuple, Union
import os
import pandas as pd
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
import torch.nn.functional as F
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import torch
import collections
import logging
import os
import re
import codecs
import unicodedata
from typing import List, Optional
from transformers import PreTrainedTokenizer
from SmilesPE.tokenizer import SPE_Tokenizer
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdDesc
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem import BRICS
import numpy as np
from itertools import chain, repeat, islice
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.utils as utils
from rdkit.Chem import AllChem
from rdkit import DataStructs
from transformers import AutoTokenizer, AutoModel
import torch


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab

def pad_infinite(iterable, padding=None):
   return chain(iterable, repeat(padding))

def pad(iterable, size, padding=None):
   return islice(pad_infinite(iterable, padding), size)


# Atom feature sizes
MAX_ATOMIC_NUM = 100
ATOM_FEATURES = {
    'atomic_num': list(range(MAX_ATOMIC_NUM)),
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'chiral_tag': [0, 1, 2, 3],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}

# Distance feature sizes
PATH_DISTANCE_BINS = list(range(10))
THREE_D_DISTANCE_MAX = 20
THREE_D_DISTANCE_STEP = 1
THREE_D_DISTANCE_BINS = list(range(0, THREE_D_DISTANCE_MAX + 1, THREE_D_DISTANCE_STEP))

# len(choices) + 1 to include room for uncommon values; + 2 at end for IsAromatic and mass
ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values()) + 2
BOND_FDIM = 14

# Memoization
SMILES_TO_GRAPH = {}

def onek_encoding_unk(value: int, choices: List[int]) -> List[int]:
    """
    Creates a one-hot encoding.

    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the value in a list of length len(choices) + 1.
    If value is not in the list of choices, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding

def atom_features(atom: Chem.rdchem.Atom, functional_groups: List[int] = None) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for an atom.

    :param atom: An RDKit atom.
    :param functional_groups: A k-hot vector indicating the functional groups the atom belongs to.
    :return: A list containing the atom features.
    """
    features = onek_encoding_unk(atom.GetAtomicNum() - 1, ATOM_FEATURES['atomic_num']) + \
           onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
           onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) + \
           onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag']) + \
           onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) + \
           onek_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) + \
           [1 if atom.GetIsAromatic() else 0] + \
           [atom.GetMass() * 0.01]  # scaled to about the same range as other features
    if functional_groups is not None:
        features += functional_groups
    return features

def bond_features(bond: Chem.rdchem.Bond) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for a bond.

    :param bond: A RDKit bond.
    :return: A list containing the bond features.
    """
    if bond is None:
        fbond = [1] + [0] * (BOND_FDIM - 1)
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0)
        ]
        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
    return fbond

def extract_graph_and_subgraphs_from_smile(molecule_smile, k_hop=2, use_edge_attr=False):
    
    molecule = Chem.MolFromSmiles(molecule_smile)
    if molecule is None:
        print(f"[Invalid SMILES] {molecule_smile}")

    molecule = Chem.AddHs(molecule)
    

    try:
        Chem.SanitizeMol(molecule)

        for atom in molecule.GetAtoms():
            atom.UpdatePropertyCache()

        #mol = Chem.AddHs(molecule)
        AllChem.EmbedMolecule(molecule)
        AllChem.UFFOptimizeMolecule(molecule)
        conformer = molecule.GetConformer()
        pos = conformer.GetPositions()  #
        # mol = Chem.RemoveHs(molecule)
        # conformer = mol.GetConformer()

    except (RuntimeError, ValueError):

        print(f"Conformer generation failed: {molecule_smile}")
        AllChem.Compute2DCoords(molecule)
        conformer = molecule.GetConformer()
        pos = conformer.GetPositions()

    # Extract node features and edge features from the molecule
    features = rdDesc.GetFeatureInvariants(molecule)
    stereo = Chem.FindMolChiralCenters(molecule)
    chiral_centers = [0] * molecule.GetNumAtoms()
    for i in stereo:
        chiral_centers[i[0]] = i[1]

    node_features = []
    edge_features = []
    bonds = []
    for i in range(molecule.GetNumAtoms()):
        atom_i = molecule.GetAtomWithIdx(i)
        atom_i_features = atom_features(atom_i)
        node_features.append(atom_i_features)

        for j in range(molecule.GetNumAtoms()):
            bond_ij = molecule.GetBondBetweenAtoms(i, j)
            if bond_ij is not None:
                bonds.append([i, j])
                bond_features_ij = bond_features(bond_ij)
                edge_features.append(bond_features_ij)

    num_nodes = molecule.GetNumAtoms()
    
    # Convert bonds to a PyTorch tensor for use with k-hop subgraph extraction
    edge_index = torch.tensor(bonds, dtype=torch.long).T

    if edge_index.numel() == 0 or edge_index.shape[0] != 2:
        print(f"[No edges] {molecule_smile}")
            

    # Initialize subgraph storage
    subgraph_node_index = []
    subgraph_edge_index = []
    subgraph_indicator_index = []
    subgraph_edge_attr = [] if use_edge_attr else None
    edge_index_start = 0

    map = [0] * num_nodes

    # Extract k-hop subgraphs for each node
    for node_idx in range(num_nodes):
        try:         
            sub_nodes, sub_edge_index, _, edge_mask = utils.k_hop_subgraph(
            node_idx, k_hop, edge_index,
            relabel_nodes=True, num_nodes=num_nodes
            )
        except Exception as e:
            print(f"[Subgraph error] SMILES: {molecule_smile}, node: {node_idx}, error: {e}")
            return None
        
        
        # Store subgraph information
        subgraph_node_index.append(sub_nodes)
        subgraph_edge_index.append(sub_edge_index + edge_index_start)
        subgraph_indicator_index.append(torch.full((sub_nodes.shape[0],), node_idx))

        for sub_node in sub_nodes:
            map[sub_node] = node_idx


        # Extract edge features for the subgraph if needed
        if use_edge_attr:
            subgraph_edge_attr.append(edge_features[edge_mask])

        edge_index_start += len(sub_nodes)

    return node_features, bonds, edge_features, num_nodes, subgraph_node_index, subgraph_edge_index, subgraph_indicator_index, subgraph_edge_attr, pos, map

def get_chemberta_features(smiles):
    tokenizer = AutoTokenizer.from_pretrained("ChemBERTa-77M-MLM")
    tokenizer_config = tokenizer.init_kwargs
    model = AutoModel.from_pretrained("ChemBERTa-77M-MLM")

    inputs = tokenizer(smiles, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    features = outputs.last_hidden_state[:, 0, :].squeeze() 
    #features = features.cpu().numpy()
    return features

def create_dataset(datasets):
   
    compound_iso_smiles = []
    df = pd.read_csv('data/preprocessing/{}.csv'.format(datasets))
    compound_iso_smiles += list(df['Drug_ASMILES'])
    compound_iso_smiles += list(df['Drug_BSMILES'])
    compound_iso_smiles = set(compound_iso_smiles)
    smile_graph = {}
    smile_xulie = {}
    
    for smile in compound_iso_smiles:
        g = extract_graph_and_subgraphs_from_smile(smile)
        if g is None:
            print("g is kong")
            continue 
        smile_graph[smile] = g
        xulie1 = get_chemberta_features(smile)
        smile_xulie[smile] = xulie1
        

    label = df['label']
    df = pd.read_csv('data/preprocessing/' + datasets + '.csv')
    drug1, drug2 = list(df['Drug_ASMILES']), list(df['Drug_BSMILES'])
    drug1, drug2 = np.asarray(drug1), np.asarray(drug2)
    DDIDataset(root='data/', dataset=datasets + '_drug1', xd=drug1, y=label, smile_graph=smile_graph, xulie = smile_xulie)
    DDIDataset(root='data/', dataset=datasets + '_drug2', xd=drug2, y=label, smile_graph=smile_graph, xulie = smile_xulie)
    print("创建数据成功")


class DDIDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='_drug1', xd=None, y=None, smile_graph=None, xulie =None, transform=None, pre_transform=None):
        self.dataset = dataset
        # root is required for save preprocessed data, default is '/tmp'
        super(DDIDataset, self).__init__(root, transform, pre_transform)
        
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, y, smile_graph, xulie)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, xd, y, smile_graph, xulie):
        assert (len(xd) == len(y)), "The lists must be the same length!"
        data_list = []
        data_len = len(xd)
        print('number of data', data_len)
        for i in range(data_len):
            smiles = xd[i]
            labels = y[i]
            if smiles not in smile_graph:
                print(f"[Skipped] No graph for: {smiles}")
                continue
            if smiles not in xulie:
                print(f"[Skipped] No sequence embedding for: {smiles}")
                continue

            smilesss = xulie[smiles]
            node_features, bonds, edge_features, num_nodes, subgraph_node_index, subgraph_edge_index, subgraph_indicator_index, subgraph_edge_attr, pos, map = smile_graph[smiles]
            GCNData = Data(
                        x = torch.tensor(node_features, dtype = torch.float),
                        edge_index = torch.tensor(bonds, dtype = torch.long).T,
                        edge_attr = torch.tensor(edge_features, dtype = torch.float),
                        xulie = torch.FloatTensor(smilesss).unsqueeze(0),
                        pos = torch.tensor(pos, dtype=torch.float32),
                        map = torch.LongTensor(map),
                        
                        y=torch.Tensor([labels]),
                        subgraph_node_index = torch.cat(subgraph_node_index, dim=0),
                        subgraph_edge_index = torch.cat(subgraph_edge_index, dim=1),
                        subgraph_indicator_index = torch.cat(subgraph_indicator_index, dim=0),
                        )
            
            data_list.append(GCNData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        print('构建完成. Saving to file.')
        torch.save(self.collate(data_list), self.processed_paths[0])

if __name__ == "__main__":
    create_dataset('DDinter_2')
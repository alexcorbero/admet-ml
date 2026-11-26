"""Smoke test for the RDKit molecule to PyG batch integration path."""

import unittest


try:
    import torch
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from torch_geometric.data import Batch

    from src.gnn.config import GNNConfig
    from src.gnn.graph_builder import build_graph
except ModuleNotFoundError:
    torch = None
    Chem = None
    AllChem = None
    Batch = None
    GNNConfig = None
    build_graph = None


@unittest.skipIf(
    Batch is None,
    "RDKit, PyTorch, and PyTorch Geometric are required for the graph smoke test.",
)
class GraphSmokeTests(unittest.TestCase):
    def test_rdkit_molecule_to_data_to_batch(self):
        mol = Chem.AddHs(Chem.MolFromSmiles("CO"))
        embed_status = AllChem.EmbedMolecule(
            mol,
            randomSeed=0xF00D,
            useRandomCoords=False,
        )
        self.assertEqual(embed_status, 0)

        config = GNNConfig()
        graph_a = build_graph(
            mol,
            conf_id=0,
            config=config,
            y=-1.25,
            smiles="CO",
            mol_id="methanol-a",
        )
        graph_b = build_graph(
            mol,
            conf_id=0,
            config=config,
            y=-0.75,
            smiles="CO",
            mol_id="methanol-b",
        )

        self.assertEqual(graph_a.num_nodes, mol.GetNumAtoms())
        self.assertEqual(tuple(graph_a.pos.shape), (mol.GetNumAtoms(), 3))
        self.assertEqual(tuple(graph_a.y.shape), (1,))
        self.assertEqual(tuple(graph_a.conf_id.shape), (1,))
        self.assertTrue(torch.isfinite(graph_a.pos).all())

        batch = Batch.from_data_list([graph_a, graph_b])

        self.assertEqual(batch.num_graphs, 2)
        self.assertEqual(tuple(batch.y.shape), (2,))
        self.assertEqual(tuple(batch.conf_id.shape), (2,))
        self.assertEqual(tuple(batch.batch.shape), (2 * mol.GetNumAtoms(),))
        self.assertEqual(batch.smiles, ["CO", "CO"])
        self.assertEqual(batch.mol_id, ["methanol-a", "methanol-b"])
        self.assertTrue(torch.isfinite(batch.pos).all())


if __name__ == "__main__":
    unittest.main()

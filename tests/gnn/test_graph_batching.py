"""Integration checks for batching the raw graph contract with PyG."""

import unittest


try:
    import torch
    from torch_geometric.data import Batch, Data
except ModuleNotFoundError:
    torch = None
    Batch = None
    Data = None


@unittest.skipIf(
    Batch is None,
    "PyTorch and PyTorch Geometric are required for the batching check.",
)
class GraphBatchingTests(unittest.TestCase):
    def test_batch_concatenates_graphs_and_offsets_edges(self):
        graph_a = Data(
            z=torch.tensor([6, 8], dtype=torch.long),
            x=torch.ones((2, 3), dtype=torch.float32),
            x_chem=torch.ones((2, 2), dtype=torch.float32),
            x_dg=torch.ones((2, 1), dtype=torch.float32),
            pos=torch.zeros((2, 3), dtype=torch.float32),
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            edge_attr=torch.ones((2, 2), dtype=torch.float32),
            y=torch.tensor([-1.25], dtype=torch.float32),
            conf_id=torch.tensor([7], dtype=torch.long),
            smiles="CO",
            mol_id="a",
        )
        graph_b = Data(
            z=torch.tensor([6, 6, 8], dtype=torch.long),
            x=torch.full((3, 3), 2.0, dtype=torch.float32),
            x_chem=torch.full((3, 2), 2.0, dtype=torch.float32),
            x_dg=torch.full((3, 1), 2.0, dtype=torch.float32),
            pos=torch.ones((3, 3), dtype=torch.float32),
            edge_index=torch.tensor(
                [[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long
            ),
            edge_attr=torch.full((4, 2), 2.0, dtype=torch.float32),
            y=torch.tensor([-0.5], dtype=torch.float32),
            conf_id=torch.tensor([11], dtype=torch.long),
            smiles="CCO",
            mol_id="b",
        )

        batch = Batch.from_data_list([graph_a, graph_b])

        self.assertEqual(batch.num_graphs, 2)
        self.assertEqual(tuple(batch.z.shape), (5,))
        self.assertEqual(tuple(batch.x.shape), (5, 3))
        self.assertEqual(tuple(batch.x_chem.shape), (5, 2))
        self.assertEqual(tuple(batch.x_dg.shape), (5, 1))
        self.assertEqual(tuple(batch.pos.shape), (5, 3))
        self.assertEqual(tuple(batch.edge_index.shape), (2, 6))
        self.assertEqual(tuple(batch.edge_attr.shape), (6, 2))
        self.assertEqual(tuple(batch.y.shape), (2,))
        self.assertEqual(tuple(batch.conf_id.shape), (2,))

        torch.testing.assert_close(
            batch.edge_index,
            torch.tensor(
                [[0, 1, 2, 3, 3, 4], [1, 0, 3, 2, 4, 3]],
                dtype=torch.long,
            ),
        )
        torch.testing.assert_close(
            batch.batch,
            torch.tensor([0, 0, 1, 1, 1], dtype=torch.long),
        )
        torch.testing.assert_close(
            batch.y,
            torch.tensor([-1.25, -0.5], dtype=torch.float32),
        )
        torch.testing.assert_close(
            batch.conf_id,
            torch.tensor([7, 11], dtype=torch.long),
        )
        self.assertEqual(batch.smiles, ["CO", "CCO"])
        self.assertEqual(batch.mol_id, ["a", "b"])


if __name__ == "__main__":
    unittest.main()

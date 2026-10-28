"""Save a mesh dictionary and reload its geometry and node sets."""
import argparse
from pathlib import Path
from torch_fea_mesh.generation.mesh_example import create_quad_grid_mesh
from torch_fea_mesh.mesh.quad_mesh import QuadMesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'data/example/mesh_io_output')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = create_quad_grid_mesh(4,4)
    path = args.output/'mesh_io_quad.pt'
    source.save_as_torch(path)
    restored = QuadMesh()
    restored.load_from_torch(path)
    print('Loaded:', path, 'nodes:',len(restored.node), 'node sets:',list(restored.node_set))


if __name__ == '__main__':
    main()

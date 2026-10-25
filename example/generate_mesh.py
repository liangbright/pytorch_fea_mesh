"""Generate structured quad and hex grids and save them locally."""
import argparse
from pathlib import Path
from torch_fea_mesh.generation.mesh_example import create_quad_grid_mesh, create_hex_grid_mesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('data/example'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, mesh in [('quad_grid',create_quad_grid_mesh(4,5)), ('hex_grid',create_hex_grid_mesh(4,5,3))]:
        mesh.save_as_torch(args.output/(name+'.pt'))
        mesh.save_as_vtk(str(args.output/(name+'.vtk')))
        print(name, 'nodes:',len(mesh.node), 'elements:',len(mesh.element))


if __name__ == '__main__':
    main()

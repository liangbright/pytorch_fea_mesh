"""Remesh a synthetic curved triangle patch with a fixed boundary."""
import argparse
from pathlib import Path
import torch
from torch_fea_mesh.mesh.triangle_mesh import TriangleMesh
from torch_fea_mesh.processing.isotropic_triangle_surface_remesher import isotropic_triangle_surface_remesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('data/example'))
    parser.add_argument('--iterations', type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    n = 4
    nodes = [[i,j,.05*(i*i+j*j)] for j in range(n) for i in range(n)]
    faces = []
    for j in range(n-1):
        for i in range(n-1):
            a = j*n+i
            faces.extend([[a,a+1,a+n+1],[a,a+n+1,a+n]])
    source = TriangleMesh(torch.tensor(nodes,dtype=torch.float64),faces)
    output = isotropic_triangle_surface_remesh(source,.7,False,max_iter=args.iterations)
    source.save_as_vtk(str(args.output/'source.vtk'))
    output.save_as_vtk(str(args.output/'remeshed.vtk'))
    print(output.mesh_data['remeshing_report'])


if __name__ == '__main__':
    main()

"""Compute triangle area, normals, and boundary nodes."""
import torch
from torch_fea_mesh.mesh.triangle_mesh import TriangleMesh


def main():
    mesh = TriangleMesh(torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]], dtype=torch.float64), [[0,1,2]])
    mesh.update_element_area_and_normal()
    print('Area:', mesh.element_area)
    print('Normal:', mesh.element_normal)
    print('Boundary nodes:', mesh.find_boundary_node())


if __name__ == '__main__':
    main()

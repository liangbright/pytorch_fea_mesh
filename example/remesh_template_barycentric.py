"""Map a quad template to a curved triangle patch; optionally fix a landmark.

Requires the remeshing and vtk extras.
"""
import argparse
from pathlib import Path
import torch
from torch_fea_mesh.mesh.polygon_mesh import PolygonMesh
from torch_fea_mesh.processing.template_barycentric_mapping_based_remesher import (
    template_barycentric_mapping_based_remesh,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'data/example/template_barycentric_output')
    parser.add_argument('--landmark', action='store_true')
    args = parser.parse_args()
    source = PolygonMesh(torch.tensor([
        [0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.], [.4, .6, .3]],
        dtype=torch.float64), [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
    template = PolygonMesh(torch.tensor([
        [0., 0., 0.], [.5, 0., 0.], [1., 0., 0.],
        [0., .5, 0.], [.5, .5, 0.], [1., .5, 0.],
        [0., 1., 0.], [.5, 1., 0.], [1., 1., 0.]], dtype=torch.float64),
        [[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6], [4, 5, 8, 7]])
    output = template_barycentric_mapping_based_remesh(
        source, [[0, 1], [1, 2], [2, 3], [3, 0]],
        template, [[0, 1, 2], [2, 5, 8], [8, 7, 6], [6, 3, 0]],
        landmark_of_source_mesh=[4] if args.landmark else None,
        landmark_of_template_mesh=[4] if args.landmark else None)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, mesh in [('source', source), ('template', template), ('remeshed', output)]:
        mesh.save_as_vtk(str(args.output / f'template_barycentric_{name}.vtk'))
    report = output.mesh_data['template_barycentric_mapping_based_remesh']
    print('Selected diffusion coefficient:', report['auto_diffusion']['selected_coefficient'])
    print('Stretch history:', report['stretch_history'])


if __name__ == '__main__':
    main()

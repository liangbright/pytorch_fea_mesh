"""Deterministic curved examples; all meshes use the user's TriangleMesh class.

Each builder returns a mesh and metadata in the original numbering. Example
files are written by the remeshing test suite through the mesh library's VTK I/O.
"""
import numpy as np
import torch
from torch_fea_mesh.mesh.triangle_mesh import TriangleMesh


def curved_disk(height=1.2, exterior=True):
    """Three triangles spanning a triangular cycle, bent out of its plane."""
    nodes = [[-1,0,0],[1,0,0],[0,3,0],[0,1,height]]
    faces = [(0,1,3),(1,2,3),(2,0,3)]
    if exterior:
        nodes.append([2,1,-.4])
        faces.append((2,1,4))
    return TriangleMesh(torch.tensor(nodes,dtype=torch.float64),faces), dict(
        name='curved_disk', edge=(0,1), cycle=(0,1,2), expected_enclosed=set(range(3)),
        expected_exterior={3} if exterior else set(), feature_points=[], target=4.)


def nested_curved_disk(n=8, exterior=True, chords=True):
    """A curved fan plus N-1 redundant UV chords, including the outer UVW face.

    There are 2*N-1 support triangles and N-1 chord triangles. Every w_i belongs
    to the support disk. With an exterior annulus, u and v are interior vertices;
    the default remesher can exercise the real late-collapse path without mocks.
    """
    if n < 2:
        raise ValueError('n must be at least two')
    nodes = [[-.1,0,0],[.1,0,0]]
    nodes += [[.08*np.sin(4*t), .25+.75*t, .45*np.sin(2.4*t)+.12*t]
              for t in np.linspace(0,1,n)]
    faces = [(0,1,2)]
    for w in range(2,n+1):
        faces.extend([(0,w,w+1),(1,w+1,w)])
    support = set(range(len(faces)))
    if chords:
        faces.extend((0,1,w) for w in range(3,n+2))
    enclosed = set(range(len(faces)))
    outside = set()
    if exterior:
        a = len(nodes)
        nodes.extend([[-1.4,-1.4,0],[1.4,-1.4,0],[1.4,1.4,0],[-1.4,1.4,0]])
        A,B,C,D = range(a,a+4)
        W = n+1
        outside = set(range(len(faces),len(faces)+7))
        faces.extend([(A,B,1),(A,1,0),(A,0,D),(D,0,W),(D,W,C),(C,W,1),(C,1,B)])
    mesh = TriangleMesh(torch.tensor(nodes,dtype=torch.float64),faces)
    return mesh, dict(name=f'nested_curved_{n}',edge=(0,1),cycle=(0,1,n+1),
                     expected_enclosed=enclosed,expected_exterior=outside,
                     support=support, feature_points=list(range(2,n+2)), target=2.)


def folded_disk():
    """Embedded disk whose interior apex projects OUTSIDE the boundary triangle."""
    mesh, metadata = curved_disk(height=5.)
    mesh.node[3,0] = 3.5
    return mesh, dict(metadata, name='folded_disk')


def repairable_curved_patch():
    """A short nonfacial link obstruction that subdivision can repair accurately."""
    nodes = np.array([[0,0,0],[4,0,0],[4,4,0],[0,4,0],
                      [1.95,2,0],[2.05,2,0],[2.3,3,0],[2.11,2.3,0]],dtype=float)
    nodes[:,2] = .05*nodes[:,0]*nodes[:,1]
    faces = [(0,1,5),(0,5,4),(0,4,3),(3,4,6),(3,6,2),(2,6,5),(2,5,1),
             (4,5,7),(5,6,7),(6,4,7)]
    return TriangleMesh(torch.from_numpy(nodes),faces), dict(name='repairable_curved',
        edge=(4,5),cycle=(4,5,6),expected_enclosed={7,8,9},
        expected_exterior=set(range(7)),feature_points=[],target=.2)


def ambiguous_two_sided_disk():
    """Two caps with one boundary; the historical name predates paired deletion."""
    nodes = [[-1,0,0],[1,0,0],[0,3,0],[0,1,1.2],[0,1,-.8]]
    top = [(0,1,3),(1,2,3),(2,0,3)]
    bottom = [(1,0,4),(2,1,4),(0,2,4)]
    return TriangleMesh(torch.tensor(nodes,dtype=torch.float64),top+bottom), dict(
        name='ambiguous_two_sided',edge=(0,1),cycle=(0,1,2),expected_enclosed=set(range(6)),
        expected_exterior=set(),feature_points=[],target=4.)


def curved_disk_with_hole():
    """Triangular annulus: the extra boundary prevents disk certification."""
    nodes = [[-1,0,0],[1,0,0],[0,3,0],[-.3,.6,.7],[.3,.6,.7],[0,1.8,1.1]]
    faces = [(0,1,4),(0,4,3),(1,2,5),(1,5,4),(2,0,3),(2,3,5)]
    return TriangleMesh(torch.tensor(nodes,dtype=torch.float64),faces), dict(
        name='curved_hole',edge=(0,1),cycle=(0,1,2),expected_enclosed=None,
        expected_exterior=set(),feature_points=[],target=4.)


def rotated_example(builder, *args, **kwargs):
    """An orientation-independent check, including translation and change of scale."""
    mesh, metadata = builder(*args, **kwargs)
    q, _ = np.linalg.qr(np.array([[1.,2.,3.],[4.,-1.,2.],[2.,3.,-4.]]))
    if np.linalg.det(q) < 0:
        q[:,0] *= -1
    mesh.node = mesh.node @ torch.from_numpy(q) * 2.5 + torch.tensor([11.,-7.,5.])
    metadata = dict(metadata, name=metadata['name']+'_rotated', target=metadata['target']*2.5)
    return mesh, metadata

import torch
import torch_scatter
import numpy as np
from ..mesh.polyhedron_mesh import PolyhedronMesh
from ..mesh.tetrahedron_mesh import TetrahedronMesh
from ..mesh.hexahedron_mesh import HexahedronMesh
from ..mesh.tet10_mesh import Tet10Mesh
from ..mesh.polygon_mesh import PolygonMesh
from ..mesh.quad_mesh import QuadMesh
from ..mesh.triangle_mesh import TriangleMesh
from ..mesh.tri6_mesh import Tri6Mesh
#%%
def ExtractSurfaceElement_slow(mesh):
    #extract surface and make sure surface normal is from inside to outside
    if ((not isinstance(mesh, TetrahedronMesh)) 
        and (not isinstance(mesh, Tet10Mesh))
        and (not isinstance(mesh, HexahedronMesh))):
        print('mesh is ', type(mesh))
        raise NotImplementedError
    face_idx_list=mesh.find_boundary_face()
    adj_table=mesh.face_to_element_adj_table    
    surface_element=[]
    for n in range(0, len(face_idx_list)):
        face_idx=face_idx_list[n]
        fa=mesh.face[face_idx]
        if not isinstance(fa, list):
            fa=fa.tolist()
        fa=np.array([fa], dtype=np.int64)
        fa=np.sort(fa, axis=1)
        if len(adj_table[face_idx]) != 1:
            raise ValueError
        elm_idx=adj_table[face_idx][0]
        elm=mesh.element[elm_idx]
        if not isinstance(elm, list):
            elm=elm.tolist()
        face=[]
        if isinstance(mesh, TetrahedronMesh):
            id0, id1, id2, id3=elm
            face.append([id0, id2, id1])
            face.append([id0, id1, id3])
            face.append([id0, id3, id2])
            face.append([id1, id2, id3])
        elif isinstance(mesh, Tet10Mesh):
            id0, id1, id2, id3, id4, id5, id6, id7, id8, id9=elm
            face.append([id0, id2, id1, id6, id5, id4])
            face.append([id0, id1, id3, id4, id8, id7])
            face.append([id0, id3, id2, id7, id9, id6])
            face.append([id1, id2, id3, id5, id9, id8])
        elif isinstance(mesh, HexahedronMesh):
            id0, id1, id2, id3, id4, id5, id6, id7=elm
            face.append([id0, id3, id2, id1])
            face.append([id4, id5, id6, id7])
            face.append([id0, id1, id5, id4])
            face.append([id1, id2, id6, id5])
            face.append([id2, id3, id7, id6])
            face.append([id3, id0, id4, id7])
        else:
            raise ValueError
        face=np.array(face, dtype=np.int64)
        face_sorted=np.sort(face, axis=1)
        best_idx=np.abs((face_sorted-fa)).sum(axis=1).argmin().item()
        surface_element.append(face[best_idx].tolist())
    return surface_element
#%%
def extract_surface_element(mesh):
    #extract surface
    #surface normal is from inside to outside - this is ensured when each face is defined
    face_idx_list=mesh.find_boundary_face()
    surface_element=mesh.face[face_idx_list].tolist()
    return surface_element
#%%
ExtractSurfaceElement=extract_surface_element
#%%
def extract_surface_mesh(mesh):
    surface_element=extract_surface_element(mesh)
    try:
        surface_element=torch.tensor(surface_element, dtype=torch.int64)
        if surface_element.shape[1] == 3:
            temp_mesh=TriangleMesh(mesh.node, surface_element)
        elif surface_element.shape[1] == 4:
            temp_mesh=QuadMesh(mesh.node, surface_element)
        elif surface_element.shape[1] == 6:
            if isinstance(mesh, Tet10Mesh):
                temp_mesh=Tri6Mesh(mesh.node, surface_element)
            else:
                temp_mesh=PolygonMesh(mesh.node, surface_element)
        else:
            temp_mesh=PolygonMesh(mesh.node, surface_element)
    except:
        temp_mesh=PolygonMesh(mesh.node, surface_element)
    surface_mesh=temp_mesh.get_sub_mesh(torch.arange(0, len(surface_element)).tolist())
    return surface_mesh
#%%
ExtractSurfaceMesh=extract_surface_mesh

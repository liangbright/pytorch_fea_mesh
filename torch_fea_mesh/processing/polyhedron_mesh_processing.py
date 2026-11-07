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
#the nodes of each face, by the number of nodes in an element, with the normal from inside to outside
FACE_OF_ELEMENT={4:[[0,2,1], [0,1,3], [0,3,2], [1,2,3]],
                 8:[[0,3,2,1], [4,5,6,7], [0,1,5,4], [1,2,6,5], [2,3,7,6], [3,0,4,7]]}
#%%
def extract_surface_element_of_mixed_mesh(mesh):
    #for a plain PolyhedronMesh, which may mix tet4 and hex8 and has no face tables
    face_table={}
    for elm in mesh.make_element_copy('list'):
        if len(elm) not in FACE_OF_ELEMENT:
            raise NotImplementedError('a plain PolyhedronMesh supports tet4 and hex8, not '+str(len(elm))+' nodes')
        for idx_list in FACE_OF_ELEMENT[len(elm)]:
            face=[elm[idx] for idx in idx_list]
            key=tuple(sorted(face))
            #a face used by two elements is inside the mesh
            face_table[key]=None if key in face_table else face
    return [face for face in face_table.values() if face is not None]
#%%
def extract_surface_element(mesh):
    #extract surface
    #surface normal is from inside to outside - this is ensured when each face is defined
    if not hasattr(mesh, 'build_element_to_face_adj_table'):
        return extract_surface_element_of_mixed_mesh(mesh)
    face_idx_list=mesh.find_boundary_face()
    surface_element=mesh.face[face_idx_list].tolist()
    return surface_element
#%%
def extract_surface_mesh(mesh, return_node_idx_list=False):
    #if return_node_idx_list is True, also return node_idx_list:
    #surface_mesh.node[k] is mesh.node[node_idx_list[k]], which maps a load or a boundary condition to mesh
    surface_element=extract_surface_element(mesh)
    try:
        surface_element=torch.tensor(surface_element, dtype=torch.int64, device=mesh.node.device)
    except (ValueError, TypeError, RuntimeError):
        #faces of different sizes
        temp_mesh=PolygonMesh(mesh.node, surface_element)
    else:
        n_node=surface_element.shape[1] if len(surface_element.shape) == 2 else 0
        if n_node == 3:
            temp_mesh=TriangleMesh(mesh.node, surface_element)
        elif n_node == 4:
            temp_mesh=QuadMesh(mesh.node, surface_element)
        elif n_node == 6 and isinstance(mesh, Tet10Mesh):
            temp_mesh=Tri6Mesh(mesh.node, surface_element)
        else:
            temp_mesh=PolygonMesh(mesh.node, surface_element)
    surface_mesh, node_idx_list=temp_mesh.get_sub_mesh(torch.arange(0, len(surface_element)).tolist(), return_node_idx_list=True)
    if return_node_idx_list == True:
        return surface_mesh, node_idx_list
    return surface_mesh
#%%

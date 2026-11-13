import numpy as np
import torch
from torch.linalg import vector_norm as norm
from torch.linalg import cross
import torch_scatter
from torch_fea_mesh.mesh.polygon_mesh import PolygonMesh
from torch_fea_mesh.mesh.quad_mesh import QuadMesh
from torch_fea_mesh.mesh.triangle_mesh import TriangleMesh
#%%
class QuadTriangleMesh(PolygonMesh):
    #element could be quad or triangle

    def __init__(self, node=None, element=None):
        super().__init__(node=node, element=element)
        self.mesh_type='polygon_quad4_tri3'
        if element is not None:
            self.classify_element()

    def clear(self):
        super().clear()

    def clear_geometry_derived_data(self, clear_adjacency_data=True):
        super().clear_geometry_derived_data(clear_adjacency_data)
        self.node_normal=None
        self.element_area=None
        self.element_normal=None
        self.quad_element_corner_angle=None
        self.tri_element_corner_angle=None
        if clear_adjacency_data == True:
            #like the adjacency tables, these depend only on self.element
            self.quad_element=None
            self.quad_element_idx=None#index list of quad elements
            self.tri_element=None
            self.tri_element_idx=None#index list of tri elements

    def classify_element(self):
        self.node_normal=None
        self.element_area=None
        self.element_normal=None
        self.quad_element_corner_angle=None
        self.tri_element_corner_angle=None
        if self.element is None or len(self.element) == 0:
            self.quad_element=[]
            self.quad_element_idx=[]
            self.tri_element=[]
            self.tri_element_idx=[]
            return
        if isinstance(self.element,  torch.Tensor):
            if len(self.element.shape) == 2: 
                if len(self.element[0]) == 4:
                    self.quad_element=self.element
                    self.quad_element_idx=np.arange(0, len(self.element)).tolist()
                    self.tri_element=[]
                    self.tri_element_idx=[]
                elif len(self.element[0]) == 3:
                    self.tri_element=self.element
                    self.tri_element_idx=np.arange(0, len(self.element)).tolist()
                    self.quad_element=[]
                    self.quad_element_idx=[]
                else:
                    raise ValueError("len(self.element[0])="+str(len(self.element[0])))
                return
        quad_element=[]
        quad_element_idx=[]
        tri_element=[]
        tri_element_idx=[]
        for m in range(0, len(self.element)):
            elm=self.element[m]
            if torch.is_tensor(elm) or isinstance(elm, np.ndarray):
                elm=elm.tolist()
            if len(elm) == 4:
                quad_element.append(elm)
                quad_element_idx.append(m)
            elif len(elm) == 3:
                tri_element.append(elm)
                tri_element_idx.append(m)
            else:
                raise ValueError("len(elm)="+str(len(elm))+",m="+str(m))
        device=None if self.node is None else self.node.device
        self.quad_element=torch.tensor(quad_element, dtype=torch.int64, device=device)
        self.quad_element_idx=quad_element_idx
        self.tri_element=torch.tensor(tri_element, dtype=torch.int64, device=device)
        self.tri_element_idx=tri_element_idx

    def check_element_width(self, element):
        #an element must be a quad or a triangle, so that a mesh with another element is rejected before it changes
        if element is None:
            return
        found=PolygonMesh.find_element_width(element)
        if len(found-{3, 4}) > 0:
            raise ValueError('a QuadTriangleMesh element must have 3 or 4 nodes, but found '+str(sorted(found-{3, 4}, key=str)))

    def load_from_vtk(self, filename, dtype, return_mesh_vtk=False):
        return super().load_from_vtk(filename, dtype, return_mesh_vtk)
    
    def load_from_vtp(self, filename, dtype):
        super().load_from_vtp(filename, dtype)
    
    def load_from_stl(self, filename, dtype):
        super().load_from_stl(filename, dtype)

    def read_mesh_vtk(self, mesh_vtk, dtype):
        super().read_mesh_vtk(mesh_vtk, dtype)
        self.classify_element()

    def load_from_torch(self, filename, dtype=None):
        #load_from_torch has no width check, so a file with another element fails here: keep the mesh as it was
        state=dict(vars(self))
        super().load_from_torch(filename, dtype)
        try:
            self.classify_element()
        except Exception:
            vars(self).clear()
            vars(self).update(state)
            raise

    def copy(self, node, element, dtype=None, detach=True):
        super().copy(node, element, dtype, detach)
        self.classify_element()

    def update_node_normal(self, angle_weighted=True, eps=None, normal_source='corner'):
        #normal_source: the normal that each corner of a quad adds to its node, see QuadMesh.cal_node_normal
        #a triangle gives the same result with either one
        if normal_source not in ('corner', 'element'):
            raise ValueError("normal_source must be 'corner' or 'element', not "+repr(normal_source))
        if eps is None:
            if self.node.dtype in (torch.float32, torch.float64):
                eps=1e-12
            else:
                raise NotImplementedError()
        if self.quad_element is None or self.tri_element is None:
            self.classify_element()
        #-----------------------------------
        #each corner of each element adds to its node, as in QuadMesh.cal_node_normal and TriangleMesh.cal_node_normal.
        #quads and triangles are added together and normalized once, so a corner counts the same in either type.
        #on a mesh with only quads or only triangles, this gives the result of QuadMesh or TriangleMesh.
        node=self.node
        N=node.shape[0]
        normal=torch.zeros((N,3), dtype=node.dtype, device=node.device)
        if len(self.quad_element) > 0:
            element=self.quad_element.to(node.device)
            M=element.shape[0]
            if normal_source == 'corner':
                x0=node[element[:,0]]
                x1=node[element[:,1]]
                x2=node[element[:,2]]
                x3=node[element[:,3]]
                # x3--x2
                # |   |
                # x0--x1
                normal_quad=torch.cat([cross(x1-x0, x3-x0, dim=-1).reshape(M,1,3),
                                       cross(x2-x1, x0-x1, dim=-1).reshape(M,1,3),
                                       cross(x3-x2, x1-x2, dim=-1).reshape(M,1,3),
                                       cross(x0-x3, x2-x3, dim=-1).reshape(M,1,3)], dim=1)
                normal_quad=normal_quad/norm(normal_quad, ord=2, dim=-1, keepdim=True).clamp_min(eps)
            else:
                normal_quad=QuadMesh.cal_element_normal(node, element, eps=eps).reshape(M,1,3).expand(M,4,3)
            if angle_weighted == True:
                e_angle=QuadMesh.cal_element_corner_angle(node, element, eps=eps)#e_angle: (M,4)
                normal_quad=normal_quad*e_angle.reshape(M,4,1)
            normal=normal+torch_scatter.scatter(normal_quad.reshape(-1,3), element.reshape(-1), dim=0, dim_size=N, reduce="sum")
        if len(self.tri_element) > 0:
            element=self.tri_element.to(node.device)
            M=element.shape[0]
            normal_tri=TriangleMesh.cal_element_normal(node, element, eps=eps).reshape(M,1,3).expand(M,3,3)
            if angle_weighted == True:
                e_angle=TriangleMesh.cal_element_corner_angle(node, element, eps=eps)#e_angle: (M,3)
                normal_tri=normal_tri*e_angle.reshape(M,3,1)
            normal=normal+torch_scatter.scatter(normal_tri.reshape(-1,3), element.reshape(-1), dim=0, dim_size=N, reduce="sum")
        error=torch.isnan(normal).sum()
        if error > 0:
            print("error: nan in normal @ QuadTriangleMesh:update_node_normal")
        normal_norm=norm(normal, ord=2, dim=-1, keepdim=True)
        normal=normal/normal_norm.clamp_min(eps)
        normal=normal.contiguous()
        self.node_normal=normal
    
    @staticmethod
    def cal_node_normal(node, element, angle_weighted=True, eps=None, normal_source='corner'):
        temp=QuadTriangleMesh(node, element)
        temp.update_node_normal(angle_weighted=angle_weighted, eps=eps, normal_source=normal_source)
        return temp.node_normal

    def update_element_area_and_normal(self, eps=None):
        if self.quad_element is None or self.tri_element is None:
            self.classify_element()
        area=torch.zeros((len(self.element), 1), dtype=self.node.dtype, device=self.node.device)
        normal=torch.zeros((len(self.element), 3), dtype=self.node.dtype, device=self.node.device)
        if len(self.quad_element) > 0:
            area_quad, normal_quad=QuadMesh.cal_element_area_and_normal(self.node, self.quad_element, eps=eps)
            area[self.quad_element_idx]=area_quad
            normal[self.quad_element_idx]=normal_quad
        if len(self.tri_element) > 0:
            area_tri, normal_tri=TriangleMesh.cal_element_area_and_normal(self.node, self.tri_element, eps=eps)
            area[self.tri_element_idx]=area_tri
            normal[self.tri_element_idx]=normal_tri
        self.element_area=area
        self.element_normal=normal

    def update_element_normal(self, eps=None):
        if self.quad_element is None or self.tri_element is None:
            self.classify_element()
        normal=torch.zeros((len(self.element), 3), dtype=self.node.dtype, device=self.node.device)
        if len(self.quad_element) > 0:
            normal[self.quad_element_idx]=QuadMesh.cal_element_normal(self.node, self.quad_element, eps=eps)
        if len(self.tri_element) > 0:
            normal[self.tri_element_idx]=TriangleMesh.cal_element_normal(self.node, self.tri_element, eps=eps)
        self.element_normal=normal
        
    @staticmethod
    def cal_element_area_and_normal(node, element, eps=None):
        temp=QuadTriangleMesh(node, element)
        temp.update_element_area_and_normal(eps=eps)
        return temp.element_area, temp.element_normal

    @staticmethod
    def cal_element_normal(node, element, eps=None):
        temp=QuadTriangleMesh(node, element)
        temp.update_element_normal(eps=eps)
        return temp.element_normal

    def update_element_corner_angle(self, eps=None, return_cos=False):
        if self.quad_element is None or self.tri_element is None:
            self.classify_element()
        if len(self.quad_element) > 0:
            self.quad_element_corner_angle = QuadMesh.cal_element_corner_angle(self.node, self.quad_element, return_cos, eps=eps)
        if len(self.tri_element) > 0:
            self.tri_element_corner_angle = TriangleMesh.cal_element_corner_angle(self.node, self.tri_element, return_cos, eps=eps)
    
    @staticmethod
    def cal_element_corner_angle(node, element, return_cos=False, eps=None):
        temp=QuadTriangleMesh(node, element)
        temp.update_element_corner_angle(eps=eps, return_cos=return_cos)
        return temp.quad_element_corner_angle, temp.tri_element_corner_angle
        
    def subdivide(self):
        #return a new mesh
        if self.quad_element is None or self.tri_element is None:
            self.classify_element()
        #add a node in the middle of each edge
        if self.edge is None:
            self.build_edge()
        x_j=self.node[self.edge[:,0]]
        x_i=self.node[self.edge[:,1]]
        nodeA=(x_j+x_i)/2
        #add a node in the middle of each quad element
        n_nodeB=0
        if len(self.quad_element) > 0:
            nodeB=self.node[self.quad_element].mean(dim=1) #(N,3) => (M,4,3) => (M,3)
            n_nodeB=nodeB.shape[0]
        #create new mesh
        if n_nodeB > 0:
            node_new=torch.cat([self.node, nodeA, nodeB], dim=0)
        else:
            node_new=torch.cat([self.node, nodeA], dim=0)

        element_new=[]
        for m in range(0, len(self.quad_element)):
            #-----------
            # x3--x6--x2
            # |   |   |
            # x7--x8--x5
            # |   |   |
            # x0--x4--x1
            #-----------
            elm=self.quad_element[m]
            id0=int(elm[0])
            id1=int(elm[1])
            id2=int(elm[2])
            id3=int(elm[3])
            id4=self.node.shape[0]+self.get_edge_idx_from_node_pair(id0, id1)
            id5=self.node.shape[0]+self.get_edge_idx_from_node_pair(id1, id2)
            id6=self.node.shape[0]+self.get_edge_idx_from_node_pair(id2, id3)
            id7=self.node.shape[0]+self.get_edge_idx_from_node_pair(id0, id3)            
            id8=self.node.shape[0]+nodeA.shape[0]+m
            element_new.append([id0, id4, id8, id7])
            element_new.append([id4, id1, id5, id8])
            element_new.append([id7, id8, id6, id3])
            element_new.append([id8, id5, id2, id6])
            
        for m in range(0, len(self.tri_element)):
            #-----------
            #     x2
            #    /  \
            #   x5-- x4
            #  / \  / \
            # x0--x3--x1
            #-----------
            elm=self.tri_element[m]
            id0=int(elm[0])
            id1=int(elm[1])
            id2=int(elm[2])            
            id3=self.node.shape[0]+self.get_edge_idx_from_node_pair(id0, id1)
            id4=self.node.shape[0]+self.get_edge_idx_from_node_pair(id1, id2)
            id5=self.node.shape[0]+self.get_edge_idx_from_node_pair(id0, id2)    
            element_new.append([id0, id3, id5])
            element_new.append([id3, id4, id5])
            element_new.append([id3, id1, id4])
            element_new.append([id5, id4, id2])
            
        mesh_new=QuadTriangleMesh(node_new, element_new)
        return mesh_new

    def subdivide_to_quad(self):
        #return a new mesh
        if self.quad_element is None or self.tri_element is None:
            self.classify_element()
        #add a node in the middle of each edge
        if self.edge is None:
            self.build_edge()
        x_j=self.node[self.edge[:,0]]
        x_i=self.node[self.edge[:,1]]
        nodeA=(x_j+x_i)/2
        #add a node in the middle of each quad element
        n_nodeB=0
        if len(self.quad_element) > 0:
            nodeB=self.node[self.quad_element].mean(dim=1) #(N,3) => (M,4,3) => (M,3)
            n_nodeB=nodeB.shape[0]
        #add a node in the middle of each tri element
        n_nodeC=0
        if len(self.tri_element) > 0:
            nodeC=self.node[self.tri_element].mean(dim=1) #(N,3) => (M,3,3) => (M,3)
            n_nodeC=nodeC.shape[0]
        #create new mesh
        if n_nodeB > 0 and n_nodeC > 0:
            node_new=torch.cat([self.node, nodeA, nodeB, nodeC], dim=0)
        elif n_nodeB == 0 and n_nodeC > 0:
            node_new=torch.cat([self.node, nodeA, nodeC], dim=0)
        elif n_nodeB > 0 and n_nodeC == 0:
            node_new=torch.cat([self.node, nodeA, nodeB], dim=0)
        else: # n_nodeB == 0 and n_nodeC == 0:
            node_new=torch.cat([self.node, nodeA], dim=0)

        element_new=[]
        for m in range(0, len(self.quad_element)):
            #-----------
            # x3--x6--x2
            # |   |   |
            # x7--x8--x5
            # |   |   |
            # x0--x4--x1
            #-----------
            elm=self.quad_element[m]
            id0=int(elm[0])
            id1=int(elm[1])
            id2=int(elm[2])
            id3=int(elm[3])
            id4=self.node.shape[0]+self.get_edge_idx_from_node_pair(id0, id1)
            id5=self.node.shape[0]+self.get_edge_idx_from_node_pair(id1, id2)
            id6=self.node.shape[0]+self.get_edge_idx_from_node_pair(id2, id3)
            id7=self.node.shape[0]+self.get_edge_idx_from_node_pair(id0, id3)            
            id8=self.node.shape[0]+nodeA.shape[0]+m
            element_new.append([id0, id4, id8, id7])
            element_new.append([id4, id1, id5, id8])
            element_new.append([id7, id8, id6, id3])
            element_new.append([id8, id5, id2, id6])
            
        for m in range(0, len(self.tri_element)):
            #-----------
            #     x2
            #     /\
            #   x5  x4
            #   / \/ \
            #  /  x6  \
            # /   |    \
            #x0---x3---x1
            #-----------            
            elm=self.tri_element[m]
            id0=int(elm[0])
            id1=int(elm[1])
            id2=int(elm[2])
            id3=self.node.shape[0]+self.get_edge_idx_from_node_pair(id0, id1)
            id4=self.node.shape[0]+self.get_edge_idx_from_node_pair(id1, id2)
            id5=self.node.shape[0]+self.get_edge_idx_from_node_pair(id0, id2)
            id6=self.node.shape[0]+nodeA.shape[0]+n_nodeB+m
            element_new.append([id6, id5, id0, id3])
            element_new.append([id6, id3, id1, id4])
            element_new.append([id6, id4, id2, id5])
            
        mesh_new=QuadTriangleMesh(node_new, element_new)
        return mesh_new

    def get_sub_mesh(self, element_idx_list, return_node_idx_list=False):
        new_mesh, node_idx_list=super().get_sub_mesh(element_idx_list, return_node_idx_list=True)
        new_mesh=QuadTriangleMesh(new_mesh.node, new_mesh.element)
        if return_node_idx_list == False:
            return new_mesh
        else:
            return new_mesh, node_idx_list
#%%

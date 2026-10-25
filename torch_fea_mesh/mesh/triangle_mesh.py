import torch
from torch.linalg import vector_norm as norm
import torch_scatter
import numpy as np
from .polygon_mesh import PolygonMesh
from .quad_mesh import QuadMesh
from ..utility.geometry_utility import cal_angle_between_3d_vector
#%%
class TriangleMesh(PolygonMesh):
    #3-node triangle element mesh
    def __init__(self, node=None, element=None):
        super().__init__(node=node, element=element)
        self.mesh_type='polygon_tri3'
        if (node is not None) and (element is not None):
            if not self.is_tri():
                raise ValueError('not a triangle mesh')

    def clear(self):
        super().clear()

    def clear_geometry_derived_data(self):
        super().clear_geometry_derived_data()
        self.node_normal=None
        self.element_area=None
        self.element_normal=None
        self.element_corner_angle=None

    def update_node_normal(self, angle_weighted=True, eps=None):
        self.element_area, self.element_normal=TriangleMesh.cal_element_area_and_normal(self.node, self.element, eps=eps)
        self.node_normal=TriangleMesh.cal_node_normal(self.node, self.element, angle_weighted, self.element_normal, eps=eps)
        error=torch.isnan(self.node_normal).sum()
        if error > 0:
            print("error: nan in normal_quad @ TriangleMesh:update_node_normal")

    @staticmethod
    def cal_node_normal(node, element, angle_weighted=True, element_normal=None, eps=None):
        if eps is None:
            if node.dtype in (torch.float32, torch.float64):
                eps=1e-12
            else:
                raise NotImplementedError()        
        #-----------------------------------
        if element_normal is None:
            element_area, element_normal=TriangleMesh.cal_element_area_and_normal(node, element, eps=eps)
        #-----------------------------------
        M=element.shape[0]
        e_normal=element_normal.reshape(M, 1, 3)
        e_normal=e_normal.expand(M, 3, 3)
        e_normal=e_normal.reshape(M*3, 3)
        N=node.shape[0]
        if angle_weighted == True:
            e_angle=TriangleMesh.cal_element_corner_angle(node, element, eps=eps)#e_angle: (M,3)
            weight=e_angle/e_angle.sum(dim=1, keepdim=True)
            e_normal=e_normal*weight.reshape(M*3,1)
        node_normal = torch_scatter.scatter(e_normal, element.reshape(-1), dim=0, dim_size=N, reduce="sum")        
        normal_norm=norm(node_normal, ord=2, dim=-1, keepdim=True)
        node_normal=node_normal/normal_norm.clamp_min(eps)
        node_normal=node_normal.contiguous()
        return node_normal

    def update_element_area_and_normal(self, eps=None):
         self.element_area, self.element_normal=TriangleMesh.cal_element_area_and_normal(self.node, self.element, eps=eps)

    def update_element_corner_angle(self, eps=None):
        self.element_corner_angle = TriangleMesh.cal_element_corner_angle(self.node, self.element, eps=eps)

    @staticmethod
    def cal_element_area_and_normal(node, element, eps=None):
        if eps is None:
            if node.dtype in (torch.float32, torch.float64):
                eps=1e-12
            else:
                raise NotImplementedError()
        #-----------------------------------
        x0=node[element[:,0]] #(M,3)
        x1=node[element[:,1]] #(M,3)
        x2=node[element[:,2]] #(M,3)
        #   x2
        #  /  \
        # x0--x1
        #normal is undefined if area is 0
        cross_product=torch.cross(x1-x0, x2-x0, dim=-1)
        length=norm(cross_product, ord=2, dim=-1, keepdim=True)
        area=0.5*length #(M,1)
        normal=cross_product/length.clamp_min(eps) #(M,3)
        return area, normal

    @staticmethod
    def cal_element_normal(node, element, eps=None):
        element_area, element_normal=TriangleMesh.cal_element_area_and_normal(node, element, eps=eps)
        return element_normal

    @staticmethod
    def cal_element_corner_angle(node, element, return_cos=False, eps=None):
        x0=node[element[:,0]]
        x1=node[element[:,1]]
        x2=node[element[:,2]]
        #   x2
        #  /  \
        # x0--x1
        angle0=cal_angle_between_3d_vector(x1-x0, x2-x0, return_cos, eps=eps)
        angle1=cal_angle_between_3d_vector(x2-x1, x0-x1, return_cos, eps=eps)
        angle2=cal_angle_between_3d_vector(x0-x2, x1-x2, return_cos, eps=eps)
        angle=torch.cat([angle0.reshape(-1,1), angle1.reshape(-1,1), angle2.reshape(-1,1)], dim=1)
        return angle

    def sample_point(self, n_point):
        return TriangleMesh.uniformly_sample_point(self.node, self.element, n_point)

    @staticmethod
    def uniformly_sample_point(node, element, n_point):
        area, normal=TriangleMesh.cal_element_area_and_normal(node, element)
        prob = area / area.sum()
        if n_point > len(element):
            sample = torch.multinomial(prob.reshape(-1), n_point-len(element), replacement=True)
            #print("sample_points", area.shape, prob.shape, sample.shape)
            element = torch.cat([element, element[sample]], dim=0)
        else:
            sample = torch.multinomial(prob.reshape(-1), n_point, replacement=True)
            #print("sample_points", area.shape, prob.shape, sample.shape)
            element = element[sample]
        a = torch.rand(2, n_point, 1, dtype=node.dtype, device=node.device)
        x0=node[element[:,0]]
        x1=node[element[:,1]]
        x2=node[element[:,2]]
        #x=a[1]*(a[0]*x1+(1-a[0])*x2)+(1-a[1])*x0
        u=a[0].sqrt()
        v=a[1]
        x=(1-u)*x0+u*(1-v)*x1+u*v*x2
        return x

    def subdivid_to_tri_1to4(self):
        #return a new mesh
        #add a node in the middle of each edge
        if self.edge is None:
            self.build_edge()
        x_j=self.node[self.edge[:,0]]
        x_i=self.node[self.edge[:,1]]
        nodeA=(x_j+x_i)/2
        #create new mesh
        node_new=torch.cat([self.node, nodeA], dim=0)        
        element=self.element.tolist()
        element_new=[]
        for m in range(0, len(element)):
            #-----------
            #     x2
            #    /  \
            #   x5-- x4
            #  / \  / \
            # x0--x3--x1
            #-----------
            id0=element[m][0]
            id1=element[m][1]
            id2=element[m][2]
            id3=self.node.shape[0]+self.get_edge_idx_from_node_pair(id0, id1)
            id4=self.node.shape[0]+self.get_edge_idx_from_node_pair(id1, id2)
            id5=self.node.shape[0]+self.get_edge_idx_from_node_pair(id0, id2)    
            element_new.append([id0, id3, id5])
            element_new.append([id3, id4, id5])
            element_new.append([id3, id1, id4])
            element_new.append([id5, id4, id2])
        mesh_new=TriangleMesh(node_new, element_new)
        return mesh_new
    
    def subdivid_to_tri_1to3(self, mode=None):
        #return a new mesh
        #add a node in the middle of each face        
        nodeA=self.node[self.element].mean(dim=1)
        #create new mesh
        node_new=torch.cat([self.node, nodeA], dim=0)        
        element=self.element.tolist()
        element_new=[]
        for m in range(0, len(element)):
            #-----------
            #     x2
            #    / | \
            #   / x3  \
            #  / /  \  \
            #  x0-----x1
            #-----------
            id0=element[m][0]
            id1=element[m][1]
            id2=element[m][2]
            id3=self.node.shape[0]+m
            element_new.append([id0, id1, id3])
            element_new.append([id0, id3, id2])
            element_new.append([id1, id2, id3])
        mesh_new=TriangleMesh(node_new, element_new)
        return mesh_new    

    def subdivid_to_tri(self, mode):
        if mode == "1to4":
            return self.subdivid_to_tri_1to4()
        elif mode == "1to3":
            return self.subdivid_to_tri_1to3()
        else:
            raise ValueError('unsupported mode: '+str(mode))
        
    def subdivide_to_quad(self):
        #return a new mesh
        #add a node in the middle of each edge
        if self.edge is None:
            self.build_edge()
        x_j=self.node[self.edge[:,0]]
        x_i=self.node[self.edge[:,1]]
        nodeA=(x_j+x_i)/2
        #add a node in the middle of each element
        nodeB=self.node[self.element].mean(dim=1)
        #create new mesh
        node_new=torch.cat([self.node, nodeA, nodeB], dim=0)
        element=self.element.tolist()
        element_new=[]
        for m in range(0, len(element)):
            #-----------
            #     x2
            #     /\
            #   x5  x4
            #   / \/ \
            #  /  x6  \
            # /   |    \
            #x0---x3---x1
            #-----------
            id0=element[m][0]
            id1=element[m][1]
            id2=element[m][2]
            id3=self.node.shape[0]+self.get_edge_idx_from_node_pair(id0, id1)
            id4=self.node.shape[0]+self.get_edge_idx_from_node_pair(id1, id2)
            id5=self.node.shape[0]+self.get_edge_idx_from_node_pair(id0, id2)
            id6=self.node.shape[0]+nodeA.shape[0]+m
            element_new.append([id6, id5, id0, id3])
            element_new.append([id6, id3, id1, id4])
            element_new.append([id6, id4, id2, id5])
        mesh_new=QuadMesh(node_new, element_new)
        return mesh_new

    def get_sub_mesh(self, element_idx_list, return_node_idx_list=False):
        sub_mesh, node_idx_list=super().get_sub_mesh(element_idx_list, return_node_idx_list=True)
        sub_mesh=TriangleMesh(sub_mesh.node, sub_mesh.element)
        if return_node_idx_list == False:
            return sub_mesh
        else:
            return sub_mesh, node_idx_list
#%%

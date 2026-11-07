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

    def clear(self):
        super().clear()

    def clear_geometry_derived_data(self, clear_adjacency_data=True):
        super().clear_geometry_derived_data(clear_adjacency_data)
        self.node_normal=None
        self.element_area=None
        self.element_normal=None
        self.element_corner_angle=None

    def update_node_normal(self, angle_weighted=True, eps=None):
        self.element_area, self.element_normal=TriangleMesh.cal_element_area_and_normal(self.node, self.element, eps=eps)
        self.node_normal=TriangleMesh.cal_node_normal(self.node, self.element, angle_weighted, self.element_normal, eps=eps)
        error=torch.isnan(self.node_normal).sum()
        if error > 0:
            print("error: nan in node_normal @ TriangleMesh:update_node_normal")

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
        N=node.shape[0]
        if angle_weighted == True:
            e_angle=TriangleMesh.cal_element_corner_angle(node, element, eps=eps)#e_angle: (M,3)            
            e_normal=e_normal*e_angle.reshape(M,3,1)
        #to e_normal.device: torch_scatter needs src and index on the same device
        node_normal = torch_scatter.scatter(e_normal.reshape(-1,3), element.reshape(-1).to(e_normal.device), dim=0, dim_size=N, reduce="sum")        
        normal_norm=norm(node_normal, ord=2, dim=-1, keepdim=True)
        node_normal=node_normal/normal_norm.clamp_min(eps)
        node_normal=node_normal.contiguous()
        return node_normal

    def update_element_area_and_normal(self, eps=None):
         self.element_area, self.element_normal=TriangleMesh.cal_element_area_and_normal(self.node, self.element, eps=eps)
        
    def update_element_normal(self, eps=None):
         __, self.element_normal=TriangleMesh.cal_element_area_and_normal(self.node, self.element, eps=eps)

    def update_element_corner_angle(self, eps=None):
        self.element_corner_angle = TriangleMesh.cal_element_corner_angle(self.node, self.element, eps=eps)

    @staticmethod
    def cal_element_area_and_normal(node, element, eps=None):
        """Return element_area (M,1) and element_normal (M,3).

        The normal will not work for very small triangles. If the cross product of two edges is shorter than eps,
        which with the default eps=1e-12 is a right triangle of side 1e-6 or less, the normal is shorter than 1:
        its length is 0.01 at side 1e-7. Pass a smaller eps for such triangles.
        """
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

    def sample_point(self, n_point, at_least_one_point_on_one_element=False):
        return TriangleMesh.uniformly_sample_point(self.node, self.element, n_point, at_least_one_point_on_one_element)

    @staticmethod
    def uniformly_sample_point(node, element, n_point, at_least_one_point_on_one_element=False):
        #at_least_one_point_on_one_element=False: each point picks its element by area,
        #  so the points are uniform on the surface, and a small element may get no point
        #at_least_one_point_on_one_element=True: each element gets one point, and the other
        #  n_point-len(element) points pick their elements by area, so an element gets
        #  1+(n_point-len(element))*prob points on average, and small elements get more
        #  than their share of the area. n_point must be >= len(element)
        M=len(element)
        n_pick=n_point
        if at_least_one_point_on_one_element:
            if n_point < M:
                raise ValueError('n_point must be >= the number of elements if at_least_one_point_on_one_element is True')
            n_pick=n_point-M
        sample=torch.empty(0, dtype=torch.int64, device=element.device)
        if n_pick > 0:
            #float64: so that the area of a tiny triangle in a float32 mesh does not become 0
            x=node.detach().to(torch.float64)
            x0=x[element[:,0]]
            area=norm(torch.cross(x[element[:,1]]-x0, x[element[:,2]]-x0, dim=1), ord=2, dim=1)
            #this is faster than torch.multinomial
            cdf=torch.cumsum(area, dim=0)
            if M == 0 or not (0 < cdf[-1] < float('inf')):
                raise ValueError('cannot sample points on a mesh without a valid area')
            r=torch.rand(n_pick, dtype=torch.float64, device=node.device)*cdf[-1]
            #right=True: an element with zero area is not picked
            #to element.device: a CPU element can't be indexed by a CUDA sample
            sample=torch.searchsorted(cdf, r, right=True).clamp_max(M-1).to(element.device)
        if at_least_one_point_on_one_element:
            sample=torch.cat([torch.arange(M, device=element.device), sample], dim=0)
        element=element[sample]
        a = torch.rand(2, n_point, 1, dtype=node.dtype, device=node.device)
        x0=node[element[:,0]]
        x1=node[element[:,1]]
        x2=node[element[:,2]]
        #x=a[1]*(a[0]*x1+(1-a[0])*x2)+(1-a[1])*x0
        u=a[0].sqrt()
        v=a[1]
        x=(1-u)*x0+u*(1-v)*x1+u*v*x2
        return x

    def subdivide_to_tri_1to4(self):
        #return a new mesh
        element=self.make_element_tensor().tolist()
        #add a node in the middle of each edge
        if self.edge is None:
            self.build_edge()
        x_j=self.node[self.edge[:,0]]
        x_i=self.node[self.edge[:,1]]
        nodeA=(x_j+x_i)/2
        #create new mesh
        node_new=torch.cat([self.node, nodeA], dim=0)        
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
    
    def subdivide_to_tri_1to3(self, mode=None):
        #return a new mesh
        #add a node in the middle of each face        
        element=self.make_element_tensor()
        nodeA=self.node[element].mean(dim=1)
        #create new mesh
        node_new=torch.cat([self.node, nodeA], dim=0)        
        element=element.tolist()
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

    def subdivide_to_tri(self, mode):
        if mode == "1to4":
            return self.subdivide_to_tri_1to4()
        elif mode == "1to3":
            return self.subdivide_to_tri_1to3()
        else:
            raise ValueError('unsupported mode: '+str(mode))
        
    def subdivide_to_quad(self):
        #return a new mesh
        element=self.make_element_tensor()
        #add a node in the middle of each edge
        if self.edge is None:
            self.build_edge()
        x_j=self.node[self.edge[:,0]]
        x_i=self.node[self.edge[:,1]]
        nodeA=(x_j+x_i)/2
        #add a node in the middle of each element
        nodeB=self.node[element].mean(dim=1)
        #create new mesh
        node_new=torch.cat([self.node, nodeA, nodeB], dim=0)
        element=element.tolist()
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

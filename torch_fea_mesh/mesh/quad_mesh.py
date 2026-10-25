import torch
from torch.linalg import vector_norm as norm
from torch.linalg import cross
import torch_scatter
import numpy as np
from .polygon_mesh import PolygonMesh
from ..utility.geometry_utility import cal_angle_between_3d_vector
#%%
class QuadMesh(PolygonMesh):
    #4-node quad element mesh

    def __init__(self, node=None, element=None):
        super().__init__(node=node, element=element)
        self.mesh_type='polygon_quad4'
        if (node is not None) and (element is not None):
            if not self.is_quad():
                raise ValueError('not a quad mesh')

    def clear(self):
        super().clear()

    def clear_geometry_derived_data(self):
        super().clear_geometry_derived_data()
        self.node_normal=None
        self.element_area=None
        self.element_normal=None
        self.element_corner_angle=None
        self.element_flatness=None

    def update_node_normal(self, angle_weighted=True, eps=None):
        self.node_normal=QuadMesh.cal_node_normal(self.node, self.element, angle_weighted=angle_weighted, eps=eps)
        error=torch.isnan(self.node_normal).sum()
        if error > 0:
            print("error: nan in normal_quad @ TriangleMesh:update_node_normal")

    @staticmethod
    def cal_node_normal(node, element, angle_weighted=True, eps=None):
        if eps is None:
            if node.dtype in (torch.float32, torch.float64):
                eps=1e-12
            else:
                raise NotImplementedError()
        #-----------------------------------
        x0=node[element[:,0]]
        x1=node[element[:,1]]
        x2=node[element[:,2]]
        x3=node[element[:,3]]
        # x3--x2
        # |   |
        # x0--x1
        #normal could be nan if xa==xb (e.g., x0 == x1)
        normal0=cross(x1-x0, x3-x0, dim=-1)
        normal0_norm=norm(normal0, ord=2, dim=-1, keepdim=True)
        normal0=normal0/normal0_norm.clamp_min(eps)
        
        normal1=cross(x2-x1, x0-x1, dim=-1)
        normal1_norm=norm(normal1, ord=2, dim=-1, keepdim=True)
        normal1=normal1/normal1_norm.clamp_min(eps)
        
        normal2=cross(x3-x2, x1-x2, dim=-1)
        normal2_norm=norm(normal2, ord=2, dim=-1, keepdim=True)
        normal2=normal2/normal2_norm.clamp_min(eps)

        normal3=cross(x0-x3, x2-x3, dim=-1)
        normal3_norm=norm(normal3, ord=2, dim=-1, keepdim=True)
        normal3=normal3/normal3_norm.clamp_min(eps)

        M=element.shape[0]
        N=node.shape[0]
        normal0123=torch.cat([normal0.reshape(M,1,3),
                              normal1.reshape(M,1,3),
                              normal2.reshape(M,1,3),
                              normal3.reshape(M,1,3)], dim=1)
        if angle_weighted == True:
            e_angle=QuadMesh.cal_element_corner_angle(node, element, eps=eps)#e_angle: (M,3)
            weight=e_angle/e_angle.sum(dim=1, keepdim=True)
            normal0123=normal0123*weight.reshape(M,4,1)
        normal=torch_scatter.scatter(normal0123.reshape(-1,3), element.reshape(-1), dim=0, dim_size=N, reduce="sum")
        error=torch.isnan(normal).sum()
        if error > 0:
            print("error: nan in normal_quad @ QuadMesh:cal_node_normal")
        normal_norm=norm(normal, ord=2, dim=-1, keepdim=True)
        normal=normal/normal_norm.clamp_min(eps)
        normal=normal.contiguous()
        return normal

    def update_element_area_and_normal(self, eps=None):
         self.element_area, self.element_normal=QuadMesh.cal_element_area_and_normal(self.node, self.element, eps=eps)

    @staticmethod
    def cal_element_area_and_normal(node, element, eps=None):
        if eps is None:
            if node.dtype in (torch.float32, torch.float64):
                eps=1e-12
            else:
                raise NotImplementedError()
        #-----------------------------------
        #area is an estimation using 1 integration point
        #normal is at the center
        x0=node[element[:,0]] #(M,3)
        x1=node[element[:,1]] #(M,3)
        x2=node[element[:,2]] #(M,3)
        x3=node[element[:,3]] #(M,3)
        # x3--x2
        # |   |
        # x0--x1
        dxdu=(1/4)*((x1+x2)-(x0+x3))
        dxdv=(1/4)*((x2+x3)-(x0+x1))
        cross_uv=cross(dxdu, dxdv, dim=-1)
        length=norm(cross_uv, ord=2, dim=-1, keepdim=True)
        area=4*length #(M,1)
        normal=cross_uv/length.clamp_min(eps) #(M,3)
        return area, normal

    def update_element_corner_angle(self, eps=None):
        self.element_corner_angle = QuadMesh.cal_element_corner_angle(self.node, self.element, eps=eps)

    @staticmethod
    def cal_element_corner_angle(node, element, return_cos=False, eps=None):
        x0=node[element[:,0]]
        x1=node[element[:,1]]
        x2=node[element[:,2]]
        x3=node[element[:,3]]
        # x3--x2
        # |   |
        # x0--x1
        angle0=cal_angle_between_3d_vector(x1-x0, x3-x0, return_cos, eps=eps)
        angle1=cal_angle_between_3d_vector(x2-x1, x0-x1, return_cos, eps=eps)
        angle2=cal_angle_between_3d_vector(x3-x2, x1-x2, return_cos, eps=eps)
        angle3=cal_angle_between_3d_vector(x0-x3, x2-x3, return_cos, eps=eps)
        angle=torch.cat([angle0.reshape(-1,1), angle1.reshape(-1,1), angle2.reshape(-1,1), angle3.reshape(-1,1)], dim=1)
        return angle
    
    def upate_element_flatness(self, eps=None):
        self.element_flatness=QuadMesh.cal_element_flatness(self.node, self.element, eps=eps)

    @staticmethod
    def cal_element_flatness(node, element, eps=None):
        if eps is None:
            if node.dtype in (torch.float32, torch.float64):
                eps=1e-12
            else:
                raise NotImplementedError()
        #-----------------------------------
        # 0 (fold) <= flatness <= 1 (flat)
        x0=node[element[:,0]]
        x1=node[element[:,1]]
        x2=node[element[:,2]]
        x3=node[element[:,3]]
        # x3--x2
        # |   |
        # x0--x1
        #-------------------------
        d023=cross(x2-x0, x3-x0, dim=-1)
        d023_norm=norm(d023, ord=2, dim=-1, keepdim=True)
        d023=d023/d023_norm.clamp_min(eps)
        #-------------------------
        d012=cross(x1-x0, x2-x0, dim=-1)
        d012_norm=norm(d012, ord=2, dim=-1, keepdim=True)
        d012=d012/d012_norm.clamp_min(eps)
        #-------------------------
        d123=cross(x2-x1, x3-x1, dim=-1)
        d123_norm=norm(d123, ord=2, dim=-1, keepdim=True)
        d123=d123/d123_norm.clamp_min(eps)
        #-------------------------
        d130=cross(x3-x1, x0-x1, dim=-1)
        d130_norm=norm(d130, ord=2, dim=-1, keepdim=True)
        d130=d130/d130_norm.clamp_min(eps)
        #-------------------------
        #bad: gradient is not 0 when it is flat
        #flatness=0.5*((d023*d012).sum(dim=-1)+(d123*d130).sum(dim=-1))
        #good
        #0.125: flatness is 0.5 when d023 ⊥ d012 and d123 ⊥ d130
        #0.125: flatness is 0 when d023 = -d012 and d123 = -d130
        flatness=1-0.125*(((d023-d012)**2).sum(dim=-1)+((d123-d130)**2).sum(dim=-1))
        return flatness
    
    @staticmethod
    def cal_local_surface_flatness(node, element, element_normal=None, eps=None):
        if eps is None:
            if node.dtype in (torch.float32, torch.float64):
                eps=1e-12
            else:
                raise NotImplementedError()
        #-----------------------------------
        #local surface at each node
        if element_normal is None:
            element_area, element_normal=QuadMesh.cal_element_area_and_normal(node, element, eps=eps)
        N=node.shape[0]
        M=element.shape[0]
        element_normal=element_normal.reshape(M,1,3).expand(M,4,3)
        normal_mean=torch_scatter.scatter(element_normal.reshape(-1,3), element.reshape(-1), dim=0, dim_size=N, reduce="mean")
        #normal_mean.shape (N,3)
        normal_mean_norm=norm(normal_mean, ord=2, dim=-1, keepdim=True)
        normal_mean=normal_mean/normal_mean_norm.clamp_min(eps)
        normal_mean=normal_mean[element] #(M,4,3)
        diff=((element_normal-normal_mean)**2).sum(dim=2)#(M,4)
        variance=torch_scatter.scatter(diff.reshape(-1), element.reshape(-1), dim=0, dim_size=N, reduce="mean")
        flatness=1-variance               
        return flatness
       
    def sample_point(self, n_point):
         return QuadMesh.uniformly_sample_point(self.node, self.element, n_point)

    @staticmethod
    def uniformly_sample_point(node, element, n_point):
        area, normal=QuadMesh.cal_element_area_and_normal(node, element)
        prob = area / area.sum()
        if n_point > len(element):
            sample = torch.multinomial(prob.reshape(-1), n_point-len(element), replacement=True)
            #print("sample_points", area.shape, prob.shape, sample.shape)
            element = torch.cat([element, element[sample]], dim=0)
        else:
            sample = torch.multinomial(prob.reshape(-1), n_point, replacement=True)
            #print("sample_points", area.shape, prob.shape, sample.shape)
            element = element[sample]
        a = torch.rand(3, n_point, 1, dtype=node.dtype, device=node.device)
        x0=node[element[:,0]]
        x1=node[element[:,1]]
        x2=node[element[:,2]]
        x3=node[element[:,3]]
        x=a[2]*(a[0]*x0+(1-a[0])*x1)+(1-a[2])*(a[1]*x2+(1-a[1])*x3)
        return x

    def subdivide_to_quad(self):
        #return a new mesh        
        #add a node in the middle of each edge
        if self.edge is None:
            self.build_edge()
        x_j=self.node[self.edge[:,0]]
        x_i=self.node[self.edge[:,1]]
        nodeA=(x_j+x_i)/2        
        #add a node in the middle of each element
        nodeB=self.node[self.element].mean(dim=1) #(N,3) => (M,8,3) => (M,3)
        #create new mesh
        node_new=torch.cat([self.node, nodeA, nodeB], dim=0)
        element_new=[]
        if torch.is_tensor(self.element):
            element=self.element.cpu().numpy()
        for m in range(0, element.shape[0]):
            # x3--x6--x2
            # |   |   |
            # x7--x8--x5
            # |   |   |
            # x0--x4--x1
            #-----------
            id0=int(element[m][0])
            id1=int(element[m][1])
            id2=int(element[m][2])
            id3=int(element[m][3])
            id4=self.node.shape[0]+self.get_edge_idx_from_node_pair(id0, id1)
            id5=self.node.shape[0]+self.get_edge_idx_from_node_pair(id1, id2)
            id6=self.node.shape[0]+self.get_edge_idx_from_node_pair(id2, id3)
            id7=self.node.shape[0]+self.get_edge_idx_from_node_pair(id0, id3)            
            id8=self.node.shape[0]+nodeA.shape[0]+m
            element_new.append([id0, id4, id8, id7])
            element_new.append([id4, id1, id5, id8])
            element_new.append([id7, id8, id6, id3])
            element_new.append([id8, id5, id2, id6])
        mesh_new=QuadMesh(node_new, element_new)
        return mesh_new

    def get_sub_mesh(self, element_idx_list, return_node_idx_list=False):
        sub_mesh, node_idx_list=super().get_sub_mesh(element_idx_list, return_node_idx_list=True)
        sub_mesh=QuadMesh(sub_mesh.node, sub_mesh.element)
        if return_node_idx_list == False:
            return sub_mesh
        else:
            return sub_mesh, node_idx_list
#%%

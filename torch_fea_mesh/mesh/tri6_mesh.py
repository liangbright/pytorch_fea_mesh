import torch
from torch.linalg import vector_norm as norm
import torch_scatter
import numpy as np
from .triangle_mesh import TriangleMesh
from .polygon_mesh import PolygonMesh
from ..utility.geometry_utility import cal_angle_between_3d_vector
#%%
class Tri6Mesh(PolygonMesh):
    #6-node triangle element mesh
    #     x2
    #    /  \
    #   x5   x4
    #  /      \
    # x0--x3--x1
    # node order in a tri6 element: [x0, x1, x2, x3, x4, x5]   
    # note: a linear polygon (sixgon, hexagon) element has this node order [x0, x3, x1, x4, x2, x5]
    #----------------------------------------------------------------------------
    def __init__(self, node=None, element=None):
        super().__init__(node=node, element=element)
        self.mesh_type='polygon_tri6'
        #if (node is not None) and (element is not None):
        #    if not self.is_tri6():
        #        raise ValueError('not a tri6 mesh')

    def clear(self):
        super().clear()

    def clear_geometry_derived_data(self):
        super().clear_geometry_derived_data()
        self.node_normal=None
        self.element_area=None
        self.element_normal=None
        self.element_corner_angle=None

    def create_from_tri3(self, tri3_mesh):
        #tri3_mesh is a TriangleMesh (3 nodes in an element)
        #add a node in the middle of each edge of tri3_mesh
        if tri3_mesh.edge is None:
            tri3_mesh.build_edge()
        x_j=tri3_mesh.node[tri3_mesh.edge[:,0]]
        x_i=tri3_mesh.node[tri3_mesh.edge[:,1]]
        nodeA=(x_j+x_i)/2
        #create new mesh
        node_new=torch.cat([tri3_mesh.node, nodeA], dim=0)        
        element=tri3_mesh.element.tolist()
        element_new=[]
        for m in range(0, len(element)):
            #-----------
            #     x2
            #    /  \
            #   x5  x4
            #  /      \
            # x0--x3--x1
            #-----------
            id0=element[m][0]
            id1=element[m][1]
            id2=element[m][2]
            id3=tri3_mesh.node.shape[0]+tri3_mesh.get_edge_idx_from_node_pair(id0, id1)
            id4=tri3_mesh.node.shape[0]+tri3_mesh.get_edge_idx_from_node_pair(id1, id2)
            id5=tri3_mesh.node.shape[0]+tri3_mesh.get_edge_idx_from_node_pair(id0, id2)    
            element_new.append([id0, id1, id2, id3, id4, id5])
        self.__init__(node_new, element_new)
    
    def simplify_to_tri3(self, return_node_idx_list=False):
        element_out=self.element[:,0:3].clone()
        node_idx_list, element_out=torch.unique(element_out.reshape(-1), return_inverse=True)
        node_out=self.node[node_idx_list]
        element_out=element_out.reshape(-1,3)
        mesh_new=TriangleMesh(node_out, element_out)
        if return_node_idx_list == False:
            return mesh_new
        else:
            return mesh_new, node_idx_list
        
    def subdivide_to_tri3(self, clone_node=True):        
        element_out=[]
        for m in range(0, self.element.shape[0]):
            #-----------
            #     x2
            #    /  \
            #   x5  x4
            #  /      \
            # x0--x3--x1
            #-----------
            id0, id1, id2, id3, id4, id5 = self.element[m].tolist()
            element_out.append([id0, id3, id5])
            element_out.append([id3, id4, id5])
            element_out.append([id1, id4, id3])
            element_out.append([id2, id5, id4])
            node_out=self.node
            if clone_node==True:
                node_out=node_out.clone()
        mesh_new=TriangleMesh(node_out, element_out)
        return mesh_new
        
    def create_from_sixgon(self, sixgon_mesh, clone_node=True):
        #sixgon: hexagon 
        element_new=[]
        for m in range(0, len(sixgon_mesh.element)):
            elm=sixgon_mesh.copy_to_list(sixgon_mesh.element[m])
            id0, id3, id1, id4, id2, id5 = elm
            element_new.append([id0, id1, id2, id3, id4, id5])
        node=sixgon_mesh.node
        if clone_node == True:
            node=node.clone()
        self.__init__(node, element_new)
    
    def convert_to_sixgon(self, clone_node=True):
        #sixgon: hexagon 
        element_new=self.element[:,[0,3,1,4,2,5]].clone()        
        node=self.node
        if clone_node == True:
            node=node.clone()
        mesh_new=PolygonMesh(node, element_new)
        return mesh_new

    def update_node_normal(self, angle_weighted=True, eps=None):
        self.element_area, self.element_normal=Tri6Mesh.cal_element_area_and_normal(self.node, self.element, eps=eps)
        self.node_normal=Tri6Mesh.cal_node_normal(self.node, self.element, angle_weighted, self.element_normal, eps=eps)
        error=torch.isnan(self.node_normal).sum()
        if error > 0:
            print("error: nan in normal_quad @ Tri6Mesh:update_node_normal")

    @staticmethod
    def cal_node_normal(node, element, angle_weighted=True, element_normal=None, eps=None):
        if eps is None:
            if node.dtype in (torch.float32, torch.float64):
                eps=1e-12
            else:
                raise NotImplementedError()
        #-----------------------------------
        if element_normal is None:
            element_area, element_normal=Tri6Mesh.cal_element_area_and_normal(node, element, eps=eps)
        M=element.shape[0]
        e_normal=element_normal.reshape(M, 1, 3)
        e_normal=e_normal.expand(M, 6, 3)
        e_normal=e_normal.reshape(M*6, 3)
        N=node.shape[0]
        if angle_weighted == True:
            e_angle=Tri6Mesh.cal_element_corner_angle(node, element, eps=eps)#e_angle: (M,6)
            weight=e_angle/e_angle.sum(dim=1, keepdim=True)
            e_normal=e_normal*weight.reshape(M*6,1)
        node_normal = torch_scatter.scatter(e_normal, element.reshape(-1), dim=0, dim_size=N, reduce="sum")        
        normal_norm=norm(node_normal, ord=2, dim=-1, keepdim=True)
        node_normal=node_normal/normal_norm.clamp_min(eps)
        node_normal=node_normal.contiguous()
        return node_normal

    def update_element_area_and_normal(self, eps=None):
         self.element_area, self.element_normal=Tri6Mesh.cal_element_area_and_normal(self.node, self.element, eps=eps)

    def update_element_corner_angle(self, eps=None):
        self.element_corner_angle = Tri6Mesh.cal_element_corner_angle(self.node, self.element, eps=eps)

    @staticmethod
    def cal_element_area_and_normal(node, element, eps=None):
        if eps is None:
            if node.dtype in (torch.float32, torch.float64):
                eps=1e-12
            else:
                raise NotImplementedError()
        #-----------------------------------
        def cal_tri3(x0, x1, x2):
            #   x2            
            #  /  \
            # x0--x1
            cross_product=torch.cross(x1-x0, x2-x0, dim=-1)
            length=norm(cross_product, ord=2, dim=-1, keepdim=True)
            area=0.5*length
            normal=cross_product/length.clamp_min(eps)
            return area, normal
        #------------------------------------
        x0=node[element[:,0]]
        x1=node[element[:,1]]
        x2=node[element[:,2]]
        x3=node[element[:,3]]
        x4=node[element[:,4]]
        x5=node[element[:,5]]
        #     x2
        #    /  \
        #   x5   x4
        #  /      \
        # x0--x3--x1
        #normal is undefined if area is 0
        area035, normal035=cal_tri3(x0, x3, x5)
        area345, normal345=cal_tri3(x3, x4, x5)
        area143, normal143=cal_tri3(x1, x4, x3)
        area254, normal254=cal_tri3(x2, x5, x4)
        area=area035+area345+area143+area254
        normal=0.25*(normal035+normal345+normal143+normal254)
        return area, normal

    @staticmethod
    def cal_element_normal(node, element, eps=None):
        element_area, element_normal=Tri6Mesh.cal_element_area_and_normal(node, element, eps=eps)
        return element_normal

    @staticmethod
    def cal_element_corner_angle(node, element, return_cos=False, eps=None):
        x0=node[element[:,0]]
        x1=node[element[:,1]]
        x2=node[element[:,2]]
        x3=node[element[:,3]]
        x4=node[element[:,4]]
        x5=node[element[:,5]]
        #     x2
        #    /  \
        #   x5   x4
        #  /      \
        # x0--x3--x1
        angle0=cal_angle_between_3d_vector(x3-x0, x5-x0, return_cos, eps=eps)
        angle1=cal_angle_between_3d_vector(x4-x1, x3-x1, return_cos, eps=eps)
        angle2=cal_angle_between_3d_vector(x5-x2, x4-x2, return_cos, eps=eps)
        angle3=cal_angle_between_3d_vector(x1-x3, x0-x3, return_cos, eps=eps)
        angle4=cal_angle_between_3d_vector(x2-x4, x1-x4, return_cos, eps=eps)
        angle5=cal_angle_between_3d_vector(x0-x5, x2-x5, return_cos, eps=eps)
        angle=torch.cat([angle0.reshape(-1,1), angle1.reshape(-1,1), angle2.reshape(-1,1),
                         angle3.reshape(-1,1), angle4.reshape(-1,1), angle5.reshape(-1,1)], dim=1)
        return angle

    def get_sub_mesh(self, element_idx_list, return_node_idx_list=False):
        sub_mesh, node_idx_list=super().get_sub_mesh(element_idx_list, return_node_idx_list=True)
        sub_mesh=Tri6Mesh(sub_mesh.node, sub_mesh.element)
        if return_node_idx_list == False:
            return sub_mesh
        else:
            return sub_mesh, node_idx_list
#%%

import torch
from torch.linalg import vector_norm as norm
from torch.linalg import cross
import torch_scatter
import numpy as np
from torch_fea_mesh.mesh.polygon_mesh import PolygonMesh
from torch_fea_mesh.utility.geometry_utility import cal_angle_between_3d_vector
#%%
class QuadMesh(PolygonMesh):
    #4-node quad element mesh

    def __init__(self, node=None, element=None):
        super().__init__(node=node, element=element)
        self.mesh_type='polygon_quad4'

    def clear(self):
        super().clear()

    def clear_geometry_derived_data(self, clear_adjacency_data=True):
        super().clear_geometry_derived_data(clear_adjacency_data)
        self.node_normal=None
        self.element_area=None
        self.element_normal=None
        self.element_corner_angle=None
        self.element_flatness=None

    def update_node_normal(self, angle_weighted=True, eps=None, normal_source='corner'):
        self.node_normal=QuadMesh.cal_node_normal(self.node, self.element, angle_weighted=angle_weighted, eps=eps, normal_source=normal_source)
        error=torch.isnan(self.node_normal).sum()
        if error > 0:
            print("error: nan in node_normal @ QuadMesh:update_node_normal")

    @staticmethod
    def cal_node_normal(node, element, angle_weighted=True, eps=None, normal_source='corner'):
        #normal_source: the normal that each corner of an element adds to its node
        #  'corner':  the normal at that corner, cross(next-a, previous-a)
        #  'element': the normal of the whole element, as in TriangleMesh
        if normal_source not in ('corner', 'element'):
            raise ValueError("normal_source must be 'corner' or 'element', not "+repr(normal_source))
        if eps is None:
            if node.dtype in (torch.float32, torch.float64):
                eps=1e-12
            else:
                raise NotImplementedError()
        #-----------------------------------
        M=element.shape[0]
        N=node.shape[0]
        if normal_source == 'corner':
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

            normal0123=torch.cat([normal0.reshape(M,1,3),
                                  normal1.reshape(M,1,3),
                                  normal2.reshape(M,1,3),
                                  normal3.reshape(M,1,3)], dim=1)
        else:
            normal0123=QuadMesh.cal_element_normal(node, element, eps=eps).reshape(M,1,3).expand(M,4,3)
        if angle_weighted == True:
            e_angle=QuadMesh.cal_element_corner_angle(node, element, eps=eps)#e_angle: (M,4)  
            normal0123=normal0123*e_angle.reshape(M,4,1)
        #to normal0123.device: torch_scatter needs src and index on the same device
        normal=torch_scatter.scatter(normal0123.reshape(-1,3), element.reshape(-1).to(normal0123.device), dim=0, dim_size=N, reduce="sum")
        error=torch.isnan(normal).sum()
        if error > 0:
            print("error: nan in node_normal @ QuadMesh:cal_node_normal")
        normal_norm=norm(normal, ord=2, dim=-1, keepdim=True)
        normal=normal/normal_norm.clamp_min(eps)
        normal=normal.contiguous()
        return normal

    def update_element_area_and_normal(self, eps=None):
         self.element_area, self.element_normal=QuadMesh.cal_element_area_and_normal(self.node, self.element, eps=eps)

    def update_element_normal(self, eps=None):
         self.element_normal=QuadMesh.cal_element_normal(self.node, self.element, eps=eps)

    @staticmethod
    def cal_element_area_and_normal(node, element, eps=None):
        """Return element_area (M,1) and element_normal (M,3).

        The normal will not work for very small quads. See cal_element_normal.
        """
        if eps is None:
            if node.dtype in (torch.float32, torch.float64):
                eps=1e-12
            else:
                raise NotImplementedError()
        #-----------------------------------
        #area is the area of the bilinear surface through the 4 nodes, using 2x2 Gauss integration points
        #  flat quad: exact, and a concave quad is included
        #  warped quad: an estimation. On a unit square, the error is 0.01% if one corner is lifted by 0.8,
        #  +0.8% for a saddle with heights of +-0.5, and +5% for a saddle with heights of +-2
        #normal: see cal_element_normal
        x0=node[element[:,0]] #(M,3)
        x1=node[element[:,1]] #(M,3)
        x2=node[element[:,2]] #(M,3)
        x3=node[element[:,3]] #(M,3)
        # x3--x2
        # |   |
        # x0--x1
        d02=x2-x0
        d13=x3-x1
        cross_d=cross(d02, d13, dim=-1)
        length_d=norm(cross_d, ord=2, dim=-1, keepdim=True)
        normal=cross_d/length_d.clamp_min(8*eps) #(M,3)
        dxdu=(1/4)*(d02-d13)
        dxdv=(1/4)*(d02+d13)
        cross_uv=cross_d/8
        length=length_d/8
        #twist is 0 if the quad is a parallelogram
        #the jacobian (the cross product of dx/du and dx/dv) at (u,v) is linear in u and v:
        #  cross_uv + u*cross(dxdu,twist) + v*cross(twist,dxdv)
        #the 4 Gauss points are u=+-g and v=+-g with g=1/sqrt(3), and each one has a weight of 1
        #the differences come first: adding large coordinates before subtracting would lose digits in float32
        twist=(1/4)*((x0-x1)+(x2-x3))
        cross_ut=cross(dxdu, twist, dim=-1)
        cross_tv=cross(twist, dxdv, dim=-1)
        a=(cross_ut+cross_tv)/3**0.5
        b=(cross_ut-cross_tv)/3**0.5
        area=(norm(cross_uv+a, ord=2, dim=-1, keepdim=True)+norm(cross_uv-a, ord=2, dim=-1, keepdim=True)
             +norm(cross_uv+b, ord=2, dim=-1, keepdim=True)+norm(cross_uv-b, ord=2, dim=-1, keepdim=True)) #(M,1)
        #the bilinear map of a concave quad folds: the jacobian faces the other way in a part of the quad,
        #and adding up its length would count that part as extra area. jacobian_n is its part along the normal.
        #|jacobian|+2*min(jacobian_n,0) is |jacobian| where there is no fold, and it is jacobian_n in the folded
        #part of a flat quad. So the sum is the area of the polygon, and it is never below 4*length.
        #jacobian_n at the 4 points is length+-a_n and length+-b_n, and only the minus sign can be below 0.
        direction=cross_uv/torch.where(length > 0, length, torch.ones_like(length))
        a_n=(a*direction).sum(dim=-1, keepdim=True)
        b_n=(b*direction).sum(dim=-1, keepdim=True)
        area=area+2*((length-a_n.abs()).clamp_max(0)+(length-b_n.abs()).clamp_max(0))
        #length is 0 if the diagonals are parallel: this is a flat quad with no area, such as a symmetric bowtie
        area=torch.where(length > 0, area, torch.zeros_like(area))
        return area, normal

    @staticmethod
    def cal_element_normal(node, element, eps=None):
        """Return element_normal (M,3).

        It will not work for very small quads. If the cross product of the diagonals is shorter than 8*eps,
        which with the default eps=1e-12 is a square of side 2e-6 or less, the normal is shorter than 1:
        its length is 0.25 at side 1e-6 and 0.0025 at side 1e-7. Pass a smaller eps for such quads.
        """
        if eps is None:
            if node.dtype in (torch.float32, torch.float64):
                eps=1e-12
            else:
                raise NotImplementedError()
        #-----------------------------------
        #normal is at the center. It is also the area-weighted mean normal of the bilinear surface, and the
        #direction of the vector area of either cut into two triangles, so the starting node does not matter.
        #on a warped quad, the surface normal at another point is different.
        #the differences come first: adding large coordinates before subtracting would lose digits in float32.
        #cross_d=8*cross(dxdu,dxdv) with dxdu and dxdv at the center, and eps is a limit on the length of cross(dxdu,dxdv)
        cross_d=cross(node[element[:,2]]-node[element[:,0]], node[element[:,3]]-node[element[:,1]], dim=-1)
        length_d=norm(cross_d, ord=2, dim=-1, keepdim=True)
        normal=cross_d/length_d.clamp_min(8*eps) #(M,3)
        return normal

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
    
    def update_element_flatness(self, eps=None):
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
        # a flat concave quad gets 0.5, because a concave quad is a bad shape: one diagonal is outside it
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
            element_normal=QuadMesh.cal_element_normal(node, element, eps=eps)
        N=node.shape[0]
        M=element.shape[0]
        element_normal=element_normal.reshape(M,1,3).expand(M,4,3)
        #to element_normal.device: torch_scatter needs src and index on the same device
        normal_mean=torch_scatter.scatter(element_normal.reshape(-1,3), element.reshape(-1).to(element_normal.device), dim=0, dim_size=N, reduce="mean")
        #normal_mean.shape (N,3)
        normal_mean_norm=norm(normal_mean, ord=2, dim=-1, keepdim=True)
        normal_mean=normal_mean/normal_mean_norm.clamp_min(eps)
        normal_mean=normal_mean[element] #(M,4,3)
        diff=((element_normal-normal_mean)**2).sum(dim=2)#(M,4)
        variance=torch_scatter.scatter(diff.reshape(-1), element.reshape(-1).to(diff.device), dim=0, dim_size=N, reduce="mean")
        flatness=1-variance               
        return flatness
       
    def sample_point(self, n_point, at_least_one_point_on_one_element=False):
         return QuadMesh.uniformly_sample_point(self.node, self.element, n_point, at_least_one_point_on_one_element)

    @staticmethod
    def uniformly_sample_point(node, element, n_point, at_least_one_point_on_one_element=False):
        #each quad is cut into two triangles, and a point is placed uniformly in a triangle picked by area.
        #this is uniform on a flat quad of any shape, concave included.
        #on a warped quad, the areas are measured along the normal of the whole quad, and the
        #points lie on the two flat triangles, not on the curved surface.
        #at_least_one_point_on_one_element=False: each point picks its element by area,
        #  so the points are uniform on the mesh, and a small element may get no point
        #at_least_one_point_on_one_element=True: each element gets one point, and the other
        #  n_point-len(element) points pick their elements by area, so an element gets
        #  1+(n_point-len(element))*prob points on average, and small elements get more
        #  than their share of the area. n_point must be >= len(element)
        #-----------------------------------
        # x3------x2
        # |     / |
        # |   /   |    triangles (x0, x1, x2) and (x0, x3, x2)
        # | /     |    if x0-x2 is outside the quad, the quad is renumbered to start at x1
        # x0------x1
        #-----------------------------------
        M=len(element)
        n_pick=n_point
        if at_least_one_point_on_one_element:
            if n_point < M:
                raise ValueError('n_point must be >= the number of elements if at_least_one_point_on_one_element is True')
            n_pick=n_point-M
        #normal: cross product of the two diagonals, its length is 2*(area of the quad seen along this normal)
        #float64: so that the area of a tiny quad in a float32 mesh does not become 0
        x=node.detach().to(torch.float64)
        normal=cross(x[element[:,2]]-x[element[:,0]], x[element[:,3]]-x[element[:,1]], dim=1)
        sample=torch.empty(0, dtype=torch.int64, device=element.device)
        if n_pick > 0:
            #this is faster than torch.multinomial
            cdf=torch.cumsum(norm(normal, ord=2, dim=1), dim=0)
            if M == 0 or not (0 < cdf[-1] < float('inf')):
                raise ValueError('cannot sample points on a mesh without a valid area')
            r=torch.rand(n_pick, dtype=torch.float64, device=node.device)*cdf[-1]
            #right=True: an element with zero area is not picked
            #to element.device: a CPU element can't be indexed by a CUDA sample
            sample=torch.searchsorted(cdf, r, right=True).clamp_max(M-1).to(element.device)
        if at_least_one_point_on_one_element:
            sample=torch.cat([torch.arange(M, device=element.device), sample], dim=0)
        #the rest is done only for the picked quads
        element=element[sample]
        normal=normal[sample]
        x0=node[element[:,0]]
        x1=node[element[:,1]]
        x2=node[element[:,2]]
        x3=node[element[:,3]]
        def cal_area_x0x1x2(x0, x1, x2, normal):
            #area of triangle (x0, x1, x2) times 2*(length of normal), and it is negative if the triangle is flipped
            x0, x1, x2=x0.detach().to(torch.float64), x1.detach().to(torch.float64), x2.detach().to(torch.float64)
            return (cross(x1-x0, x2-x0, dim=1)*normal).sum(dim=1)
        #area_quad: area of the quad times the same factor, and it is the sum for the two triangles
        area_quad=(normal*normal).sum(dim=1)
        area_x0x1x2=cal_area_x0x1x2(x0, x1, x2, normal)
        is_outside=(area_x0x1x2 <= 0) | (area_x0x1x2 >= area_quad)
        if is_outside.any():
            mask=is_outside.reshape(-1,1)
            x0, x1, x2, x3=torch.where(mask, x1, x0), torch.where(mask, x2, x1), torch.where(mask, x3, x2), torch.where(mask, x0, x3)
            area_x0x1x2[is_outside]=cal_area_x0x1x2(x0[is_outside], x1[is_outside], x2[is_outside], normal[is_outside])
        a = torch.rand(3, n_point, 1, dtype=node.dtype, device=node.device)
        #this comparison has no division, so a quad with zero area is safe
        is_x0x1x2=a[2]*area_quad.reshape(-1,1) < area_x0x1x2.reshape(-1,1)
        y=torch.where(is_x0x1x2, x1, x3)
        u=a[0].sqrt()
        v=a[1]
        x=(1-u)*x0+u*(1-v)*y+u*v*x2
        return x

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
        nodeB=self.node[element].mean(dim=1) #(N,3) => (M,4,3) => (M,3)
        #create new mesh
        node_new=torch.cat([self.node, nodeA, nodeB], dim=0)
        element_new=[]
        element=element.cpu().numpy()
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

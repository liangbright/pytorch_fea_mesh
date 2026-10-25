import pytest

pytestmark = [pytest.mark.geometry]

"""Copy-helper correctness tests for torch_fea_mesh."""
import unittest
from unittest.mock import patch

import numpy as np
import torch

from torch_fea_mesh.mesh.mesh import Mesh


def container(node=None, element=None):
    mesh=Mesh(None, None, None, 'polygon')
    if node is not None:
        mesh.node=node
    if element is not None:
        mesh.element=element
    return mesh


def array(value):
    if torch.is_tensor(value):
        value=value.detach().cpu().numpy()
    return np.asarray(value)


class CopyHelperTest(unittest.TestCase):
    def assert_independent(self, source, result):
        before=array(source).copy()
        if isinstance(result, list):
            result[0][0]=123
        else:
            result[0,0]=123
        np.testing.assert_array_equal(array(source), before)

    def test_node_representation_dtype_and_ownership(self):
        for source_dtype in (np.float32,np.float64):
            values=np.array([[1.125,2.25,3.5],[4.5,5.25,6.125]],dtype=source_dtype)
            for source in (values,torch.from_numpy(values.copy()),values.tolist()):
                mesh=container(node=source)
                for output_type in (None,'list',list,'numpy',np.ndarray,'torch',torch.Tensor):
                    for dtype in (None,torch.float32,torch.float64,np.float32,
                                  np.dtype('float64'),'float32','float64'):
                        with self.subTest(source=type(source),output=output_type,dtype=dtype):
                            result=mesh.make_node_copy(output_type,dtype)
                            expected_type=type(source) if output_type is None else output_type
                            if expected_type in (list,'list'):
                                self.assertIsInstance(result,list)
                                expected=values
                            else:
                                requested=dtype
                                if requested is None:
                                    requested=source_dtype if not isinstance(source,list) else np.float32
                                if isinstance(requested,torch.dtype):
                                    requested={torch.float32:np.float32,torch.float64:np.float64}[requested]
                                expected=values.astype(requested)
                                self.assertEqual(array(result).dtype,expected.dtype)
                                if expected_type in (torch.Tensor,'torch'):
                                    self.assertTrue(torch.is_tensor(result))
                                    self.assertEqual(result.device.type,'cpu')
                                else:
                                    self.assertIsInstance(result,np.ndarray)
                            np.testing.assert_array_equal(array(result),expected)
                            self.assert_independent(source,result)

    def test_element_representation_dtype_and_ownership(self):
        for source_dtype in (np.int32,np.int64):
            values=np.array([[0,1,2],[2,1,3]],dtype=source_dtype)
            for source in (values,torch.from_numpy(values.copy()),values.tolist()):
                mesh=container(element=source)
                for output_type in (None,'list',list,'numpy',np.ndarray,'torch',torch.Tensor):
                    with self.subTest(source=type(source),output=output_type):
                        result=mesh.make_element_copy(output_type)
                        np.testing.assert_array_equal(array(result),values)
                        if not isinstance(result,list):
                            self.assertEqual(array(result).dtype,np.dtype('int64'))
                        if torch.is_tensor(result):
                            self.assertEqual(result.device.type,'cpu')
                        self.assert_independent(source,result)

    def test_tensor_copy_detachment_preserves_source_graph(self):
        leaf=torch.arange(6,dtype=torch.float64,requires_grad=True)
        source=(leaf*2).reshape(2,3)
        mesh=container(node=source)
        for output_type in (None,'torch','numpy','list'):
            result=mesh.make_node_copy(output_type)
            if torch.is_tensor(result):
                self.assertFalse(result.requires_grad)
                self.assertIsNone(result.grad_fn)
            self.assert_independent(source,result)
        source.sum().backward()
        torch.testing.assert_close(leaf.grad,torch.full_like(leaf,2))

    def test_empty_array_shape_and_uninitialized_mesh(self):
        for make in (np.empty,torch.empty):
            mesh=container(node=make((0,3)),element=make((0,4)))
            for output_type in (None,'numpy','torch'):
                self.assertEqual(mesh.make_node_copy(output_type).shape,(0,3))
                self.assertEqual(mesh.make_element_copy(output_type).shape,(0,4))
            self.assertEqual(mesh.make_node_copy('list'),[])
            self.assertEqual(mesh.make_element_copy('list'),[])
        empty=container()
        self.assertEqual(empty.make_node_copy(),[])
        self.assertEqual(empty.make_element_copy(),[])
        self.assertEqual(empty.make_node_copy('torch').dtype,torch.float32)
        self.assertEqual(empty.make_node_copy('numpy').dtype,np.dtype('float32'))

    def test_ragged_list_deep_copy(self):
        source=[[0,1,2],[2,3,4,5]]
        mesh=container(element=source)
        for output_type in (None,list,'list'):
            result=mesh.make_element_copy(output_type)
            self.assertEqual(result,source)
            result[0][0]=99
            self.assertEqual(source[0][0],0)
        for output_type in ('numpy','torch'):
            with self.assertRaises((ValueError,TypeError)):
                mesh.make_element_copy(output_type)

    def test_noncontiguous_readonly_and_non_native_input(self):
        values=np.arange(24,dtype=np.float64).reshape(4,6)
        readonly=values.copy()
        readonly.flags.writeable=False
        for source in (values[:,::2],values[::-1,::2],values.T,readonly,
                       values.astype('>f8'),torch.from_numpy(values)[:,::2]):
            mesh=container(node=source,element=source)
            for output_type in ('numpy','torch'):
                with self.subTest(source=type(source),strides=getattr(source,'strides',None),
                                  output=output_type):
                    nodes=mesh.make_node_copy(output_type,dtype=np.float64)
                    elements=mesh.make_element_copy(output_type)
                    np.testing.assert_array_equal(array(nodes),array(source))
                    np.testing.assert_array_equal(array(elements),array(source).astype(np.int64))
                    self.assert_independent(source,nodes)
                    self.assert_independent(source,elements)

    def test_precision_and_large_index(self):
        nodes=np.array([[1.+1e-10,1e-20,1e20]],dtype=np.float64)
        elements=np.array([[2**53+3,2**62+1,0]],dtype=np.int64)
        for tensor_source in (False,True):
            mesh=container(torch.from_numpy(nodes) if tensor_source else nodes,
                           torch.from_numpy(elements) if tensor_source else elements)
            for output_type in ('numpy','torch','list'):
                np.testing.assert_array_equal(array(mesh.make_node_copy(output_type)),nodes)
                np.testing.assert_array_equal(array(mesh.make_element_copy(output_type)),elements)

    def test_dense_copy_without_python_list_materialization(self):
        class NoListArray(np.ndarray):
            def tolist(self):
                raise AssertionError('array.tolist called')
        values=np.arange(6).reshape(2,3)
        for source in (values.view(NoListArray),torch.from_numpy(values)):
            mesh=container(node=source,element=source)
            with patch.object(mesh,'copy_to_list',side_effect=AssertionError('copy_to_list called')):
                with patch.object(torch.Tensor,'tolist',side_effect=AssertionError('tensor.tolist called')):
                    for output_type in ('numpy','torch'):
                        nodes=mesh.make_node_copy(output_type,dtype=torch.float64)
                        elements=mesh.make_element_copy(output_type)
                        np.testing.assert_array_equal(array(nodes),values)
                        np.testing.assert_array_equal(array(elements),values)

    def test_bfloat16_explicit_numpy_conversion(self):
        source=torch.tensor([[1.25,2.5,3.75]],dtype=torch.bfloat16)
        mesh=container(node=source)
        result=mesh.make_node_copy()
        self.assertEqual(result.dtype,torch.bfloat16)
        self.assertNotEqual(result.data_ptr(),source.data_ptr())
        for dtype in (np.float32,torch.float32,'float64'):
            np.testing.assert_array_equal(mesh.make_node_copy('numpy',dtype),[[1.25,2.5,3.75]])
        with self.assertRaises(TypeError):
            mesh.make_node_copy('numpy')

    def test_invalid_output_type_rejection(self):
        mesh=container(torch.zeros((1,3)),torch.zeros((1,3),dtype=torch.int64))
        for method in (mesh.make_node_copy,mesh.make_element_copy):
            with self.assertRaisesRegex(ValueError,'object_type'):
                method('unsupported')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_source_produces_independent_cpu_copy(self):
        nodes=torch.arange(6,dtype=torch.float64,device='cuda').reshape(2,3).requires_grad_()
        elements=torch.tensor([[0,1,2]],dtype=torch.int64,device='cuda')
        mesh=container(nodes,elements)
        for output_type in ('torch','numpy','list'):
            node_copy=mesh.make_node_copy(output_type)
            element_copy=mesh.make_element_copy(output_type)
            np.testing.assert_array_equal(array(node_copy),array(nodes))
            np.testing.assert_array_equal(array(element_copy),array(elements))
            if torch.is_tensor(node_copy):
                self.assertEqual(node_copy.device.type,'cpu')
                self.assertEqual(element_copy.device.type,'cpu')
                self.assertFalse(node_copy.requires_grad)
            self.assert_independent(nodes,node_copy)
            self.assert_independent(elements,element_copy)

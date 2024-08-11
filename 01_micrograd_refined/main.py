import matplotlib.pyplot as plt
import numpy as np
import math

############
#####################

# visualize graph

from graphviz import Digraph


def trace(root):
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root, format="svg", rankdir="LR"):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ["LR", "TB"]
    nodes, edges = trace(root)
    dot = Digraph(
        format=format, graph_attr={"rankdir": rankdir}
    )  # , node_attr={'rankdir': 'TB'})

    for n in nodes:
        dot.node(
            name=str(id(n)),
            label="{ %s| data %.4f | grad %.4f }" % (n.label, n.data, n.grad),
            # label="{ %s| data %.4f  }" % (n.label, n.data),
            shape="record",
        )
        if n._op:
            dot.node(name=str(id(n)) + n._op, label=n._op)
            dot.edge(str(id(n)) + n._op, str(id(n)))

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot

    ########################################

from typing import Union
import math

class Value:
    def __init__(self,data:float, _children=(), _op:str="", label:str=""):
        self.data = data
        self.grad = 0.0 # mean no effect
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self._backward = lambda:None  # this will be for leaf node
        

    def __repr__(self)->str:
        return f"Value(data={self.data})"
    # __str__ = __repr__

    def __str__(self)->str:
        return f"Value(data={self.data}, label={self.label})"
    
    # def __str__(self):
    #     "return string only"
    #     return f"Value(data={self.data})"

    def __add__(self, other: Union[float, 'Value']) -> 'Value':   # quotaion marks to cater naming issues
        other = other if isinstance(other, Value) else Value(other)
        if (isinstance(other, Value)):
            out =  Value(self.data + other.data, (self, other), "+")

            def _backward():
                self.grad += 1.0 * out.grad
                other.grad +=  1.0 * out.grad
            
            out._backward = _backward
            return out
        else:
            raise TypeError("Operand must be of type 'Value', 'float', 'int")


    def __radd__(self, other: Union[float, 'Value']): #reflected addition
        return self + other
    
    def __neg__(self):
        return Value(self.data * -1)
    
    def __sub__(self, other):
        return self + (-other)


    def __mul__(self, other: Union[float, 'Value']) -> 'Value':   # quotaion marks to cater naming 
        other = other if isinstance(other, Value) else Value(other)

        if (isinstance(other, Value)):

            out = Value(self.data * other.data, (self, other),"*")
            def _backward():
                self.grad += other.data * out.grad
                other.grad += self.data * out.grad
            
            
            out._backward = _backward
            return out
        else:
            raise TypeError("Operand must be of type 'Value', 'float', 'int")


    def __rmul__(self, other: Union[float, 'Value']): #reflected mul
        return self * other
    
    def __truediv__(self, other):
        # div  = a/b  => a* 1/b =? a*b**-1
        return self * other**-1
    
    def __pow__(self, other):
        assert isinstance(other, (int,float)), "only suppoerts int/float powers"
        out = Value(self.data**other,(self,), f'**{other}')

        def _backward():
            self.grad += other* (self.data**(other-1))  * out.grad 
        out._backward = _backward

        return out

    
    def tanh(self):
        n = self.data
        t = (math.exp(2 * n) - 1) / (math.exp(2 * n) + 1)
        out = Value(t, (self,), "tanh")

        def _backward():
            """it will local derivative of tanh() whihch is 1-tanx**2"""
            self.grad += (1 - t**2) * out.grad
            # (1 - t**2) local derivative
            # chain rule: out.grad


        out._backward = _backward
        return out
    

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')
        def _backward():
            self.grad += out.data * out.grad  # derivative of e**x is e**x

        out._backward = _backward

        return out
    






    def backward(self):
        # to calculate backpropagating in order
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0  # o.grad
        
        for node in reversed(topo):
            node._backward()

    


# we'll compute derivative of L with  respect to each node. so in more simpler terms, compute derivative of L with respect to inputs does same things as derivative of L with respect to all intermediate is calculated too-- not skipped.that's what chain rule is all about: Consider intrmediate states.


# we'll just calculate derivative of weights because input data is fixed
# deriavtive of x with itself will be 1

# we'll compute derivative of L with  respect to each node. so in more simpler terms, compute derivative of L with respect to inputs does same things as derivative of L with respect to all intermediate is calculated too-- not skipped.that's what chain rule is all about: Consider intrmediate states.


# we'll just calculate derivative of weights because input data is fixed
# deriavtive of x with itself will be 1
# we'll compute derivative of L with  respect to each node. so in more simpler terms, compute derivative of L with respect to inputs does same things as derivative of L with respect to all intermediate is calculated too-- not skipped.that's what chain rule is all about: Consider intrmediate states.


# we'll just calculate derivative of weights because input data is fixed
# deriavtive of x with itself will be 1
#####################################

import random
from typing import Any,  List

class Neuron:
    def __init__(self, nin) -> None:
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))
        

    def __call__(self, x):
        if len(x) != len(self.w):
            raise ValueError(f"Expected input of length {len(self.w)}, but got {len(x)}.")
        
        act = sum((xi*wi for xi, wi in zip(x , self.w)), start=self.b)
        # print(act)
        out= act.tanh()
        return out
            # print(i.data, j.data)

    def parameters(self):
        return self.w + [self.b]

neu = Neuron(nin =3)
neu.w
x= neu(neu.w)
neu.parameters()





class Layer():
    def __init__(self, nin, nout) -> None:
        self.neurons = [ Neuron(nin=nin) for _ in range(nout)]

    def __call__(self, x) -> Any:
        if len(x) != len(self.neurons[0].w):
            raise ValueError(f"Expected input of length {len(self.neurons[0].w)}, but got {len(x)}.")
        
        outs = [n(x) for n in self.neurons]
        # print(outs)
        return outs[0] if len(outs) ==1 else outs
    
    def parameters(self):
        params =[]
        for neuron in self.neurons:
            ps= neuron.parameters()
            params.extend(ps)
        return params
        

l = Layer(nin =3,nout = 1)
l([1.0,2.0,4.0])





class MLP:
    def __init__(self, nin:int, nouts: List[int]) -> None: # nout of each layer
        sz:list = [nin] + nouts  # [nin, ]
        self.layers: List[Layer] = [ Layer(sz[i], sz[i+1])  for i in range(len(nouts))]
    

    def __call__(self, x) -> Any:
        for layer in self.layers:
            x = layer(x)
        return x
    def parameters(self):
        params =[]
        for layer in self.layers:
            ps= layer.parameters()
            params.extend(ps)
        return params
    
    
# Create an instance of MLP
n = MLP(3, [4, 4 ,1])  # An MLP with 3 inputs, first layer with 4 neurons, second layer with 5 neurons, and third layer with 2 neurons

# Call the MLP with an input vector
x = [2.0, 3.0, -1.0]

print(n(x))  # Print the output




xs =[
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]

]
ys = [1.0, -1.0, -1.0, 1.0]



for k in range(30):
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
    for p in n.parameters():
        p.grad=0.0
    loss.backward()

    # optimization
    for p in n.parameters():
        p.data+= -0.05 * p.grad

    print(k, loss.data)



print(f"\n\nresults: ")

for ygt, yout in zip(ys, ypred):
    print(f"\tygt: {ygt} ; yout:{yout.data}")
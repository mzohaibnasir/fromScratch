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


class Value:
    def __init__(self, data: float, _children=(), _op: str = "", label: str = ""):
        self.data = data
        self.grad = 0.0  # mean no effect
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self) -> str:
        return f"Value(data={self.data})"

    # __str__ = __repr__

    def __str__(self) -> str:
        return f"Value(data={self.data}, label={self.label})"

    # def __str__(self):
    #     "return string only"
    #     return f"Value(data={self.data})"

    def __add__(
        self, other: Union[float, "Value"]
    ) -> "Value":  # quotaion marks to cater naming issues
        other = other if isinstance(other, Value) else Value(other)
        if isinstance(other, Value):
            return Value(self.data + other.data, (self, other), "+")
        else:
            raise TypeError("Operand must be of type 'Value', 'float', 'int")

    def __radd__(self, other: Union[float, "Value"]):  # reflected addition
        return self + other

    def __mul__(
        self, other: Union[float, "Value"]
    ) -> "Value":  # quotaion marks to cater naming
        other = other if isinstance(other, Value) else Value(other)

        if isinstance(other, Value):
            return Value(self.data * other.data, (self, other), "*")
        else:
            raise TypeError("Operand must be of type 'Value', 'float', 'int")

    def __rmul__(self, other: Union[float, "Value"]):  # reflected mul
        return self * other

    def tanh(self):
        n = self.data
        t = (math.exp(2 * n) - 1) / (math.exp(2 * n) + 1)
        out = Value(t, (self,), "tanh")
        return out

    def __neg__(self):
        return Value(-self.data)


# we'll compute derivative of L with  respect to each node. so in more simpler terms, compute derivative of L with respect to inputs does same things as derivative of L with respect to all intermediate is calculated too-- not skipped.that's what chain rule is all about: Consider intrmediate states.


# we'll just calculate derivative of weights because input data is fixed
# deriavtive of x with itself will be 1
#####################################

x1 = Value(2.0, label="x1")
w1 = Value(-3.0, label="w1")
x2 = Value(0.0, label="x2")
w2 = Value(1.0, label="w2")
b = Value(-3.0, label="b")

x1w1 = x1 * w1
x1w1.label = "x1w1"

x2w2 = x2 * w2
x2w2.label = "x2w2"

x1w1x2w2 = x1w1 + x2w2
x1w1x2w2.label = "x1w1 + x2w2"
n = x1w1x2w2 + b
n.label = "n"
o = n.tanh()
print(o)
draw_dot(o)

# MICROGRAD

A tiny scalar-valued autograd engine and a small NN library on top of it with pytorch-like API. It implements back propagtation(reverse-mode auto diff) over a dynamically built DAG and a small NN library on top of it with pytorch-like API. Both are tiny with about 100 and 50 lines of code respectively. The DAG only operates over scalar values, so, we chop up each neuron into all of its individual tiny adds and multiplies. However this is enough to build up entire deep NNs doing binary classification, as the demo notebook shows.

`
Below is a slightly contrived example showing a number of possible supported operations:

## This is the expression graph with two inputs a & b and we creating upto value g.

            from micrograd.engine import Value

            a = Value(-4.0)
            b = Value(2.0)
            c = a + b
            d = a * b + b**3
            c += c + 1
            c += 1 + c + (-a)
            d += d * 2 + (b + a).relu()
            d += 3 * d + (b - a).relu()
            e = c - d
            f = e**2
            g = f / 2.0
            g += 10.0 / f
            print(f'{g.data:.4f}') # prints 24.7041, the outcome of this forward pass
            g.backward()
            print(f'{a.grad:.4f}') # prints 138.8338, i.e. the numerical value of dg/da
            print(f'{b.grad:.4f}') # prints 645.5773, i.e. the numerical value of dg/db

`
Here a and b are input variables and they are transformed into c,d,e,f,g.

Micrgrad supports

1. addition,
2. subtraction,
3. multiplication,
4. devision,
5. power,
6. negate,
7. squash at 0 usingRELU

Micrograd will keep track of everything like c is aalso a value and it is result of addition operation and a & b are child nodes of c.

Micrograd can not only do forward pass i.e. look at value of g.data = 24.7041
but we can also do g.backward() which initialies backpropagation at node g and calculates gradients. So, backprogation will start from g and will go backwards through that expression graph and is going recursively apply chain rule from calculus which mean we'll be evaluating derivative of g w.r.t to each internal nodes like e,d,c and also inputs a,b.

then we can query derivate of g w.r.t a like

    print(f'{a.grad:.4f}') # prints 138.8338, i.e. the numerical value of dg/da

and b

    print(f'{b.grad:.4f}') # prints 645.5773, i.e. the numerical value of dg/db

# so, it tells us how a and b are effecting g.

    so, a.grad= 138 means if I slightly nudge a and make slightly larger then slope of growth for g will be 138. This shows how g will respond if we tweak a alittle in +ve direction,

# '+1' is just distributor of gradients

# + mean both exps have positive effect on ouput

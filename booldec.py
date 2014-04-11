'''booldec, a solver for boolean decision problems that maximize an
objective function, subject to constraints.
'''

# Copyright (c) 2014 Robert Schwarz

import pulp

class Problem(object):
    '''An optimization problem with boolean decisions'''

    def __init__(self, name=''):
        self._vars = {}     # maps names to bool dec vars
        self._obj = {}      # maps names to objective coeffs
        self._conss = set() # contains names of 'True' vars

        # mip model used for actual solving
        self._mip = pulp.LpProblem(name=name, sense=pulp.LpMaximize)
        self._mipvars = {} # maps names to PuLP vars

    def add_var(self, name, obj=0.0, aux=False):
        '''add new binary decision variable'''
        if name in self._vars:
            raise KeyError('Variable %s already exists!' % name)
        bd =  BoolDec(name, aux=aux)
        self._vars[name] = bd
        self._obj[name] = obj
        self._mipvars[name] = pulp.LpVariable(name, cat=pulp.LpBinary)
        return bd

    def __reform(self, expr):
        '''recursively reformulate expression tree.

        for each expression that is not a variable, create an
        auxiliary variable which is returned.
        '''
        if not isinstance(expr, BoolExpr):
            raise ValueError('Can not reformulate %s' % expr)

        name = repr(expr)
        if name in self._vars:
            # expression was alread reformulated
            return self._vars[name]

        res = self.add_var(name, aux=True)

        if isinstance(expr, Not):
            op = self.__reform(expr.operand)
            # one is negation of other
            self._mip += (self._mipvars[name] == 1 - self._mipvars[repr(op)])
        elif isinstance(expr, And):
            ops = [self.__reform(o) for o in expr.operands]
            # any False operand sets result to False
            for op in ops:
                self._mip += (self._mipvars[name] <= self._mipvars[repr(op)])
            # if all are True, then also result
            self._mip += (sum(self._mipvars[repr(op)] for op in ops)
                          <= self._mipvars[name] + len(ops) - 1)
        elif isinstance(expr, Or):
            ops = [self.__reform(o) for o in expr.operands]
            # any True operand sets result to True
            for op in ops:
                self._mip += (self._mipvars[name] >= self._mipvars[repr(op)])
            # if all are False, then also result
            self._mip += (sum(self._mipvars[repr(op)] for op in ops)
                          >= self._mipvars[name])

        return res

    def add_cons(self, expr):
        '''constraint built from expression that must be True'''
        if not isinstance(expr, BoolExpr):
            raise ValueError('Can not build constraint from %s' % expr)

        name = repr(expr)
        if name in self._conss:
            # this constraint was already added
            pass
        elif isinstance(expr, BoolDec):
            # add equation, fixing the MIP variable to 1
            self._mip += (self._mipvars[name] == 1)
            self._conss.add(name)
        else:
            # create variable for expression
            operand = self.__reform(expr)
            self.add_cons(operand)
            
    def solve(self):
        '''solve the defined optimization problem, return status.'''
        # set objective
        obj = sum(v * self._mipvars[k] for k,v in self._obj.iteritems() if v)
        self._mip += obj

        # solve the MIP formulation
        status = self._mip.solve()
        return pulp.LpStatus[status]

    def solution(self):
        '''return set of variables that are True in solution'''
        sol = set()
        for name, var in self._vars.iteritems():
            if var.aux:
                continue
            if pulp.value(self._mipvars[name]) > 0.5:
                sol.add(var)
        return sol

class BoolExpr(object):
    '''Base class used for operator overloading'''

    def __invert__(self):
        return Not(self)

    def __and__(self, other):
        return And(self, other)

    def __or__(self, other):
        return Or(self, other)


class BoolDec(BoolExpr):
    '''A single boolean decision variable.

    Do not create variables directly, but through Problem.add_var()
    method.
    '''
    __slots__ = ()

    def __init__(self, name, aux=False):
        self.name = name
        self.aux = aux

    def __str__(self):
        return self.name

    __repr__ = __str__

class Not(BoolExpr):
    __slots__ = ()

    def __init__(self, operand):
        self.operand = operand

    def __str__(self):
        return '~%s' % self.operand

    def __repr__(self):
        return 'Not(%s)' % repr(self.operand)

class And(BoolExpr):
    __slots__ = ()

    def __init__(self, *args):
        self.operands = tuple(args)

    def __str__(self):
        return '(' + ' & '.join(str(o) for o in self.operands) + ')'

    def __repr__(self):
        return 'And(%s)' % '_'.join(repr(o) for o in self.operands)

class Or(BoolExpr):
    __slots__ = ()

    def __init__(self, *args):
        self.operands = tuple(args)

    def __str__(self):
        return '(' + ' | '.join(str(o) for o in self.operands) + ')'

    def __repr__(self):
        return 'Or(%s)' % '_'.join(repr(o) for o in self.operands)


if __name__ == '__main__':
    p = Problem()
    x = p.add_var('x', 2.0)
    y = p.add_var('y', 3.0)
    p.add_cons(~(x & y))
    p.add_cons(x | ~y)
    status = p.solve()
    print status
    sol = p.solution()
    print sol

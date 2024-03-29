"""The underlying data structure used in gplearn.

The :mod:`gplearn._program` module contains the underlying representation of a
computer program. It is used for creating and evolving programs used in the
:mod:`gplearn.genetic` module.
"""


from copy import copy
#from utilities import calculator
import numpy as np
from sklearn.utils.random import sample_without_replacement

# from .functions import _Function, function_weights
# from .utils import check_random_state

from new_functions import _Function, function_weights
from new_utils import check_random_state


def get_subtree_con(random_state, program):
    '''Return a constant node from a given program,
       which is usually a donor
       often called by donor'''
    const_idx = np.array([program.index(node) if isinstance(node, float) else -1
                     for node in program])
    const_idx = list(np.where(const_idx > 0)[0])
    if not const_idx:
        return None
    else:
        idx = const_idx[random_state.randint(0,len(const_idx))]
        return idx, idx+1 

def get_subtree_var(random_state, program):
    '''Return a variable node from a given program,
       which is usually a donor
       often called by donor'''
    variable_idx = np.array([program.index(node) if isinstance(node, int) else -1
                     for node in program])
    variable_idx = list(np.where(variable_idx > 0)[0])
    if not variable_idx:
        return None
    else:
        idx = variable_idx[random_state.randint(0,len(variable_idx))]
        return idx, idx+1
    
def get_subtree_func(random_state, program):
    '''Return a subtree from a given program
    often called by donor'''
    func_idx = sorted([program.index(node) if isinstance(node, _Function) else -1
                     for node in program])

    idx = func_idx.index(0)
    func_idx = func_idx[idx:]
    start = func_idx[random_state.randint(0,len(func_idx))]
    
    stack = 1
    end = start
    while stack > end-start:
        node = program[end]
        if isinstance(node, _Function):
            stack += node.arity
        end += 1
        
    return start, end 


class _Program(object):

    """A program-like representation of the evolved program.

    This is the underlying data-structure used by the public classes in the
    :mod:`gplearn.genetic` module. It should not be used directly by the user.

    Parameters
    ----------
    function_set : list
        A list of valid functions to use in the program. 

    arities : dict
        A dictionary of the form `{arity: [functions]}`. The arity is the
        number of arguments that the function takes, the functions must match
        those in the `function_set` parameter.
        
    paras : dict
        A dictionary of the form `{para: [functions]}`. The para is the type 
        of arguments that the function takes, the functions must match 
        those in the `function_set` parameter.

    init_depth : tuple of two ints
        The range of tree depths for the initial population of naive formulas.
        Individual trees will randomly choose a maximum depth from this range.
        When combined with `init_method='half and half'` this yields the well-
        known 'ramped half and half' initialization method.

    init_method : str
        - 'grow' : Nodes are chosen at random from both functions and
          terminals, allowing for smaller trees than `init_depth` allows. Tends
          to grow asymmetrical trees.
        - 'full' : Functions are chosen until the `init_depth` is reached, and
          then terminals are selected. Tends to grow 'bushy' trees.
        - 'half and half' : Trees are grown through a 50/50 mix of 'full' and
          'grow', making for a mix of tree shapes in the initial population.

    n_features : int
        The number of features in `X`.

    const_range : tuple of two floats
        The range of constants to include in the formulas.

    metric : _Fitness object
        The raw fitness metric.

    p_point_replace : float
        The probability that any given node will be mutated during point
        mutation.

    parsimony_coefficient : float
        This constant penalizes large programs by adjusting their fitness to
        be less favorable for selection. Larger values penalize the program
        more which can control the phenomenon known as 'bloat'. Bloat is when
        evolution is increasing the size of programs without a significant
        increase in fitness, which is costly for computation time and makes for
        a less understandable final result. This parameter may need to be tuned
        over successive runs.

    random_state : RandomState instance
        The random number generator. Note that ints, or None are not allowed.
        The reason for this being passed is that during parallel evolution the
        same program object may be accessed by multiple parallel processes.

    transformer : _Function object, optional (default=None)
        The function to transform the output of the program to probabilities,
        only used for the SymbolicClassifier.

    feature_names : list, optional (default=None)
        Optional list of feature names, used purely for representations in
        the `print` operation or `export_graphviz`. If None, then X0, X1, etc
        will be used for representations.

    program : list, optional (default=None)
        The flattened tree representation of the program. If None, a new naive
        random tree will be grown. If provided, it will be validated.

    Attributes
    ----------
    program : list
        The flattened tree representation of the program.

    raw_fitness_ : float
        The raw fitness of the individual program.

    fitness_ : float
        The penalized fitness of the individual program.

    oob_fitness_ : float
        The out-of-bag raw fitness of the individual program for the held-out
        samples. Only present when sub-sampling was used in the estimator by
        specifying `max_samples` < 1.0.

    parents : dict, or None
        If None, this is a naive random program from the initial population.
        Otherwise it includes meta-data about the program's parent(s) as well
        as the genetic operations performed to yield the current program. This
        is set outside this class by the controlling evolution loops.

    depth_ : int
        The maximum depth of the program tree.

    length_ : int
        The number of functions and terminals in the program.

    """

    def __init__(self,
                 function_set,
                 arities,
                 paras,
                 init_depth,
                 init_method,
                 n_features,
                 const_range,
                 metric,
                 p_point_replace,
                 parsimony_coefficient,
                 random_state,
                 transformer=None,
                 feature_names=None,
                 program=None):

        self.function_set = function_set
        self.arities = arities
        self.paras = paras
        self.init_depth = (init_depth[0], init_depth[1] + 1)
        self.init_method = init_method
        self.n_features = n_features
        self.const_range = const_range
        self.metric = metric
        self.p_point_replace = p_point_replace
        self.parsimony_coefficient = parsimony_coefficient
        self.transformer = transformer
        self.feature_names = feature_names
        self.program = program

        if self.program is not None:
            if not self.validate_program():
                raise ValueError('The supplied program is incomplete.')
        else:
            # Create a naive random program
            self.program = self.build_program(random_state)

        self.raw_fitness_ = None
        self.fitness_ = None
        self.parents = None
        self._n_samples = None
        self._max_samples = None
        self._indices_state = None

    def build_program(self, random_state):
        """Build a naive random program. 
        
        ## edited by Yan:
        every time we append a function node to the program, we append the 
        type list of the function to the para_stack, when a teriminal is 
        needed, we append check the type needed(0 for constant, 1 for variable)
        
    
        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.
    
        Returns
        -------
        program : list
            The flattened tree representation of the program.
    
        """
        if self.init_method == 'half and half':
            method = ('full' if random_state.randint(2) else 'grow')
        else:
            method = self.init_method
        max_depth = random_state.randint(*self.init_depth)
    
        # Start a program with a function to avoid degenerative programs
        probs = np.array(list(function_weights.values()))
        probs = np.cumsum(probs) / np.sum(probs)
        func = np.searchsorted(probs, random_state.uniform())
        function = list(function_weights.keys())[func]
        #function = random_state.randint(len(self.function_set))
        #function = self.function_set[function]
        if function.para == None:
            func_para = [0,1] if random_state.uniform() < 0.5 else [1,1]
        else:
            func_para = function.para.copy()
        program = [function]
        terminal_stack = [function.arity]
        para_stack = [func_para]
    
        while terminal_stack:
            depth = len(terminal_stack)                      
            choice = self.n_features + len(self.function_set)
            choice = random_state.randint(choice)
            # Determine if we are adding a function or terminal
            if (depth < max_depth) and (method == 'full' or
                                        choice <= len(self.function_set)) and para_stack[-1][-1]:        
                
                probs = np.array(list(function_weights.values()))
                probs = np.cumsum(probs) / np.sum(probs)
                func = np.searchsorted(probs, random_state.uniform())
                function = list(function_weights.keys())[func]                
                #function = random_state.randint(len(self.function_set))
                #function = self.function_set[function]
                if function.para == None:
                    func_para = [0,1] if random_state.uniform() < 0.5 else [1,1]
                else:
                    func_para = function.para.copy()
                program.append(function)
                terminal_stack.append(function.arity)         
                para_stack[-1].pop()
                para_stack.append(func_para)            
            else:
                if para_stack[-1][-1]:
                    terminal = random_state.randint(self.n_features)
                else: 
                    terminal = float(random_state.randint(*self.const_range))
                    if self.const_range is None:
                        # We should never get here
                        raise ValueError('A constant was produced with '
                                         'const_range=None.')
                program.append(terminal)             
                terminal_stack[-1] -= 1
                para_stack[-1].pop()
                while terminal_stack[-1] == 0:
                    terminal_stack.pop()
                    para_stack.pop()
                    if not terminal_stack:
                        return program
                    terminal_stack[-1] -= 1
    
        # We should never get here
        return None

    def validate_program(self):
        """Rough check that the embedded program in the object is valid."""
        terminals = [0]
        for node in self.program:
            if isinstance(node, _Function):
                terminals.append(node.arity)
            else:
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
        return terminals == [-1]

    def __str__(self):
        """Overloads `print` output of the object to resemble a LISP tree."""
        terminals = [0]
        output = ''
        for i, node in enumerate(self.program):
            if isinstance(node, _Function):
                terminals.append(node.arity)
                output += node.name + '('
            else:
                if isinstance(node, int):
                    if self.feature_names is None:
                        output += 'X%s' % node
                    else:
                        output += self.feature_names[node]
                else:
                    output += '%.3f' % node
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
                    output += ')'
                if i != len(self.program) - 1:
                    output += ', '
        return output

    def export_graphviz(self, fade_nodes=None):
        """Returns a string, Graphviz script for visualizing the program.

        Parameters
        ----------
        fade_nodes : list, optional
            A list of node indices to fade out for showing which were removed
            during evolution.

        Returns
        -------
        output : string
            The Graphviz script to plot the tree representation of the program.

        """
        terminals = []
        if fade_nodes is None:
            fade_nodes = []
        output = 'digraph program {\nnode [style=filled]\n'
        for i, node in enumerate(self.program):
            fill = '#cecece'
            if isinstance(node, _Function):
                if i not in fade_nodes:
                    fill = '#136ed4'
                terminals.append([node.arity, i])
                output += ('%d [label="%s", fillcolor="%s"] ;\n'
                           % (i, node.name, fill))
            else:
                if i not in fade_nodes:
                    fill = '#60a6f6'
                if isinstance(node, int):
                    if self.feature_names is None:
                        feature_name = 'X%s' % node
                    else:
                        feature_name = self.feature_names[node]
                    output += ('%d [label="%s", fillcolor="%s"] ;\n'
                               % (i, feature_name, fill))
                else:
                    output += ('%d [label="%.3f", fillcolor="%s"] ;\n'
                               % (i, node, fill))
                if i == 0:
                    # A degenerative program of only one node
                    return output + '}'
                terminals[-1][0] -= 1
                terminals[-1].append(i)
                while terminals[-1][0] == 0:
                    output += '%d -> %d ;\n' % (terminals[-1][1],
                                                terminals[-1][-1])
                    terminals[-1].pop()
                    if len(terminals[-1]) == 2:
                        parent = terminals[-1][-1]
                        terminals.pop()
                        if not terminals:
                            return output + '}'
                        terminals[-1].append(parent)
                        terminals[-1][0] -= 1

        # We should never get here
        return None

    def _depth(self):
        """Calculates the maximum depth of the program tree."""
        terminals = [0]
        depth = 1
        for node in self.program:
            if isinstance(node, _Function):
                terminals.append(node.arity)
                depth = max(len(terminals), depth)
            else:
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
        return depth - 1

    def _length(self):
        """Calculates the number of functions and terminals in the program."""
        return len(self.program)

    def execute(self, X):
        """Execute the program according to X.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        y_hats : array-like, shape = [n_samples]
            The result of executing the program on X.
            
        """
        # Check for single-node programs
        node = self.program[0]
        if isinstance(node, float):
            return np.tile(node, (X.shape[0], X.shape[1]))
        if isinstance(node, int):
            return X[:, :, node]

        apply_stack = []

        for node in self.program:

            if isinstance(node, _Function):
                apply_stack.append([node])
            else:
                # Lazily evaluate later
                apply_stack[-1].append(node)

            while len(apply_stack[-1]) == apply_stack[-1][0].arity + 1:
                # Apply functions that have sufficient arguments
                function = apply_stack[-1][0]
                terminals = [t if isinstance(t, float)
                             else X[:, :, t] if isinstance(t, int)
                             else t for t in apply_stack[-1][1:]]
                intermediate_result = function(*terminals)
                if len(apply_stack) != 1:
                    apply_stack.pop()
                    apply_stack[-1].append(intermediate_result)
                else:
                    return intermediate_result
        
        # We call the calculator function defined in utilities.py,
        # which is the same as the function defination above
        
        # return calculator(self.program, X)

    def get_all_indices(self, n_samples=None, max_samples=None,
                        random_state=None):
        """Get the indices on which to evaluate the fitness of a program.

        Parameters
        ----------
        n_samples : int
            The number of samples.

        max_samples : int
            The maximum number of samples to use.

        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        indices : array-like, shape = [n_samples]
            The in-sample indices.

        not_indices : array-like, shape = [n_samples]
            The out-of-sample indices.

        """
        if self._indices_state is None and random_state is None:
            raise ValueError('The program has not been evaluated for fitness '
                             'yet, indices not available.')

        if n_samples is not None and self._n_samples is None:
            self._n_samples = n_samples
        if max_samples is not None and self._max_samples is None:
            self._max_samples = max_samples
        if random_state is not None and self._indices_state is None:
            self._indices_state = random_state.get_state()

        indices_state = check_random_state(None)
        indices_state.set_state(self._indices_state)

        not_indices = sample_without_replacement(
            self._n_samples,
            self._n_samples - self._max_samples,
            random_state=indices_state)
        sample_counts = np.bincount(not_indices, minlength=self._n_samples)
        indices = np.where(sample_counts == 0)[0]

        return indices, not_indices

    def _indices(self):
        """Get the indices used to measure the program's fitness."""
        return self.get_all_indices()[0]

    def raw_fitness(self, X, y, X_test=None,y_test=None,sample_weight=None):
        """Evaluate the raw fitness of the program according to X, y.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        sample_weight : array-like, shape = [n_samples]
            Weights applied to individual samples.

        Returns
        -------
        raw_fitness : float
            The raw fitness of the program.

        """
        y_pred = self.execute(X)
        if self.transformer:
            y_pred = self.transformer(y_pred)
        raw_fitness = self.metric(y, y_pred, sample_weight)
        
        if X_test is not None and y_test is not None:
            y_pred_test = self.execute(X_test)
            raw_fitness_test = self.metric(y_test,y_pred_test,sample_weight)
            raw_fitness = raw_fitness + ((raw_fitness_test - raw_fitness)**2)**(1/2)

        return raw_fitness
    
    def fitness_coef(self,X,y,pca1,flag):
        coef_tempt = self.coef(X,y,pca1)*self.metric.sign
        val = np.abs(self.execute(X))
        m,n = val.shape
        cc = 0
        for mm in range(m):
            for nn in range(n):
                if val[mm][nn]<=1e-10:
                    cc+=1
        if cc/(m*n)>0.5:
            if flag:
                fitness_coef = -100
            else:
                fitness_coef = 100

        if np.abs(coef_tempt)>0.8:
            if flag:
                fitness_coef = -100
            else:
                fitness_coef = 100
        elif np.abs(coef_tempt)>=0.5 and np.abs(coef_tempt)<=0.8:
            if flag:
                fitness_coef = self.fitness(parsimony_coefficient) - 100*np.abs(coef_tempt)
            else:
                fitness_coef = self.fitness(parsimony_coefficient) + 100*np.abs(coef_tempt)
        else:
            if flag:
                fitness_coef = self.fitness(parsimony_coefficient) - np.abs(coef_tempt)
            else:
                fitness_coef = self.fitness(parsimony_coefficient) + np.abs(coef_tempt)
        return fitness_coef
    

    def fitness(self, parsimony_coefficient=None):
        """Evaluate the penalized fitness of the program according to X, y.

        Parameters
        ----------
        parsimony_coefficient : float, optional
            If automatic parsimony is being used, the computed value according
            to the population. Otherwise the initialized value is used.

        Returns
        -------
        fitness : float
            The penalized fitness of the program.

        """
        if parsimony_coefficient is None:
            parsimony_coefficient = self.parsimony_coefficient
        penalty = parsimony_coefficient * len(self.program) * self.metric.sign
        return self.raw_fitness_ - penalty
    
    #添加因子相关性的惩罚
    def coef(self,X,y,pca1):
        y_pred = self.execute(X)
        #y_programs_pred = [program.execute(X) for program in programs]
        def mycorr(y_pred,y_other_pred):
            cov = ((y_pred - np.expand_dims(y_pred.mean(axis=0), axis=0)) *(y_other_pred - np.expand_dims(y_other_pred.mean(axis=0), axis=0)) ).sum(axis=0)/y_pred.shape[0]
            corr = cov /  ( y_pred.std(axis=0) * y_other_pred.std(axis=0) )
            return np.nan_to_num(np.mean(np.nan_to_num(corr)))
        final_corr = np.mean(mycorr(y_pred,pca1))
        return np.nan_to_num(final_corr)

    def get_subtree(self, random_state, program=None):
        """Get a random subtree from the program.
    
        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.
    
        program : list, optional (default=None)
            The flattened tree representation of the program. If None, the
            embedded tree in the object will be used.
            
    
        Returns
        -------
        start, end : tuple of two ints
            The indices of the start and end of the random subtree.
    
        """
        if program is None:
            program = self.program
        # Choice of crossover points follows Koza's (1992) widely used approach
        # of choosing functions 90% of the time and leaves 10% of the time.
        probs = np.array([0.9 if isinstance(node, _Function) else 0.1
                          for node in program])
        probs = np.cumsum(probs / probs.sum())
        start = np.searchsorted(probs, random_state.uniform())
    
        stack = 1
        end = start
        while stack > end - start:
            node = program[end]
            if isinstance(node, _Function):
                stack += node.arity
            end += 1
    
        return start, end
        

    def reproduce(self):
        """Return a copy of the embedded program."""
        return copy(self.program)

    def crossover(self, donor, random_state):
        """Perform the crossover genetic operation on the program.
    
        Crossover selects a random subtree from the embedded program to be
        replaced. A donor also has a subtree selected at random and this is
        inserted into the original parent to form an offspring.
    
        Parameters
        ----------
        donor : list
            The flattened tree representation of the donor program.
    
        random_state : RandomState instance
            The random number generator.
    
        Returns
        -------
        program : list
            The flattened tree representation of the program.
    
        """
        # Get a subtree to replace
        start, end = self.get_subtree(random_state)
        removed = range(start, end)
        
        if isinstance(self.program[start], _Function):
            # Get a subtree to donate
            donor_start, donor_end = get_subtree_func(random_state, donor)
        elif isinstance(self.program[start], int):
            donor_start, donor_end = get_subtree_var(random_state, donor)
        else:
            if get_subtree_con(random_state, donor) == None:
                return self.program, [], []
            else: 
                donor_start, donor_end = get_subtree_con(random_state, donor)
        
        donor_removed = list(set(range(len(donor))) -
                             set(range(donor_start, donor_end)))
        # Insert genetic material from donor
        return (self.program[:start] +
                donor[donor_start:donor_end] +
                self.program[end:]), removed, donor_removed  

    def subtree_mutation(self, random_state):
        """Perform the subtree mutation operation on the program.

        Subtree mutation selects a random subtree from the embedded program to
        be replaced. A donor subtree is generated at random and this is
        inserted into the original parent to form an offspring. This
        implementation uses the "headless chicken" method where the donor
        subtree is grown using the initialization methods and a subtree of it
        is selected to be donated to the parent.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        # Build a new naive program
        chicken = self.build_program(random_state)
        # Do subtree mutation via the headless chicken method!
        return self.crossover(chicken, random_state)

    def hoist_mutation(self, random_state):
        """Perform the hoist mutation operation on the program.
    
        Hoist mutation selects a random subtree from the embedded program to
        be replaced. A random subtree of that subtree is then selected and this
        is 'hoisted' into the original subtrees location to form an offspring.
        This method helps to control bloat.
    
        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.
    
        Returns
        -------
        program : list
            The flattened tree representation of the program.
    
        """
        program = self.program.copy()
        # Get a subtree to replace
        start, end = get_subtree_func(random_state, program)
        subtree = program[start:end]
        # Get a subtree of the subtree to hoist
        sub_start, sub_end = get_subtree_func(random_state, subtree)
        hoist = subtree[sub_start:sub_end]
        # Determine which nodes were removed for plotting
        removed = list(set(range(start, end)) -
                       set(range(start + sub_start, start + sub_end)))
        return self.program[:start] + hoist + self.program[end:], removed

    def point_mutation(self, random_state):
        """Perform the point mutation operation on the program.
    
        Point mutation selects random nodes from the embedded program to be
        replaced. Terminals are replaced by other terminals and functions are
        replaced by other functions that require the same number of arguments
        as the original node. The resulting tree forms an offspring.
    
        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.
    
        Returns
        -------
        program : list
            The flattened tree representation of the program.
    
        """
        program = self.program.copy()
    
        # Get the nodes to modify
        mutate = np.where(random_state.uniform(size=len(program)) <
                          self.p_point_replace)[0]
    
        for node in mutate:
            if isinstance(program[node], _Function):
                
                if not (program[node].para is None):
                    para = program[node].para.copy()
                    para = [str(p) for p in para]
                    para = ''.join(para)
                else:
                    para = None
                # Find a valid replacement with same arity
                replacement = len(self.paras[para])
                # replacement = len(self.arities[arity])
                replacement = random_state.randint(replacement)
                replacement = list(self.paras[para])[replacement]
                program[node] = replacement
            elif isinstance(program[node], int):
                # We've got a terminal, add variable
                #if self.const_range is not None:
                terminal = random_state.randint(self.n_features)
                program[node] = terminal
            else:
                #terminal = random_state.randint(self.n_features)
                #if terminal == self.n_features:
                terminal = float(random_state.randint(*self.const_range))
                if self.const_range is None:
                        # We should never get here
                    raise ValueError('A constant was produced with '
                                         'const_range=None.')
                program[node] = terminal
    
        return program, list(mutate)

    depth_ = property(_depth)
    length_ = property(_length)
    indices_ = property(_indices)

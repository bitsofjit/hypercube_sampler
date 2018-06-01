#! /usr/bin/python
#
# Author:
# Subhajit Banerjee, UC Davis
# June 2018
#
import numpy as np
from numpy import sin, cos
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, PathPatch
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from itertools import combinations, product
from scipy.stats.distributions import norm
from pyDOE import *
import math
import warnings
import sys
import os.path
from time import clock
from numba import jit
#
#
if (len(sys.argv) != 4):
    print " "
    print " Use: ",sys.argv[0]," <input_file> <output_file> <n_results> "
    print " Insufficient number of command line arguments "
    print " "
    sys.exit()
#
#
class Constraint():
    """Constraints loaded from a file."""

    def __init__(self, fname):
        """
        Construct a Constraint object from a constraints file

        :param fname: Name of the file to read the Constraint from (string)
        """
        with open(fname, "r") as f:
            lines = f.readlines()
        # Parse the dimension from the first line
        self.n_dim = int(lines[0])
        # Parse the example from the second line
        self.example = [float(x) for x in lines[1].split(" ")[0:self.n_dim]]

        # Run through the rest of the lines and compile the constraints
        self.exprs = []
        for i in range(2, len(lines)):
            # support comments in the first line
            if lines[i][0] == "#":
                continue
            self.exprs.append(compile(lines[i], "<string>", "eval"))
        return

    def get_example(self):
        """Get the example feasible vector"""
        return self.example

    def get_ndim(self):
        """Get the dimension of the space on which the constraints are defined"""
        return self.n_dim

    def apply(self, x):
        """
        Apply the constraints to a vector, returning True only if all are satisfied

        :param x: list or array on which to evaluate the constraints
        """
        for expr in self.exprs:
            if not eval(expr):
                return False
        return True
#
#
class Unithypercube():
    """

    d-dimensional, constrained unit-hypercube: [0,1]^d;
    Attributes:
        1. constraint -- an instance of the class Constraint
        2. out_fname -- result output file
        3. n_requested -- Requested number of samples withing the
                          constrained domain

    """
    def __init__(self, inequality_constraint, out_fname, n_requested):
        """
        Setup the hypercube with:
        Input Constraints as read from the constraint input_file,
        output filename where sample points are printed, and
        number of samples requested
        """
        self.constraint = inequality_constraint
        self.out_fname = out_fname
        self.sample_size = n_requested
        self.n_dim = self.constraint.get_ndim()
        self.X = np.zeros([n_requested,self.n_dim])
        self.flag = False
#
    def constraint_latin_hypercube(self, max_iter = 5):
        """

        Generates self.sample_size latin hypercube samples located inside the
        Unithypercube() in d-dimensions subject to the generic inequality
        constraints (linear or not) of the form g(x) >= 0.0 as stored in
        self.constraint. These samples are stored in as the [n x d] array
        self.X.

        This method tries to satisfy all the given constraints within for all
        entries. The maximum number of tries is max_iter.

        ===========
        Rationale
        ===========
        Latin hypercube designs are useful when you need a sample that is
        random but that is guaranteed to be relatively uniformly distributed
        over each dimension

        """
        n_cycle = 1
        # Parameters for latin hypercube sampling
        lhs_crit = 'c'  # 'maximin' ('m'), 'center' ('c'),
                           # 'centermaximin' ('cm'), and 'correlation' ('corr')
        sample_size = self.sample_size

        while n_cycle <= max_iter:
            #
            # Generate new samples from scratch if last step is failed.
            # Reusing old X values not feasible because LHS points
            # depends on number of samples. The output Design scales all
            # the variable ranges from 0.0 to 1.0 which can then be transformed
            # as the user wishes (like to a specific statistical distribution
            # using the scipy.stats.distributions ppf/inverse cumulative
            # distribution function).
            #
            print " "
            print " Generating the latin hypercube samples at Trial # %d" %(n_cycle)
            print " "
            lhd = lhs(self.n_dim, samples = sample_size, criterion = lhs_crit)
            # The following is slower and less efficient in satisfying the constraints
            # lhd = norm(loc=0, scale=1).ppf(lhd)  # this applies to all the dimensions

            true_sample_ind = np.zeros((sample_size), dtype = bool)
            ind = np.arange(sample_size)
            count_pass = 0

            for sample in ind:
                true_sample_ind[sample] = self.constraint.apply(lhd[sample,:])

            # Check number of samples satisfying all the constraints
            is_satisfied = ind[true_sample_ind == True]
            count_pass = len(is_satisfied)

            if count_pass >= self.sample_size:
                # Met the requirement. Remove rejected points
                lhd_final = lhd[is_satisfied,:]
                self.X = lhd_final[0:self.sample_size,:]

                self.flag = True
                print " "
                print " The requested %d samples are generated in %d tries!" \
                        %(self.sample_size, n_cycle)
                print " "
                break
            else:
                n_cycle += 1
                # Did not meet the requirement. Increase sample size and try again
                if count_pass == 0:
                    print " "
                    print " Warning!"
                    print " In Trial # %d NONE of generated latin hypercube samples satisfied all the constraints" %(n_cycle-1)
                    print " "
                    sample_size = 20*sample_size    # sampleSize = 10*sampleSize
                else:
                    # new sample size = 1..20 x old sample size
                    # oversample by 10% to increase chance of sufficient samples
                    print " "
                    print " In Trial # %d: %d of the generated latin hypercube samples did not satisfy the constraints " %((n_cycle - 1), (self.sample_size - count_pass))
                    print " So, moving on to the next trial with bigger sample size. "
                    print " Number of trials left: %d" %(max_iter - n_cycle + 1)
                    print " "
                    sample_size = int(math.ceil(min(20, 1.1*self.sample_size \
                                  /count_pass)*sample_size))
        # END WHILE
        # Remove excess points
        if self.flag:
            print " "
            print " All the samples generated successfully! "
            print " "
        else:
            percnt = int((self.sample_size - count_pass)/self.sample_size)*100
            print " "
            print " Maximum number of iterations (%d) reached" %(max_iter)
            print " Sorry! The program could not generate all the requested samples. "
            print " Only %d%% samples are generated (%d out of %d)" \
                    %(percnt, count_pass, self.sample_size)
            print " "
#
    def write_output(self):
        if self.flag:
            print " "
            print " Writing all the requested samples to %s" %(self.out_fname)
            print " "
        else:
            print " "
            print " Writing the partially computed sample list to %s" %(self.out_fname)
            print " "
        np.savetxt(self.out_fname, self.X, delimiter=', ', fmt='%6.4f')
#
#
class arrow_3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs
#
    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

# jit decorator tells Numba to compile this function.
# The argument types will be inferred by Numba when function is called.
@jit
def main():
    t1 = clock()
    input_filename = sys.argv[1]
    output_filename = sys.argv[2]
    n_results = int(sys.argv[3])
    #
    print " "
    print " Setting up the constraints as input from the file %s " \
            %(input_filename)
    print " "
    input_constraint = Constraint(input_filename)
    n_constraint = len(input_constraint.exprs)    # number of constraints
    print " "
    print " Setting up the Hypercube in dimension %d: " \
            %(input_constraint.get_ndim())
    print " "
    constrained_hypercube = Unithypercube(input_constraint, output_filename, n_results)
    print " "
    print " In a %d-dimensional unit hypercube total number of Constraints as read \
    from the file %s is: %d" %(constrained_hypercube.n_dim, input_filename, \
                               n_constraint)
    print " "
    constrained_hypercube.constraint_latin_hypercube()
    constrained_hypercube.write_output()
    t2 = clock() - t1
    if t2 <= 60.0:
        print " "
        print " Elapsed CPU time to complete execution is %10.6f seconds " % (t2)
        print " "
    else:
        print " "
        print " Elapsed CPU time to complete execution is %10.6f minutes " % (t2/60.0)
        print " "
    #
    # Plot the vectors for 3D problem (only for <= 100 samples)
    if (constrained_hypercube.n_dim == 3) and (n_results <= 100):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_aspect("auto")
        ax.set_autoscale_on(True)

        # Draw the Cube
        # r = [0, 1]
        # for s, e in combinations(np.array(list(product(r,r,r))), 2):
        #     if np.sum(np.abs(s-e)) == r[1]-r[0]:
        #         ax.plot3D(*zip(s,e), color="b")

        # Draw the Points
        # ax.scatter([0],[0],[0], color="g",s=100)

        if os.path.isfile(output_filename):
            # Draw the vectors
            vectors = np.loadtxt(output_filename, delimiter=', ', )
            for v in xrange(n_results):
                arrw = arrow_3D([0,vectors[v,0]],[0,vectors[v,1]],[0,vectors[v,2]], \
                                mutation_scale=20, lw=0.5, arrowstyle="-|>", \
                                color=np.random.rand(3,)) # color="r")
                ax.add_artist(arrw)
            plt.savefig('3D_example.eps', format='eps', dpi=500)
            plt.show()
        else:
            print " "
            warnings.warn(" The 3D output file does not exist! Aborting . . .")
            print " "
            sys.exit()
#
# Call main
if __name__ == "__main__":
    main()

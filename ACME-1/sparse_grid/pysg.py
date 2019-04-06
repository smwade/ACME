import math, copy

class gridPoint: 
  """ position of a grid point, 
      also stores function value """
  def __init__(self, index=None, domain=None):
    self.hv = [] # hierarchical value
    self.fv = [] # function value
    self.coeff = 0. # This was added in.
    if index is None:
      self.pos = [] # position of grid point
    else:
      self.pos = self.pointPosition(index, domain)
  
  def pointPosition(self, index, domain=None):
    coord = list()
    if domain is None:
      for i in range(len(index)/2):
        coord.append(index[2*i+1]/2.**index[2*i])
    else:
      for i in range(len(index)/2):
        coord.append((domain[i][1] - domain[i][0]) \
                    *index[2*i+1] / 2.**index[2*i] + domain[i][0])
    return coord
  
  def printPoint(self):
    if self.pos is []:
      pass
    else:
      out = ""
      for i in range(len(self.pos)):
        out += str(self.pos[i]) + "\t"
      print out

class sparseGrid: 
  """ A sparse grid of a certain level consists of a set of indices and 
      associated grid points gP on a given domain of dimension dim.
      Action is what happens when one traverses the sparse grid.
  """
  def __init__(self,dim=1,level=1):
    self.dim = dim
    self.level = level
    self.gP = {} # hash, indexed by tuple(l_1,p_1,l_2,p_2,...,l_d,p_d)
    self.indices = [] # entries: [l_1,p_1,...,l_d,p_d], level,position
    self.domain = ((0.,1.0),)*dim
    self.action = ()
    self.generatePoints()
              
  def printGrid(self):
    print self.hSpace
              
  def evalAction(self):
    basis = copy.deepcopy(self.evalPerDim[0][self.hSpace[0]-1][0])
    value = self.evalPerDim[0][self.hSpace[0]-1][1]
    # compute index and its value on x of the one non-zero basis function
    # in this hierarchical sup-space
    for i in range(1,self.dim):
      value *= self.evalPerDim[i][self.hSpace[i]-1][1]
      basis += self.evalPerDim[i][self.hSpace[i]-1][0]
    # add contribution of this hierarchical space
    self.value += self.gP[tuple(basis)].hv*value
  
  def evalFunct(self,x):
    """ evaluate a sparse grid function, hierarchival values have to be set """
    self.value = 0.0
    self.evalPerDim = []
    # precompute values of one dim basis functions at x for the evaluation
    for i in range(self.dim):
      self.evalPerDim.append([])
      for j in range(1,self.level+1):
        # which basis is unzero on x for dim i and level j
        pos = (x[i]-self.domain[i][0])/(self.domain[i][1] \
                                        -self.domain[i][0])
        basis = int(math.ceil(pos*2**(j-1))*2-1)
        # test needed for x on left boundary
        if basis == -1: 
          basis = 1
          self.evalPerDim[i].append([[j,basis]])
        else:
          self.evalPerDim[i].append([[j,basis]])
        # value of this basis function on x[i]
        self.evalPerDim[i][j-1].append(evalBasis1D(x[i],\
                self.evalPerDim[i][j-1][0],self.domain[i]))
    self.action = self.evalAction
    self.loopHierSpaces()
    return self.value

  def loopHierSpaces(self):
    """ go through the hierarchical subspaces of the sparse grid """
    for i in range(1,self.level+1):
      self.hSpace = [i]
      self.loopHierSpacesRec(self.dim-1,self.level-(i-1))
  
  def loopHierSpacesRec(self,dim, level):
    """ d-dimensional recursion through all hierarchical subspaces """
    if dim > 1:
      for i in range(1,level+1):
        self.hSpace.append(i)
        self.loopHierSpacesRec(dim-1,level-(i-1))
        self.hSpace.pop()
    else:
      for i in range(1,level+1):
        self.hSpace.append(i)
        self.action()
        self.hSpace.pop()

  def generatePoints(self):
    """ fill self.gP with the points for the indices generated beforehand """
    # generate indices of grid points for the given level and dim
    self.indices = self.generatePointsRec(self.dim,self.level)
    # add positions of sparse grid points
    for i in range(len(self.indices)):
      self.gP[tuple(self.indices[i])] = gridPoint(self.indices[i],self.domain)

  def generatePointsRec(self,dim, level, cur_level=None):
    """ run over all hierarchical subspaces and add all their indices """
    basis_cur = list()
    if cur_level == None:
      cur_level = 1
    # generate all 1-D basis indices of current level (i.e. step 2)
    for i in range (1,2**(cur_level)+1,2):
      basis_cur.append([cur_level,i])
    if dim == 1 and cur_level == level:
      return basis_cur # we have all
    elif dim == 1: # generate some in this dim for higher level
      basis_cur += self.generatePointsRec(dim,level,cur_level+1)
      return basis_cur
    elif cur_level == level: 
      #crossproduct of this dim indices and other (dim-1) ones
      return cross(basis_cur,\
                    self.generatePointsRec(dim-1,level-cur_level+1))
    else:
      #crossproduct of this dim indices and other (dim-1) ones
      #since levels left, generate points for higher levels
      return cross(basis_cur,self.generatePointsRec(dim-1,\
                    level-cur_level+1)) \
                    + self.generatePointsRec(dim,level,cur_level+1)
  
  def nodal2Hier1D(self,node,i,j,dim):
    """ conversion from nodal to hierarchical basis in one dimension
        (i,j) gives index in this dim current node
        node is the (d-1) index to treat """
    # get left/right neighbours of node
    left = [i-1,j/2]
    right = [i-1,j/2+1]
    # left, right can be points of upper level (if index is even)
    while left[1]%2 == 0 and left[0] > 0:
      left = [left[0]-1,left[1]/2]
    while right[1]%2 == 0 and right[0] > 0:
      right = [right[0]-1,right[1]/2]
    # index of node is multi-dimensional
    if len(node) > 2:
      # build d-dim index for current node and its neighbours
      preCurDim  = node[0:2*dim]
      postCurDim = node[2*dim:len(node)+1]
      index = preCurDim + [i,j] + postCurDim
      left  = preCurDim + left  + postCurDim
      right = preCurDim + right + postCurDim
    else:
      #this case can only happen in 2D
      if dim == 0:
        index = [i,j] + node
        left  = left  + node
        right = right + node
      else: 
        index = node + [i,j]
        left  = node + left 
        right = node + right

    #in case we are on the left boundary
    if left[2*dim] == 0:
      if right[2*dim] != 0:
        self.gP[tuple(index)].hv -= 0.5*self.gP[tuple(right)].hv
    elif right[2*dim] == 0: #or the right boundary
      self.gP[tuple(index)].hv -= 0.5*self.gP[tuple(left)].hv
    else: #normal inner node
      self.gP[tuple(index)].hv -= 0.5*(self.gP[tuple(left)].hv + self.gP[tuple(right)].hv)
  
  def nodal2Hier(self):
    """ conversion from nodal to hierarchical basis """
    for i in range(len(self.indices)):
      self.gP[tuple(self.indices[i])].hv = self.gP[tuple(self.indices[i])].fv
    # conversion is done by succesive one-dim conversions
    for d in range(0,self.dim):
      for i in range(self.level,0,-1):
        # generate all indices to process
        indices = self.generatePointsRec(self.dim-1,self.level-i+1)
        for j in range(1,2**i+1,2):
          for k in range(len(indices)):
            self.nodal2Hier1D(indices[k],i,j,d)
    
  def setFunctionValues(self,f):
    """ Set the value of the function at each of the grid points """
    # This was added in and has problems
    self.f = f
    for i in xrange(len(self.indices)):
      self.gP[tuple(self.indices[i])].fv = f(self.gP[tuple(self.indices[i])].pos)

  def phi(self,p,loc):
    """ This computes the value of the basis function p at loc """
    # This was added in and has problems
    def sphi(x):
      return max(0,1-abs(x))
    def phiji(j,i,x):
      return sphi(2**(j-1)*x-(2*(i-j)+1))
    m = 1.
    for i in xrange(self.dim):
      t = phiji(p[2*i],p[2*i+1],loc[i])
      if t < m:
          m = t
    return m

  def setCoefficients(self):
    """ Set the coefficient for each basis hat function """
    # This was added in and has problems.
    for i in xrange(len(self.indices)):
      level = self.indices[i]
      loc = self.gP[tuple(level)].pos
      difference = self.f(loc)
      lesser_pts = []
      for j in xrange(i-1,-1,-1):
        p = self.indices[j]
        #print level[::2],p[::2]
        for k in xrange(self.dim):
          if (level[::2][k]) > (p[::2][k]):
            lesser_pts.append(p)
            difference -= self.gP[tuple(p)].coeff*self.phi(p,loc)
            #print True
            break
      self.gP[tuple(level)].coeff = difference

  def plotGrid(self):
    from matplotlib import pyplot as plt
    from numpy import empty as npempty
    pts = len(self.indices)
    xpts = npempty(pts)
    if self.dim==1:
      for i in xrange(pts):
        pt = tuple(self.indices[i])
        xpts[i] = self.gP[pt].pointPosition(pt)[0]
      ax = plt.gca()
      ax.yaxis.set_visible(False)
      plt.plot(xpts,xpts/xpts,'*')
    elif self.dim==2:
      ypts = npempty(pts)
      for i in xrange(pts):
        pt = tuple(self.indices[i])
        xpts[i], ypts[i] = self.gP[pt].pointPosition(pt)
      plt.plot(xpts,ypts,'*')
    else:
      if self.dim > 3:
        print "Showing first three dimensions only"
      from mpl_toolkits.mplot3d import Axes3D
      Axes3D
      fig = plt.figure()
      ax = fig.add_subplot(111, projection='3d')
      ypts = npempty(pts)
      zpts = npempty(pts)
      for i in xrange(pts):
        pt = tuple(self.indices[i])
        xpts[i], ypts[i], zpts[i] = self.gP[pt].pointPosition(pt)[:3]
      ax.scatter(xpts, ypts, zpts)    
    plt.show()


def cross(*args):
  """ compute cross-product of args """
  ans = []
  for arg in args[0]:
    for arg2 in args[1]:
      ans.append(arg+arg2)
  return ans
  #alternatively:
  #ans = [[]]
  #for arg in args:
    #ans = [x+y for x in ans for y in arg]
  
  #return ans

def evalBasis1D(x, basis,interval=None):
  """ evaluation of the basis functions in one dimension """
  if interval is None:    
    return 1. - abs(x*2**basis[0]-basis[1])
  else:
    pos = (x-interval[0])/(interval[1]-interval[0])
    return 1. - abs(pos*2**basis[0]-basis[1])
import copy

class Operation():
    #self.op = None
    #self.dirty = None
    
    def __init__(self, op, dirty):
        #TODO add an ID for printing
        self.op = op
        self.dirty = dirty
    
    def __str__(self):
        return self.op[0]+"("+", ".join(repr(e) for e in self.op[1:])+")"+" (dirty)"*self.dirty
    
    def write(self, cir):
        func = self.op[0] # this might not work, might have to pass cir.func and then can drop cir as a parameter
        getattr(cir, func)(*self.op[1:])

class SafeCircuit():
    written = dict()
    
    #self.cir = None
    
    def __init__(self, cir):
        self.cir = cir
        self.l = []
        self.oplist = []
        SafeCircuit.written[self.cir.name] = False # todo some way to hash it
    
    def add_op(self, *op, dirty=False): # advanced: isoutput should be set only on an output bit
        self.l.append(Operation(op, dirty))
        # we should be able to get away with only one level, 'dirty'.  Other levels should not be set as they would prevent uncomputation unless they are in the outermost circuit.  If they are computed in a loop, it is acceptable to measure them before the next one.
        
    def add_cir(self, subcir): # don't carry through dirty.  The dirty will be reverted as the others are finalized
        subops = subcir.finalize()
        for op in subops:
            op2 = copy.deepcopy(op)
            op2.dirty = False
            # do these actually need to be uncomputed at the end, or can they all be marked as dirty?
            # I think they all need to be to reverse the central one that is dirty, so this is right
            self.l.append(op2)
        
    def blind_concat(self, other):
        self.l += copy.deepcopy(other.l)
    
    def __iter__(self):
        for i in range(len(self.oplist)):
            yield self.oplist[i]
    
    def finalize(self):
        oplist = []
        
        for i in range(len(self.l)):
            op = self.l[i].op
            oplist.append(self.l[i])
        
        for i in range(len(self.l)-1, -1, -1):
            # dirty or not is ignored when executing individually
            if not self.l[i].dirty:
                op = self.l[i].op
                oplist.append(self.l[i])
        
        self.oplist = oplist
        return oplist
    
    def dowrite(self):
        for op in self.oplist:
            op.write(self.cir)
    
    def write(self):
        #TODO check if the self.cir is empty before writing
        
        if SafeCircuit.written[self.cir.name]:
            raise Exception("You have already written a SafeCircuit to this QuantumCircuit once; operations would not be safely uncomputed. If you want to run multiple circuits, use the add_cir method to add one to the other. Exiting")
        
        self.finalize()
        
        SafeCircuit.written[self.cir.name] = True
        self.dowrite()
        


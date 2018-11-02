'''
Created on 2018/10/27

@author: Han Zhang
'''
import logging
from enum import Enum
import graph
import numpy as np
import scipy as sc
import time
import random
from collections import defaultdict

class Status(Enum):
    INSTANTIATED = 0
    INITIALIZED = 1
    INTRAINING = 2
    FILESAVED = 3


logger = logging.getLogger("distributed learning")
class Distributed(object):
    '''
    A distributed learning simulator with multi-threads
    '''
    


    def __init__(self, graph):
        '''
        Constructor of a distributed learning simulator
        users can instantiate the object by two ways
        1.using the default settings by just call self.initDefault()
        2.using the customized settings by passing parameters to sefl.setArgs(...)
        '''
        self.graph = graph
        self.numNodes = self.graph.number_of_nodes()
        self.status = Status.INSTANTIATED
        logger.info("Instantiated the object with in-passing graph")
        self.printInProgress(status=self.status)
       
        
        
    def process(self):
        
        self.status = Status.INTRAINING
        logger.info("Start Training\n")
        self.trainingThread()
        
    
    
    def defaultArgs(self):
        '''
        1. initializing default parameters
        2. initializing the adjacent lists to store neighboring vertices,
        call the method graph.extent(distance), using BFS
        parameters:
        @distance:  the distance between the vertex that is being updated and the target vertex.
        This parameter plays the same role of the window in Skipgram
        @alpha:     the initial learning rate, may be self-adjusted in learning progress
        @numUpdates: the total number of a simple synchronized updating
        @numNegSampling: the number of samples when doing negative sampling
        @numTread:    the number of threads participating the training process
        @representSize: the dimension of the representation vector
        @numNodes:    the number the vertices in the graph
        @outputPath: the output file path and name
        '''
        self.distance = 1
        self.alpha = 0.025
        self.numUpdates = 50
        self.numNegSampling = 3
        self.maxNegLen = 10**8
        self.numThread = 4
        self.representSize = 64
        t = time.localtime(time.time())
        self.outputPath = "{}{}{}{}{}{}".format(t[1],t[2],t[0],t[3],t[4],t[5]) + ".embeddings"
        #self.outputPath = "output.embeddings"
         
        self.initStorage()
        self.status = Status.INITIALIZED
        
        logger.info("Set parameters by default")
        self.printInProgress(status=self.status)
        
    def setArgs(self,distance,alpha,numUpdates,numNegSampling,
                maxNeglen,numThread,representSize,outputPath):
        '''
        set the parameters of the training process
        '''
        self.distance = distance
        self.alpha = alpha
        self.numUpdates = numUpdates
        self.numNegSampling = numNegSampling
        self.maxNeglen = maxNeglen
        self.numThread = numThread
        self.representSize = representSize
        self.outputPath = outputPath
        
        self.initStorage()
        self.status = Status.INITIALIZED
        logger.info("Set customized parameters")
        self.printInProgress(status=self.status)
        
    def initStorage(self):
        #allocating memory space for the main representation
        t0 = time.time()
        print("start allocating storages")
        self.repMatrix = defaultdict(np.array)
        for i in self.graph.keys():
            self.repMatrix[i] = np.zeros(self.representSize)
        #set initial value, following the method of gensim.models.Word2Vec
        for j in self.repMatrix.keys():
            for k in range(self.representSize):
                self.repMatrix[j][k] = (random.random()-0.5)/self.representSize
        t1 = time.time()
        print("finished setting initial values for representation matrix, time: {} min".format((t1-t0)/60))
        t0 = t1
        
        
        #storage for negative sampling, here we use degree of the vertex to take place of count
        self.expTable = defaultdict(float)
        maxExpValue = 0
        for n in self.repMatrix.keys():
            currentExp = pow(self.graph.degree(n),0.75)
            maxExpValue += currentExp
            self.expTable[n] = currentExp
        i = 0
        d = 0
        self.negSampTable=[]
        for key, exp in self.expTable.items():
            d += exp/maxExpValue
            while float(i)/self.maxNegLen < d:
                self.negSampTable.append(key)
                i = i+1
            
        #setting a reference table which is involved in negative sampling
        '''
        for m in range(self.maxNegLen):
            self.negSampTable.append(i)
            if m/float(self.maxNegLen) > d:
                i = i+1
                d += self.expTable[l]/maxExpValue
                l = l+1
            if l>=self.numNodes:
                l = self.numNodes-1
        '''
        
        t1 = time.time()
        print("finished setting initial values for negsampling table, time: {} min".format((t1-t0)/60))
        t0 = t1
   
        #storage for adjacent table which records adjacent vertices within the distance for each vertex
        self.adjTable = defaultdict(list)
        #storage for the weights of adjacent table, for 2nd or 3rd degree neighbors who have lighter weights
        self.weightTable = defaultdict(list)
        for key, vtx in self.graph.items():
            adjlist=[]
            weilist=[]
            cur=0
            for nds in vtx:
                adjlist.append(nds)
            primelen = len(adjlist)
            lastloc = cur
            weilist.append([len(adjlist),1])        #the first element indicates the 1st location after nodes of this level
            if self.distance>1:
                for cn in range(1,self.distance):
                    for pos in range(lastloc,len(adjlist)):
                        adj = self.graph[adjlist[pos]]
                        for nodes in adj:
                            if nodes == key:
                                continue
                            if not nodes in adjlist:
                                adjlist.append(nodes)
                        cur = cur + 1
                    if not len(adjlist) == cur:
                        weilist.append([len(adjlist),float(primelen)/(len(adjlist)-cur)])
                    lastloc = cur
            self.adjTable[key] = adjlist
            self.weightTable[key] = weilist
            #print("finish, adjlen = {}".format(len(adjlist)))
        t1 = time.time()   
        print("finished setting initial values for adjacency table, time:{} min".format((t1-t0)/60))
    
    def trainingThread(self):
        '''
        a single tread function performing training process
        a modified distributed version of skip-gram with negative sampling
        each node is maintained by an agent, which receives representations of 
        adjacent vertices and updates the current nodes, and also send 
        representations of itself to limited number of vertices by request, 
        but refuse to share it with any centralized agent 
        
        v1.0: read representation matrix from memory, single thread, no disk-access
        alpha self-adjusted based on progress (this property is inherited from 
        gensim.model.Word2Vec), fixed distance (window size)
        '''
        currentProgress = 0
        while currentProgress < self.numUpdates:
            localalpha = self.alpha
            nodesSeq = list(self.graph.keys())
            random.shuffle(nodesSeq)
            #adjust local alpha wait to code
            localalpha = self.alpha * (1 - float(currentProgress)/self.numUpdates)
            if localalpha < self.alpha * 0.0001:
                localalpha = self.alpha * 0.0001
            for centralNode in nodesSeq:
                if not len(self.adjTable[centralNode]) == 0:
                    level = 0  
                    thredhold = self.weightTable[centralNode][level][0]
                    wei = self.weightTable[centralNode][level][1]
                    for index, friend in enumerate(self.adjTable[centralNode]):
                        if index >= thredhold:
                            level = level+1
                            if level >= len(self.weightTable[centralNode]):
                                level = len(self.weightTable[centralNode])-1
                            thredhold = self.weightTable[centralNode][level][0]
                            wei = self.weightTable[centralNode][level][1]
                        #positive updating from nodes within the distance
                        self.singleUpdate(centralNode, friend, localalpha, 0, wei)
                        
                        #sampling enemies
                        maxint64 = 2**64-1
                        k=0
                        while True:
                            seed = random.randrange(maxint64)
                            enemy = self.negSampTable[seed%self.maxNegLen]
                            if enemy == friend:
                                continue
                            else:
                                k = k+1
                            
                            self.singleUpdate(centralNode,enemy,localalpha,1,wei)
                            if k == 5:
                                break
            logger.info("Finished the {} th updating".format(str(currentProgress)))
            currentProgress += 1
            self.printInProgress(float(currentProgress)*100/self.numUpdates, self.status)
                    
    def singleUpdate(self,center,target,alpha,mode,weight):
        '''
        a single update process on the center node from target node
        @center : the index of the center node
        @target : the index of the target node
        @alpha  : the current alpha value
        @mode   : 1 means negative updating while 0 means positive updating
        @weight : the weight of the updating
        '''
        sigma = np.dot(self.repMatrix[center],self.repMatrix[target])
        sigma = 1.0/(1 + np.exp(-sigma))
        if mode == 0:
            g = alpha * (1-sigma) * self.repMatrix[target]
            self.repMatrix[center] += g * weight
        else:
            g = -alpha * sigma * self.repMatrix[target]
            self.repMatrix[center] += g * weight
            
    
    
     
    def save2File(self):
        '''
        save the representation matrix to the disk
        '''
        try:
            file = open(self.outputPath,'w')
            file.write("{} {}\n".format(str(self.numNodes),str(self.representSize)))
            for index, nodes in self.repMatrix.items():
                file.write(str(index)+" ")
                for num in np.nditer(nodes):
                    file.write(str(num)+" ")
                file.write("\n")
        finally:
            file.close()
            
        self.status = Status.FILESAVED   
        self.printInProgress(status=self.status)
        logger.info("embedding saved\n")
    
    def printInProgress(self,percentage=0,status=0):
        '''
        A help function helping printing the current state of the process
        @percentage: the progress of training
        @status    : the current status 
        '''
        if status == Status.INTRAINING:
            print("Training in progress, {}%\n".format(percentage))
        else:
            if status == Status.INSTANTIATED:
                print("object has just been instantiated, please set parameters before training\n")
            if status == Status.INITIALIZED:
                print("ready to train the model\n")
            if status == Status.FILESAVED:
                print("Created target file ./{}\n".format(self.outputPath))

    
    
    
    
    
    
    
    
    
    
    
    
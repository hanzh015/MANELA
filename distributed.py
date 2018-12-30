'''
Created on 2018/10/27

@author: Han Zhang

The main class for distributed network embeddings learning algorithm.
'''
import logging
from enum import Enum
import graph
import numpy as np
import scipy as sc
import time
import random
import math
from collections import defaultdict
from numpy import int64

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
        if self.poisson is not True:
            self.trainingThread()
        else:
            self.poissonTraining()
        
    
    
    def defaultArgs(self):
        '''
        1. initializing default parameters
        2. initializing the adjacent lists to store neighboring vertices,
        call the method graph.extent(distance), using BFS
        parameters:
        @alpha:     the initial learning rate, may be self-adjusted in learning progress
        @numUpdates: the total number of a simple synchronized updating
        @numNegSampling: the number of samples when doing negative sampling
        @numTread:    the number of threads participating the training process
        @representSize: the dimension of the representation vector
        @numNodes:    the number the vertices in the graph
        @outputPath: the output file path and name
        '''
        self.alpha = 0.025
        self.numUpdates = 100
        self.numNegSampling = 3
        self.maxNegLen = 10**8
        self.representSize = 128
        self.ratio = 0.5
        self.fmax = 1
        self.continued = False
        self.poisson = False
        self.window = 10
        self.timeslot = 1000
        self.cpath=""
        t = time.localtime(time.time())
        self.outputPath = "examples\\embeddings\\U={}_N={}_R={}_timestamp={}{}{}".format(
            self.numUpdates,self.numNegSampling,self.representSize,
            t[3],t[4],t[5]) + ".embeddings"
        #self.outputPath = "output.embeddings"
         
        self.initStorage()
        self.status = Status.INITIALIZED
        
        logger.info("Set parameters by default")
        self.printInProgress(status=self.status)
        
    def setArgs(self,alpha=0.025,numUpdates=100,numNegSampling=5,
                maxNeglen=10**8,representSize=128,outputPath='examples\\embeddings\\tempemb.embeddings',
                ratio=0.2,fmax=1,c=False,cpath='',poisson=True,window=10,timeslot=1000):
        '''
        set the parameters of the training process
        '''
        self.alpha = alpha
        self.numUpdates = numUpdates
        self.numNegSampling = numNegSampling
        self.maxNegLen = maxNeglen
        self.representSize = representSize
        self.outputPath = outputPath
        self.ratio = ratio
        self.fmax = fmax
        self.continued = c
        self.poisson = poisson
        self.cpath = cpath
        self.window = window
        self.timeslot =  timeslot
        
        self.initStorage()
        self.status = Status.INITIALIZED
        logger.info("Set customized parameters")
        self.printInProgress(status=self.status)
        
    def initStorage(self):
        #allocating memory space for the main representation
        t0 = time.time()
        print("start allocating storages")
        self.repMatrix = defaultdict(np.array)
        if not self.continued:
            for i in self.graph.keys():
                self.repMatrix[i] = (np.random.ranf(self.representSize)-0.5)/self.representSize
        else:
            print("continued training")
            matrix = np.loadtxt(self.cpath,delimiter=" ",skiprows=1,usecols=range(self.representSize+1))
            for i in range(len(self.graph)):
                self.repMatrix[int(matrix[i][0])] = matrix[i][1:]
        
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
            newi = math.floor(self.maxNegLen * d)
            self.negSampTable.extend([key] * (newi - i + 1))
            i = newi
            
        #setting a reference table which is involved in negative sampling
        
        
        t1 = time.time()
        print("finished setting initial values for negsampling table, time: {} min".format((t1-t0)/60))
        t0 = t1
   
        #storage for adjacent table which records adjacent vertices within the distance for each vertex
        self.adjTable = defaultdict(list)
        #storage for the weights of adjacent table, for 2nd or 3rd degree neighbors who have lighter weights
        for key, vtx in self.graph.items():
            adjlist = list(vtx)
            self.adjTable[key] = adjlist
                  
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
        fratio = (self.ratio/0.5)*self.fmax 
        sratio = (2-self.ratio/0.5)*self.fmax
        if fratio>1:
            fover=fratio-1
            fratio=1
        else:
            fover = 0
        while currentProgress < self.numUpdates:
            localalpha = self.alpha
            nodesSeq = list(self.graph.keys())
            random.shuffle(nodesSeq)
            #adjust local alpha wait to code
            localalpha = self.alpha * (1 - float(currentProgress)/self.numUpdates)
            if localalpha < self.alpha * 0.0001:
                localalpha = self.alpha * 0.0001
            for centralNode in nodesSeq:
                #random sample 2nd degree neighbors with the same quantity
                primelen = len(self.adjTable[centralNode])
                #seclist = []
                seclist = [random.choice(self.graph[
                    random.choice(self.adjTable[centralNode])
                    ]) for _ in range(int(primelen*sratio))]
                '''
                for _ in range(primelen):
                    first = random.choice(self.adjTable[centralNode])
                    second = random.choice(self.graph[first])
                    seclist.append(second)
                '''
                #self.adjTable[centralNode].extend(seclist)
                
                #updating by each agent
                for friend in random.sample(self.adjTable[centralNode],int(primelen*fratio))+random.sample(self.adjTable[centralNode],int(primelen*fover)):
                    #positive updating from nodes within the distance
                    self.singleUpdate(centralNode, friend, localalpha, 0, 1)
                    
                    #sampling enemies and negative updating
                    seeds = np.random.randint(low=np.iinfo(np.int64).max,size=self.numNegSampling,dtype=np.int64)
                    seeds = seeds%self.maxNegLen
                    for seed in seeds:
                        enemy = self.negSampTable[seed]
                        self.singleUpdate(centralNode,enemy,localalpha,1, 1)
                        
                for friend in seclist:
                    #positive updating from nodes within the distance
                    self.singleUpdate(centralNode, friend, localalpha, 0, 1)
                    
                    #sampling enemies and negative updating
                    seeds = np.random.randint(low=np.iinfo(np.int64).max,size=self.numNegSampling,dtype=np.int64)
                    seeds = seeds%self.maxNegLen
                    for seed in seeds:
                        enemy = self.negSampTable[seed]
                        self.singleUpdate(centralNode,enemy,localalpha,1,1)
                        
                #resume the original neighbors       
                #self.adjTable[centralNode] = self.adjTable[centralNode][:primelen]       
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
        sigma = sc.special.expit(sigma)
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
            print("Training in progress, {}%".format(percentage))
        else:
            if status == Status.INSTANTIATED:
                print("object has just been instantiated, please set parameters before training\n")
            if status == Status.INITIALIZED:
                print("ready to train the model\n")
            if status == Status.FILESAVED:
                print("Created target file ./{}\n".format(self.outputPath))
                
    
    '''
    Update 18/12/2018
    Author: Han Zhang
    Change Updating sequence to poisson processes, where each agent has different updating rate. 
    These updates happen concurrently
    '''
    
    def generatePoisson(self):
        '''
        this method is used for simulate a more realistic updating situation
        a stronger assumption here is that we know the information of average degree
        '''
        print("Simulating update sequence...")
        #1. calculate updating rates of each node
        coefficient = self.numUpdates / (self.window * self.timeslot)
        seq = []
        #2.simulate poisson process in every timeslot for each node
        history = {key:np.random.poisson(coefficient*len(self.graph[key]),self.timeslot) for key in self.graph.keys()}
        #3.calculate final sequence
        for i in range(self.timeslot):
            slot = []
            for key,value in history.items():
                for _ in range(value[i]):
                    slot.append(key)
            random.shuffle(slot)
            seq.extend(slot)
        
        return seq
    
    def poissonTraining(self):
        '''
        a different training thread, using with generatePoisson
        '''
        seq = self.generatePoisson()
        
        pathlen = len(seq)
        process = 0
        sentinel = 0
        alpha = self.alpha
        
        for node in seq:
            if process>=sentinel:
                print("training in progress:{}".format(int(100*sentinel/pathlen)))
                alpha = self.alpha * (1-process/float(pathlen))
                if alpha < self.alpha * 0.0001:
                    alpha = self.alpha * 0.0001
                sentinel += pathlen /(float(self.numUpdates))
            
            primelen = len(self.graph[node])
            if self.window*2*self.ratio >primelen:
                repeat = int(self.window*2*self.ratio/primelen)
                s = int(self.window*2*self.ratio)%primelen
                flist =  random.sample(self.graph[node],s)
                for _ in range(repeat):
                    flist += self.graph[node]
            else:
                flist = random.sample(self.graph[node],int(self.window*2*self.ratio))
                
            seclist = [random.choice(self.graph[random.choice(self.graph[node])])
                        for _ in range(int(self.window*2*(1-self.ratio)))]
            
            for friend in flist+seclist:
                self.singleUpdate(node, friend, alpha, 0, 1)
                
                #negative sampling
                seeds = np.random.randint(low=np.iinfo(np.int64).max,size=self.numNegSampling,dtype=np.int64)
                seeds = seeds%self.maxNegLen
                for seed in seeds:
                    enemy = self.negSampTable[seed]
                    self.singleUpdate(node,enemy,alpha,1,1)
            
            process += 1
            
            
    '''
    updated: Dec 25th,2018, Author: Han Zhang
    add functions to output embeddings as numpy matrix
    '''
    def getEmbeddings(self):
        node_num = len(self.repMatrix)
        
        emb_matrix = np.zeros((node_num,self.representSize))
        for node in range(node_num):
            emb_matrix[node] = self.repMatrix[node]
        
        return emb_matrix
                    
    
    
    
    
    
    
    
    
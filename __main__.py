'''
Created on 2018/10/27

@author: Han Zhang

An example script to train graph embeddings using DNELA, save to file afterwards
'''
import graph
import distributed as ds
import argparse




def main():
    '''
    main method of the pre_experiment program
    parameters:
    
    '''
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',help="whether to use default values",action="store_true")
    parser.add_argument('-c',help="continued training path",action="store_true")
    parser.add_argument('--embpath',help="the embedding path for continued training")
    parser.add_argument('--path',help="path of the graph")
    parser.add_argument('--output',help="output path of the embeddings")
    parser.add_argument('--dimension',type=int,default=128, help="dimension of embeddings")
    parser.add_argument('--updates',type=int,help="number of updates")
    parser.add_argument('--alpha',type=float,default=0.025,help="initial learning rate")
    parser.add_argument('--negative',type=int,default=5,help="negative sampling number")
    parser.add_argument('--neglen',type=int,default=10**8,help="the maximum number used for negative sampling")
    parser.add_argument('--ratio',type=float,help="the ratio of numbers of 1st and 2nd degree nodes")
    parser.add_argument('--fmax',default=1, type=float,help=
                        "the maximum ratio of # of 1st deg nodes participating updates to the total # of 1st deg nodes, when ratio=0.5")
    #parser.add_argument('-p',help="whether to use poisson process",action="store_true")
    parser.add_argument('--window',type=int,help="the update window in poisson update mode")
    parser.add_argument('--timeslot',default=1000,type=int,help="the number of timeslots in order to simulate poisson process")
    parser.add_argument('--seed',default=1,type=int,help="the random seed of a Distributed instance")
    
    
    args = parser.parse_args()
    
    G = graph.load_matfile(file_=args.path)
    d = ds.Distributed(G)
    
    if args.d:
        print('using default settings')
        d.defaultArgs()
    else:
        d.setArgs(alpha=args.alpha, 
                  numUpdates=args.updates, 
                  numNegSampling=args.negative, 
                  maxNeglen=10**8, 
                  representSize=args.dimension, 
                  outputPath=args.output,
                  ratio=args.ratio,
                  fmax=args.fmax,
                  c=args.c,
                  cpath=args.embpath,
                  poisson=True,
                  window=args.window,
                  timeslot=args.timeslot,
                  seed=args.seed)
            
    d.process()
    d.save2File()
    
    

if __name__ == '__main__':
    main()
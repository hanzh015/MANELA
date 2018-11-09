'''
Created on 2018/10/27

@author: Han Zhang
'''
import graph
import distributed as ds





def main():
    '''
    main method of the pre_experiment program
    parameters:
    
    '''
    
    G = graph.load_matfile(file_="D:/onedrive/FederatedLearning/Datasets/blogcatalog.mat")
    
    d = ds.Distributed(G)
    d.defaultArgs()
    d.process()
    d.save2File()
    
    

if __name__ == '__main__':
    main()
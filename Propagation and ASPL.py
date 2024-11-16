import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
import tqdm
import copy
import pandas as pd
import statistics
import datetime

class Pandemic_Network():
    
    def __init__(self, network_type: str, nodes: int, pandemicprob: float, reduced_prob: float ,mitigation_proportion: float, sicknode = 0, SW_connections = 3, edge_randomness = 0.2, SF_k = 1, plots = False, aspl = False, TTR = 15):
        '''
        sicknodes - number of sick nodes to begin with
        '''

        if network_type == 'Ring': # default is a ring network
            self.g = nx.watts_strogatz_graph(n = nodes, k = SW_connections, p = 0, seed=None)
        elif network_type == 'Small World':
            self.g = nx.watts_strogatz_graph(n = nodes, k = SW_connections, p = edge_randomness, seed=None)
        elif network_type == 'Scale Free':
            self.g = nx.barabasi_albert_graph(n = nodes, m = SF_k, seed=None, initial_graph=None) # m = Number of edges to attach from a new node to existing nodes
        else:
            self.g = nx.erdos_renyi_graph(n = nodes, p = edge_randomness, seed=None, directed=False) # if network type not specified, then generate random graph (erdos renyi model)

        # self.g = nx.watts_strogatz_graph(n = nodes, k = SW_connections, p = SW_randomness, seed=None)
        self.pos = nx.circular_layout(self.g)
        self.n = nodes
        self.t = 0
        self.p = pandemicprob
        self.p_reduced = reduced_prob # reduced probability of infection due to mitigation
        self.mit_prop = mitigation_proportion # proporation of the population that follows the mitigation
        self.want_plots = plots
        self.want_aspl = aspl
        self.masked = []
        self.sicknodes = [random.randint(0, nodes-1) for i in range(sicknode)] # randomly generate n number of sicknnodes
        # print(self.sicknodes)
        # if not sicknode:
        #     self.sicknodes = set([])
        # else:
        #     self.sicknodes = set([sicknode])
        self.recovered = []
        self.TTR = TTR # days to recover, default is 15, customisable dependent on model

        # initialise TTR
        initial_TTR_vals = [0 if i not in self.sicknodes else TTR + 1 for i in range(nodes)]
        initial_TTR_dict = {i:initial_TTR_vals[i] for i in range(nodes)}
        nx.set_node_attributes(self.g, initial_TTR_dict, name = 'TTR')
    
    def plot(self):

        node_colors = ["green" if node in self.recovered else 'magenta' if node in self.masked else "firebrick" if node in self.sicknodes else "skyblue" for node in self.g.nodes()]
        nx.draw_networkx(self.g, pos = self.pos, with_labels=False, node_size=2000/self.n, node_color=node_colors, linewidths=0.5)

        e_no_mit = [(u, v) for (u, v, d) in self.g.edges(data=True) if d["weight"] == self.p]
        e_mit = [(u, v) for (u, v, d) in self.g.edges(data=True) if d["weight"] == self.p_reduced]
        e_sus = [(u, v) for (u, v, d) in self.g.edges(data=True) if d["weight"] == 0]
        nx.draw_networkx_edges(self.g, self.pos, edgelist=e_no_mit, width=2, edge_color='red') # un-masked
        nx.draw_networkx_edges(self.g, self.pos, edgelist=e_mit, width=2, edge_color='orange') # masked
        nx.draw_networkx_edges(self.g, self.pos, edgelist=e_sus, width=2, edge_color='lime') # susceptible

        plt.title("Small Worlds Graph: Nodes = "+ str(self.n)+ ", Time = " + str(self.t))
        plt.show()
        return
    
    def ASPL(self):
        ASPL_w = []
        for C in (self.g.subgraph(c).copy() for c in nx.connected_components(self.g)):
            ASPL_w.append(nx.average_shortest_path_length(C, weight= 'weight'))
        return statistics.mean(ASPL_w)

    #def sicknode
    def propagate(self, steps: int):
        #Plot initial network
        # self.plot() 
        timestamps = []
        infectious_count = []
        recovery_count = []
        uninfected_count = []
        cumulative_case_count = []
        aspl_count = []

        for time in tqdm.tqdm(range(steps)):
            #check sick nodes
            new_sick = 0
            timestamps.append(time)
            # print('Time = ', time)
            # print([self.g.nodes[i]['TTR'] for i in range(len(list(self.g.nodes)))])
            # print('Infectious:', self.sicknodes)
            # print('Recovered:', self.recovered)
            if time == 0:
                cumulative_case_count.append(len(self.sicknodes))

                # initialise edge weights
                for i in self.sicknodes:
                    if np.random.random() < self.mit_prop: # a proportion of initially sick nodes will adopt mitigation measures
                        self.masked.append(i)
                    else: 
                        pass
                edge_list = [(u, v) for (u, v, d) in self.g.edges(data = True)] # all edges
                for j in edge_list:
                    if (j[0] in self.masked or j[1] in self.masked)  :
                        self.g.add_edge(j[0], j[1], weight = 1/self.p_reduced) 
                    elif (j[0] in self.sicknodes or j[1] in self.sicknodes) and (j[0] not in self.masked or j[1] not in self.masked):
                        self.g.add_edge(j[0], j[1], weight = 1/self.p) 
                    else:
                        self.g.add_edge(j[0], j[1], weight = 0) 

                # self.plot()
                
            else: 
                pass
                    
            infectious_count.append(len(self.sicknodes))
            recovery_count.append(len(self.recovered))
            uninfected_count.append(self.n - len(self.recovered) - len(self.sicknodes))
            
            currentsick = copy.copy(self.sicknodes) #as new sick nodes may be created and we don't want to loop through the new ones
            #print("sicknodes:", currentsick)

            for i in range(len(list(self.g.nodes))):
                if self.g.nodes[i]['TTR'] - 1 == 0: # end of infectious period
                    self.sicknodes.remove(i)
                    neighbours = list(self.g.neighbors(i))
                    for n in neighbours:
                        if n in self.sicknodes:
                            pass
                        else:
                            self.g.add_edge(i, n, weight = 0)

                    if i in self.masked:
                        self.masked.remove(i)
                    else:
                        pass
                    self.recovered.append(i)
                    nx.set_node_attributes(self.g, {i: self.g.nodes[i]['TTR'] - 1}, name='TTR') # decrement TTR value
                elif self.g.nodes[i]['TTR'] == 0: # not infectious, no action
                    pass
                else: # mid infectious period
                    nx.set_node_attributes(self.g, {i: self.g.nodes[i]['TTR'] - 1}, name='TTR') # decrement TTR value
            
            for node in currentsick:
                neighbours = list(self.g.neighbors(node))

                #try to propagate sickness
                for neighbour in neighbours:
                    if neighbour in self.sicknodes or neighbour in self.recovered: # do not infect again if already sick or recovered.
                        pass
                    elif self.g.get_edge_data(node,neighbour)['weight'] != 0 and np.random.random() < (1/self.g.get_edge_data(node,neighbour)['weight']): # new node infected
                        self.sicknodes.append(neighbour)
                        adj_neighbours = list(self.g.neighbors(neighbour)) # access the neighbours of the newly infected node to create new edge weightings.
                        new_sick += 1

                        nx.set_node_attributes(self.g, {neighbour:self.TTR + 1}, name = 'TTR') # initialise infectious node

                        if np.random.random() < self.mit_prop: # determine whether newly sick node will adopt mitigation measure
                            self.masked.append(neighbour)
                            for i in adj_neighbours:
                                self.g.add_edge(i, neighbour, weight = 1/self.p_reduced) # mitigated probability of infection -> overwrite all edges

                        else: # node did not adopt mitigation measure 
                            for i in adj_neighbours: 
                                if self.g.get_edge_data(i,neighbour)['weight'] == 1/self.p_reduced: # do not overwrite mitigated edges
                                    pass
                                else: 
                                    self.g.add_edge(i, neighbour, weight = 1/self.p) # non-mitigated probability of infection.

                    else: # node not infected
                        pass
            
            cumulative_case_count.append(cumulative_case_count[-1] + new_sick)
            self.t += 1 #timestep increase by 1
            
            if self.want_plots: 
                self.plot()
            
            if self.want_aspl:
                aspl_count.append(self.ASPL())
        
        if self.want_aspl:
            return timestamps, infectious_count, recovery_count, uninfected_count, cumulative_case_count[:-1], aspl_count
        else:    
            # self.plot()             
            return timestamps, infectious_count, recovery_count, uninfected_count, cumulative_case_count[:-1]

m_values = [i/10 for i in range(1,10)]
connections = [2, 3, 4]

results = []
for m in m_values:
    c_list = []
    for c in connections:
        G = Pandemic_Network(nodes = 500, network_type='Small World', pandemicprob = 1, sicknode = 1, SW_connections = c, edge_randomness = 0.2, 
                             reduced_prob = 0.01, mitigation_proportion = m, plots = False, aspl = True)
        data = G.propagate(300)
        c_list.append([data[1], data[-2], data[-1]]) # infectious, cumulative, ASPL
    results.append(c_list)

df = pd.DataFrame(results)
df.columns = connections

date = datetime.date.today()
df.to_csv('ASPL and Propagation Time - {}.csv'. format(date))
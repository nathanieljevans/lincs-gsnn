import numpy as np
from collections import deque
import networkx as nx


def bfs_distance(G, start_node, depth, node_names):
    """
    Perform BFS from a start node and return a dictionary where each key is a node
    and the value is the shortest path length from the start node, up to a given maximum depth.
    
    :param G: A NetworkX DiGraph (directed graph)
    :param start_node: The starting node for BFS
    :param max_depth: The maximum depth to explore from the start node
    :return: 
    """
    distances = {start_node: 0}
    queue = deque([(start_node, 0)])  # Store tuples (node, depth)

    while queue:
        current_node, current_depth = queue.popleft()

        # If we have reached the maximum depth, skip further exploration
        if current_depth > depth:
            continue

        for neighbor in G.successors(current_node):
            if neighbor not in distances:
                distances[neighbor] = current_depth + 1
                queue.append((neighbor, current_depth + 1))

    # convert to array for ease of selection later
    node2idx = {name:i for i,name in enumerate(node_names)}
    out = np.inf*np.ones((len(node_names),))
    for node, dist in distances.items(): 
        out[node2idx[node]] = dist
    return out

def get_all_possible_paths_set(G, rG, root, leaf, depth, root_distance_dict, leaf_distance_dict, node_names):
    """
    Get the union of nodes that appear in all paths from root to leaf 

    :param G: A NetworkX DiGraph (directed graph)
    :param root: The starting node (root node)
    :param leaf: The target node (leaf node)
    :return: The shortest path distance from root to leaf
    """

    # Step 1: Perform BFS from the root
    if root in root_distance_dict: 
        root_distances = root_distance_dict[root]
    else: 
        root_distances = bfs_distance(G, root, depth=depth, node_names=node_names)
        root_distance_dict[root] = root_distances

    # Step 2: Perform reverse BFS from the leaf
    if leaf in leaf_distance_dict: 
        leaf_distances = leaf_distance_dict[leaf]
    else: 
        leaf_distances = bfs_distance(rG, leaf, depth=depth, node_names=node_names)
        leaf_distance_dict[leaf] = leaf_distances

    spl = root_distances + leaf_distances 
    return spl, root_distance_dict, leaf_distance_dict 



def subset_graph(G, depth, roots, leafs, verbose=True, distance_dicts=None, return_dicts=False): 
    '''
    Given a graph G, we will subset the graph based on if a given node has at least one path (min path length=depth) from a root -> leaf 
    '''
    rG = G.reverse() # reverse all edges 
    node_names = sorted(list(G.nodes()))

    node_mask = np.zeros((len(node_names,)))
    if distance_dicts is not None: 
        root_distance_dict, leaf_distance_dict = distance_dicts
    else: 
        root_distance_dict = {}
        leaf_distance_dict = {}
    for i,root in enumerate(roots): 
        for j,leaf in enumerate(leafs): 
            if verbose: print(f'subgraph progress: {i+1}/{len(roots)} [{j+1}/{len(leafs)}]', end='\r')
            spl, root_distance_dict, leaf_distance_dict = get_all_possible_paths_set(G, 
                                                                                          rG,
                                                                                          root=root, 
                                                                                          leaf = leaf, 
                                                                                          depth=depth,
                                                                                          root_distance_dict = root_distance_dict,
                                                                                          leaf_distance_dict = leaf_distance_dict,
                                                                                          node_names = node_names)
            
            node_mask += 1.*(spl <= depth) 

    node_mask = node_mask > 0 # could threshold on a value here; interpretation: minimum number of paths from drugs->outputs that go through a given node
    nodes = set(np.array(node_names)[node_mask].tolist())
    subgraph = G.subgraph(nodes)

    if return_dicts:
        return subgraph, (root_distance_dict, leaf_distance_dict)
    else: 
        return subgraph
    


def build_nx(func_df, targets, outputs): 
    G = nx.DiGraph()

    # function -> function 
    for i,edge in func_df.iterrows(): 
        G.add_edge(edge.source, edge.target)

    # drug -> function
    for i,edge in targets.iterrows(): 
        if ('PROTEIN__' + edge.target) in G: 
            G.add_edge('DRUG__' + edge.pert_id, 'PROTEIN__' + edge.target)
        else: 
            print(f'warning: {edge.target} is not present in graph, this DTI will not be added.')

    # function -> output edges
    for out in outputs: 
        # add the edge even if the RNA doesn't exist; will get filtered in next step
        G.add_edge('RNA__' + out, 'LINCS__' + out)

    return G

def filter_func_nodes(func_names, func_df, targets, lincs, drugs, filter_depth=10): 
    G = build_nx(func_df, targets, lincs)
    
    subgraph = subset_graph(G, filter_depth, roots=['DRUG__' + x for x in drugs], leafs=['LINCS__' + x for x in lincs], verbose=True)
    nodes = list(subgraph.nodes())

    func_mask = np.array([n in nodes for n in func_names])
    drug_mask = np.array([('DRUG__' + n) in nodes for n in drugs])
    linc_mask = np.array([('LINCS__' + n) in nodes for n in lincs])

    print(f'\tfunction nodes retained: {(1.*func_mask).sum()}/{len(func_mask)}')
    print(f'\tdrug nodes retained: {(1.*drug_mask).sum()}/{len(drug_mask)}')
    print(f'\tlincs nodes retained: {(1.*linc_mask).sum()}/{len(linc_mask)}')

    func_names = np.array(func_names)[func_mask].tolist()
    drugs = np.array(drugs)[drug_mask].tolist() 
    lincs = np.array(lincs)[linc_mask].tolist()

    func_df = func_df[lambda x: x.source.isin(func_names) & (x.target.isin(func_names))]
    targets = targets[lambda x: x.target_name.isin(func_names)]

    return func_names, func_df, targets, drugs, lincs
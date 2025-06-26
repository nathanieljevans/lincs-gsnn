
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pypath.utils import mapping
import numpy as np
import imageio.v3 as iio      # pip install imageio  (v3 makes writing GIFs very easy)

def axes_to_gif(axes, gif_path="animation.gif", fps=5):
    """
    Convert a sequence of matplotlib Axes objects into a GIF.

    Parameters
    ----------
    axes : Sequence[matplotlib.axes.Axes]
        Your frames, one per Axes.
    gif_path : str
        Where to save the final GIF.
    fps : int
        Frames per second.

    """
    frames = []

    for i,ax in enumerate(axes):
        print(f'progress: {i+1}/{len(axes)} frames', end='\r')
        fig = ax.figure
        fig.canvas.draw_idle()         # ensures the figure is rendered
        w, h = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
        frames.append(img)

    # duration is in *seconds per frame*
    iio.imwrite(gif_path, frames, duration=1 / fps)
    print(f"✅  Saved {len(frames)}-frame GIF ➜  {gif_path}")


def annotate_edges(model, edge_weight): 

    res = pd.DataFrame({'src': np.array(model.homo_names)[model.edge_index[:, model.function_edge_mask][0].cpu()],
                     'dst': np.array(model.homo_names)[model.edge_index[:, model.function_edge_mask][1].cpu()],
                     'weight': edge_weight[model.function_edge_mask.cpu()]})

    res = res.assign(src_uniprot = [x.split('__')[1] for x in res.src])
    res = res.assign(dst_uniprot = [x.split('__')[1] for x in res.dst])

    res = res.assign(src_type = [x.split('__')[0] for x in res.src])
    res = res.assign(dst_type = [x.split('__')[0] for x in res.dst])

    res = res.sort_values('weight', ascending=False)

    return res 


def get_drug_edges(data, drug): 

    drug_idx = data.node_names_dict['input'].index('DRUG__' + drug)
    row,col = data.edge_index_dict['input','to','function'] 
    drug_edge_mask = row == drug_idx 
    drug_edges = data.edge_index_dict['input','to','function'][:, drug_edge_mask]

    return drug_edges



def make_subgraph(data, res, drug_edges): 

    G = nx.DiGraph()
    for i,row in res[lambda x: x.weight > 0.5].iterrows(): 
        G.add_edge(row.src, row.dst)

    ii = 0
    for i in range(drug_edges.shape[1]): 
        src,dst = drug_edges[:, i] 
        src_name = data.node_names_dict['input'][src]
        dst_name = data.node_names_dict['function'][dst]

        if dst_name in G: 
            ii+=1
            G.add_edge(src_name, dst_name)

    assert ii > 0, f'No drug targets found in the graph'

    # remove any nodes that is not an ancestor of the target node 
    #ancestors = nx.ancestors(G, target_node)
    #G = G.subgraph(list(ancestors) + [target_node]) 

    # add node type 
    for node in G.nodes():
        node_type, uniprot_id = node.split('__')
        nx.set_node_attributes(G, {node: node_type}, 'node_type')

        try: 
            gene_name = list(mapping.map_name(uniprot_id, 'uniprot', 'genesymbol'))[0]
        except: 
            gene_name = uniprot_id

        nx.set_node_attributes(G, {node: gene_name}, 'node_name')

    return G


def plot_graph(G, pos=None, ns=2500, show=True, xlim=None, ylim=None): 


    node_color = []
    for node in G.nodes(): 
        node_type = G.nodes[node]['node_type']
        if node_type == 'RNA':
            node_color.append('r')
        elif node_type == 'PROTEIN':
            node_color.append('g')
        elif node_type == 'DRUG':
            node_color.append('b')
        else: 
            node_color.append('gray')

    if pos is None: 
        H = nx.convert_node_labels_to_integers(G, label_attribute="node_label")
        H_layout = nx.nx_pydot.pydot_layout(H, prog="dot")
        pos = {H.nodes[n]["node_label"]: p for n, p in H_layout.items()}

    f,ax = plt.subplots(1,1, figsize=(40,15)) 
    nx.draw_networkx_nodes(G, pos, node_size=ns, alpha=1., node_shape='o', node_color=node_color)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=1., node_size=ns)
    nx.draw_networkx_labels(G, pos, font_size=12, font_color='black', font_family='sans-serif', labels={n: G.nodes[n]['node_name'] for n in G.nodes()})

    plt.legend(handles=[
        plt.Line2D([0], [0], marker='o', color='w', label='RNA', markerfacecolor='r', markersize=25),
        plt.Line2D([0], [0], marker='o', color='w', label='PROTEIN', markerfacecolor='g', markersize=25),
        plt.Line2D([0], [0], marker='o', color='w', label='DRUG', markerfacecolor='b', markersize=25)
    ], loc='upper left', fontsize=25)

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.axis('off')

    if show: plt.show() 
    plt.close() 
    
    return f, ax
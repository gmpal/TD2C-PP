import networkx as nx

def create_selective_undirected_graph(G, conditioning_set):
    """
    Create a graph where only conditioned colliders are made bidirectional.
    
    Logic: A collider only needs to be bidirectional if it's conditioned on
    (or its descendants are), because unconditioned colliders always block paths.
    """
    if conditioning_set is None:
        conditioning_set = set()
    
    # Start with a copy of the directed graph
    selective_graph = G.copy()
    
    # Get all nodes that are conditioned on or have conditioned descendants
    nodes_that_open_colliders = set()
    for node in conditioning_set:
        nodes_that_open_colliders.add(node)
        nodes_that_open_colliders.update(nx.descendants(G, node))
    
    # Find all nodes and check if they're colliders that need to be opened
    for node in G.nodes():
        predecessors = list(G.predecessors(node))
        
        # Check if this node is a collider AND is conditioned on (or has conditioned descendants)
        if len(predecessors) >= 2 and node in nodes_that_open_colliders:
            # This collider is conditioned on - make all incoming edges bidirectional
            for pred in predecessors:
                if not selective_graph.has_edge(node, pred):
                    selective_graph.add_edge(node, pred)
    
    return selective_graph


def count_open_paths(G, source, target, conditioning_set=None, limit=1000):
    """
    CORRECT implementation that counts open paths like dagitty::paths() in R.
    
    This enumerates all simple paths IN THE UNDIRECTED SKELETON and checks 
    each one individually for d-separation. D-separation rules apply to 
    undirected paths, not just directed paths!
    
    Parameters:
    -----------
    G : nx.DiGraph
        A directed acyclic graph (DAG)
    source : node
        Source node
    target : node
        Target node
    conditioning_set : set or None
        Set of nodes to condition on
    limit : int
        Maximum number of paths to enumerate (prevents exponential explosion)
        
    Returns:
    --------
    int
        Number of open (non-d-separated) paths
    """
    if conditioning_set is None:
        conditioning_set = set()
    
    # Convert to undirected skeleton to find all paths
    undirected_skeleton = G.to_undirected()
    
    # Get all simple paths in the undirected skeleton
    try:
        # all_paths = list(itertools.islice(
        #     nx.all_simple_paths(undirected_skeleton, source, target, cutoff=50), 
        #     limit
        # ))
        all_paths = list(nx.all_simple_paths(undirected_skeleton, source, target, cutoff=10))

    except nx.NetworkXNoPath:
        return 0
    
    # Count how many paths are open (not d-separated)
    open_count = 0
    for path in all_paths:
        if is_single_path_open(G, path, conditioning_set):
            open_count += 1
    
    return open_count


def is_single_path_open(G, path, conditioning_set):
    """
    Test if a specific path is open using NetworkX d-separation.
    
    Strategy: For a path to be blocked, there must exist some subset of 
    the conditioning set that d-separates the endpoints when restricted 
    to just the nodes on this path.
    """
    if len(path) < 2:
        return True
    
    if conditioning_set is None:
        conditioning_set = set()
    
    # Create a subgraph containing only the nodes on this path
    path_graph = nx.DiGraph()
    path_graph.add_nodes_from(path)
    
    # Add only the edges that form this specific path
    for i in range(len(path) - 1):
        current_node = path[i]
        next_node = path[i + 1]
        
        # Determine the direction of the edge in the original graph
        if G.has_edge(current_node, next_node):
            path_graph.add_edge(current_node, next_node)
        elif G.has_edge(next_node, current_node):
            path_graph.add_edge(next_node, current_node)
        # Note: for undirected paths in d-separation, we need to preserve
        # the original direction from the directed graph

    # Debug: Check what edges exist in the subgraph
    
    # Test d-separation on this specific path
    source, target = path[0], path[-1]
    
    # If the nodes are d-separated by the conditioning set in this 
    # path-specific subgraph, then this path is blocked
    try:
        is_separated = nx.is_d_separator(
            path_graph, 
            {source}, 
            {target}, 
            conditioning_set & set(path)  # Only conditioning nodes on this path matter
        )
        return not is_separated  # Path is open if NOT d-separated
    except nx.NetworkXError:
        # Handle edge cases (e.g., if subgraph is not a DAG)
        return True
    
def create_temporal_dag(rules, time_range):
    """Creates a time-unrolled DiGraph from causal rules."""
    G = nx.DiGraph()
    for t in time_range:
        for cause_base, effect_base, lag in rules:
            cause_node = f"{cause_base}_{t - lag}"
            effect_node = f"{effect_base}_{t}"
            G.add_edge(cause_node, effect_node)
    return G


def run_long_range_analysis(rules, lenght):
    """
    Performs the long-range asymmetry test for various lags 'k' and returns a DataFrame.
    """
    import pandas as pd 
    
    # Use a large time window to ensure long paths can be found
    time_window = range(-lenght, lenght)
    graph = create_temporal_dag(rules, time_window)
    t = 0  # Central time point
    
    results = []
    
    # Test for various lags k to show how the asymmetry grows
    for k in range(1, lenght):
        # Forward Test: I(Z_i(t-k); Z_j(t) | Z_j(t-k))
        f_start, f_end, f_given = f"Z_i_{t-k}", f"Z_j_{t}", {f"Z_j_{t-k}"}
        forward_paths_1 = count_open_paths(graph, f_start, f_end, f_given)

        # Backward Test: I(Z_j(t-k); Z_i(t) | Z_i(t-k))
        b_start, b_end, b_given = f"Z_j_{t-k}", f"Z_i_{t}", {f"Z_i_{t-k}"}
        backward_paths_1 = count_open_paths(graph, b_start, b_end, b_given)
        

        results.append({
            "Lag (k)": k,
            "FW1 - I(Z_i(t-k); Z_j(t) | Z_j(t-k))": forward_paths_1,
            "BW1 - I(Z_j(t-k); Z_i(t) | Z_i(t-k))": backward_paths_1,
        })
        
    return pd.DataFrame(results)
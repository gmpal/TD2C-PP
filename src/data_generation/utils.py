import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd


def from_dataframe_to_adjacency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts a DataFrame representing edges of a graph into an adjacency matrix.

    Parameters:
    df (pd.DataFrame): A DataFrame where each row represents an edge in the graph.
                       The DataFrame is expected to have three columns:
                       - The first column contains the source node as a string (e.g., 'n1').
                       - The second column contains the target node as a string (e.g., 'n2').
                       - The third column contains the weight of the edge as a numeric value.

    Returns:
    pd.DataFrame: An adjacency matrix where the rows and columns represent nodes,
                  and the values represent the weights of the edges between the nodes.
                  The nodes are sorted in ascending order based on their numeric identifiers.
    """
    nodes = set(df[0]) | set(df[1])  # Unique nodes
    nodes = [int(node[1:]) for node in nodes]  # Convert to int
    nodes = sorted(nodes)

    adj_matrix = pd.DataFrame(0, index=nodes, columns=nodes)

    # Populate the adjacency matrix
    for _, row in df.iterrows():
        node1, node2, weight = row
        adj_matrix.at[int(node1[1:]), int(node2[1:])] = weight
    return adj_matrix


def from_dataframe_to_nx(df: pd.DataFrame) -> nx.DiGraph:
    """
    Converts a pandas DataFrame to a NetworkX directed graph (DiGraph).

    The DataFrame is expected to have three columns where:
    - The first column contains the source nodes as strings with a prefix (e.g., 'n1', 'n2').
    - The second column contains the target nodes as strings with a prefix (e.g., 'n1', 'n2').
    - The third column contains the weights of the edges (only edges with weight 1 are added).

    Args:
        df (pd.DataFrame): The input DataFrame with edges information.

    Returns:
        nx.DiGraph: A directed graph constructed from the DataFrame.
    """
    nodes = set(df[0]) | set(df[1])  # Unique nodes
    nodes = [int(node[1:]) for node in nodes]  # Convert to int
    nodes = sorted(nodes)

    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)

    # Populate the adjacency matrix
    for _, row in df.iterrows():
        node1, node2, weight = row
        if weight == 1:
            graph.add_edge(int(node1[1:]), int(node2[1:]))
    return graph


def create_lagged_adjacency(M, max_lags=1):
    """
    Given an N x N adjacency matrix M for a DAG with variables x0, x1, ..., x(N-1),
    create a 2N x 2N adjacency matrix M_lagged that includes x_i(t) and x_i(t-1).

    - The first N rows/columns represent x0(t), x1(t), ..., x(N-1)(t).
    - The next N rows/columns represent x0(t-1), x1(t-1), ..., x(N-1)(t-1).
    """
    N = M.shape[0]
    M_lagged = np.zeros(((max_lags + 1) * N, (max_lags + 1) * N), dtype=int)

    # 1. Self-lag edges: x_i(t-1) -> x_i(t)
    for i in range(N):
        # row N+i = "x_i(t-1)" --> col i = "x_i(t)",
        for lag in range(1, max_lags + 1):
            # M_lagged[N+i, i] = 1
            M_lagged[N * lag + i, i] = 1

    # 2. Lagged edges from the original adjacency
    for i in range(N):
        for j in range(N):
            if M[i, j] == 1:
                M_lagged[i, j] = 1
                # If x_i -> x_j in the original DAG,
                # then x_i(t-1) -> x_j(t) in the lagged DAG
                for lag in range(1, max_lags + 1):
                    M_lagged[N * lag + i, j] = 1

    return M_lagged


def plot_time_lagged_graph_curved(G, n, rad=0.2):
    """
    Plot a time-lagged directed graph, placing nodes in horizontal layers
    based on their time step, and draw edges with a single curvature (arc).

    Parameters
    ----------
    G : nx.DiGraph
        A directed graph whose nodes are labeled (or numbered) in a way that
        every block of n nodes represents a time slice.
        - The first n nodes (0..n-1) are time=0 (present),
        - The next n nodes (n..2n-1) are time=1 (one step before), etc.
    n : int
        Number of variables in each time slice.
    rad : float
        Curvature radius for all edges. Positive = arcs bend one way,
        negative = arcs bend the other.

    Assumptions
    -----------
    - len(G) is divisible by n.
    - Node labels are integers from 0 to len(G)-1 that represent consecutive
      time slices of the same n variables.
    - The bottom row (y=0) is the present, the row above it (y=1) is one
      step before, and so forth.
    """

    total_nodes = len(G.nodes())
    if total_nodes % n != 0:
        raise ValueError(
            f"The number of nodes ({total_nodes}) is not divisible by n = {n}."
        )

    # Number of time steps
    T = total_nodes // n

    # Assign positions: x = variable index, y = time slice
    pos = {}
    for node in G.nodes():
        time_slice = node // n
        variable_idx = node % n
        pos[node] = (variable_idx, time_slice)

    plt.figure(figsize=(max(8, n), max(6, T)))

    # Draw nodes first
    nx.draw_networkx_nodes(
        G, pos, node_size=800, node_color="lightblue", edgecolors="black"
    )
    nx.draw_networkx_labels(G, pos)

    # Draw edges with curvature (arc)
    nx.draw_networkx_edges(
        G,
        pos,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=15,
        connectionstyle=f"arc3,rad={rad}",  # <--- Curvature here
        node_size=800,
    )

    # Adjust axis
    plt.xlim(-1, n)
    plt.ylim(-1, T)
    plt.title("Time-Lagged Graph (bottom = present)")
    plt.gca()  # Flip y-axis so y=0 is at the bottom
    plt.axis("off")
    plt.show()


def lag_time_series(data, max_lags=1):
    """
    Convert a time series data matrix of shape (T, N) into a lagged version
    of shape (T - max_lags, (max_lags + 1)*N).

    Parameters
    ----------
    data : numpy.ndarray
        A 2D array of shape (T, N), where T is the number of time points,
        and N is the number of variables/features at each time step.
    max_lags : int
        The number of time lags to include. Defaults to 1.

    Returns
    -------
    numpy.ndarray
        A 2D array of shape (T - max_lags, (max_lags + 1)*N). Each row
        contains the present values at time t followed by each lagged
        step: [ data[t], data[t-1], ..., data[t - max_lags] ].
    """
    T, N = data.shape
    if max_lags >= T:
        raise ValueError("max_lags must be less than the number of time steps T.")

    # The output has (T - max_lags) rows and (max_lags+1)*N columns
    output = np.zeros((T - max_lags, (max_lags + 1) * N), dtype=data.dtype)

    for t in range(max_lags, T):
        row_data = []
        # For each lag from 0 up to max_lags, collect data[t-lag]
        # If you want present first, then lag 1, etc.:
        for lag in range(max_lags + 1):
            row_data.append(data[t - lag, :])
        # row_data is now: [ data[t], data[t-1], ..., data[t - max_lags] ]
        # Concatenate them horizontally
        output[t - max_lags, :] = np.concatenate(row_data)

    return output


def get_causal_dfs(dag, n_variables, maxlags):
    """
    Start from a DAG and return a pandas dataframe with the causal relationships in the DAG.
    The dataframe has a MultiIndex with the source and target variables.
    The values are 1 if the edge is causal, 0 otherwise.
    """

    import pandas as pd

    pairs = [
        (source, effect)
        for source in range(n_variables, n_variables * maxlags + n_variables)
        for effect in range(n_variables)
    ]
    multi_index = pd.MultiIndex.from_tuples(pairs, names=["source", "target"])
    causal_dataframe = pd.DataFrame(index=multi_index, columns=["is_causal"])

    causal_dataframe["is_causal"] = 0

    # print(causal_dataframe)

    for parent_node, child_node in dag.edges:
        child_variable = int(child_node.split("_")[0])
        child_lag = int(child_node.split("-")[1])
        corresponding_value_child = child_lag * n_variables + child_variable
        if corresponding_value_child < n_variables:
            parent_variable = int(parent_node.split("_")[0])
            parent_lag = int(parent_node.split("-")[1])
            corresponding_value_parent = parent_lag * n_variables + parent_variable
            causal_dataframe.loc[
                (corresponding_value_parent, child_variable), "is_causal"
            ] = 1

    return causal_dataframe


def custom_layout(G, n_variables, t_lag):
    """
    Create a custom layout for the graph where nodes with the same identifier
    are aligned in the same column, regardless of their connections.
    """
    pos = {}
    width = 1.0 / (n_variables - 1)
    height = 1.0 / (t_lag - 1)

    for node in G.nodes():
        # if node is integer
        if isinstance(node, int):
            i, t = node % n_variables, node // n_variables
        elif "_t-" in node:
            i, t = map(int, node.split("_t-"))
        else:
            i, t = int(node), 0
        pos[node] = (i * width, t * height)

    # Scale and center the positions
    pos = {node: (x * 10, y * 3) for node, (x, y) in pos.items()}
    return pos


def show_DAG(G, n_variables, t_lag):
    """
    Plot the DAG using the custom layout.
    """

    import networkx as nx
    import matplotlib.pyplot as plt

    # Using the custom layout for plotting
    plt.figure(figsize=(10, 6))
    pos_custom = custom_layout(G, n_variables, t_lag)
    nx.draw(
        G,
        pos_custom,
        with_labels=True,
        node_size=1000,
        node_color="lightpink",
        font_size=10,
        arrowsize=10,
    )
    plt.title("Time Series DAG with Custom Layout")
    plt.show()

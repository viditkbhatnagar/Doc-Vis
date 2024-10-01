import plotly.graph_objects as go
import networkx as nx

def visualize_graph(G):
    pos = nx.spring_layout(G)

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, 
        y=edge_y,
        line=dict(width=2, color='grey'),
        hoverinfo='none',
        mode='lines'
    )

    node_x = []
    node_y = []
    node_info = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        relations = [G.edges[node, nbr]['relation'] for nbr in G.neighbors(node)]  # Collect all relations for the node
        node_info.append(f"{node}<br>Date: {G.nodes[node].get('year', 'N/A')}<br>Type: {G.nodes[node]['type']}<br>Relations: {'; '.join([G.edges[node, nbr]['relation'] for nbr in G.neighbors(node)])}")


    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        text=[node for node in G.nodes()],
        hoverinfo='text',
        hovertext=node_info,
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            size=10,
            color=[],
            line_width=2
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Network Graph with Dates, Types, and Relations',
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))
    fig.show()

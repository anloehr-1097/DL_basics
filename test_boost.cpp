#include "boost/graph/graph_concepts.hpp"
#include "boost_lib/boost/graph/adjacency_list.hpp"
#include "boost_lib/boost/graph/graph_traits.hpp"
#include "boost_lib/boost/graph/dijkstra_shortest_paths.hpp"
#include <iostream>
#include "boost_lib/boost/graph/graphviz.hpp"

typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS> Graph;

int WriteGraphToGraphvizFile(const std::string& filename, const Graph& g) {
    std::ofstream file;
    file.open(filename);

    if (!file.is_open()) {
        std::cerr << "Failed to open file " << filename << std::endl;
        return 1;
    }
    boost::write_graphviz(file, g);

    return 0;
}

int main(){
    using namespace boost;


   // Make convenient labels for the vertices
    enum { A, B, C, D, E, N };
    const int num_vertices = 5;
    const char* name = "ABCDE";

    // writing out the edges in the graph
    typedef std::pair<int, int> Edge;
    Edge edge_array[] =
    { Edge(A,B), Edge(A,D), Edge(C,A), Edge(D,C),
      Edge(C,E), Edge(B,D), Edge(D,E) };
    const int num_edges = sizeof(edge_array)/sizeof(edge_array[0]);

    // declare a graph object
    Graph g(num_vertices);

    // add the edges to the graph object
    for (int i = 0; i < num_edges; ++i)
      add_edge(edge_array[i].first, edge_array[i].second, g);

    for (int i = 0; i < num_edges; ++i){
        std::cout << edge_array[i].first << " " << edge_array[i].second << std::endl;
    }
    //write_graphviz(std::cout, g);
    WriteGraphToGraphvizFile("graph.dot", g);
         std::cout << "Created dotfile" << std::endl;

    return 0;
}
// idea: parse custom graph data structure to this graph data structure in order to visualize
//

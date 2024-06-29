import ast
import networkx as nx
import matplotlib.pyplot as plt

def build_knowledge_graph(code):
    # Parse the code
    tree = ast.parse(code)

    # Create a directed graph
    G = nx.DiGraph()

    # Helper function to add nodes and edges
    def add_node_and_edge(node_from, node_to, edge_type):
        G.add_node(node_from)
        G.add_node(node_to)
        G.add_edge(node_from, node_to, type=edge_type)

    # Visit all nodes in the AST
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            add_node_and_edge("module", node.name, "contains")
            for arg in node.args.args:
                add_node_and_edge(node.name, arg.arg, "parameter")
        elif isinstance(node, ast.ClassDef):
            add_node_and_edge("module", node.name, "contains")
            for base in node.bases:
                if isinstance(base, ast.Name):
                    add_node_and_edge(node.name, base.id, "inherits")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                add_node_and_edge("module", node.func.id, "calls")

    return G

# Example usage
code = """
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        pass

class Dog(Animal):
    def speak(self):
        return "Woof!"

def create_animal(animal_type, name):
    if animal_type == "dog":
        return Dog(name)
    return Animal(name)

my_dog = create_animal("dog", "Buddy")
my_dog.speak()
"""

graph = build_knowledge_graph(code)

# Draw the graph
pos = nx.spring_layout(graph)
nx.draw(graph, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10, font_weight='bold')
edge_labels = nx.get_edge_attributes(graph, 'type')
nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

plt.title("Knowledge Graph of Python Code")
plt.axis('off')
plt.show()
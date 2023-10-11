import torch
from graphviz import Digraph
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.autograd import Variable

model = GPT2LMHeadModel.from_pretrained('trained_gpt_py')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

input_question = 'What is PEP8 and how is it related to Python?'
input_ids = tokenizer.encode(input_question, add_special_tokens=True, return_tensors='pt')

def make_dot(var, params=None):
    """Produces Graphviz representation of PyTorch autograd graph."""
    if params is not None:
        param_map = {id(p): name for name, p in params}
    else:
        param_map = {}

    node_attr = dict(style='filled', shape='box', align='left', fontsize='12', ranksep='0.1', height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))

    def size_to_str(size):
        return '(' + (', ').join(['%d' % v for v in size]) + ')'

    def build_graph(var):
        if var not in seen:
            if torch.is_tensor(var):
                # Note: using shape as label to simplify visualization
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        build_graph(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    build_graph(t)

    seen = set()
    build_graph(var.grad_fn)
    return dot

# Wrap the input tensor with Variable for autograd
input_var = Variable(input_ids)

# Forward pass to generate the autograd graph
output = model(input_var)

# Create the graph visualization
dot = make_dot(output, params=[(name, p) for name, p in model.named_parameters()])

# Save the graph visualization as a file (optional)
dot.format = 'png'
dot.render('gpt_model_visualization')

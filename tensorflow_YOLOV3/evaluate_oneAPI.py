import tensorflow as tf
from daal4py.oneapi import mkl_link, MKL_CONTEXT
from daal4py.oneapi.algorithms import neural_networks
from daal4py.gal import Graph

# Load the pre-trained IR model
# model = tf.saved_model.load('path/to/model')

model = "./weed_model.pb"

### Model loading and execution                     ###    
with tf.gfile.GFile(model, 'rb') as f:
    restored_graph_def = tf.GraphDef()
    restored_graph_def.ParseFromString(f.read())
        
with tf.Graph().as_default() as graph:
    tf.import_graph_def(restored_graph_def, input_map=None, return_elements=None, name="")

# Use oneAPI MAD to analyze the model
mkl_link()
MKL_CONTEXT(limit='threads', threads=4)

result = neural_networks.prediction_result()
model_analysis = neural_networks.analysis.Batch()
model_analysis.input.setModel(model)
model_analysis.compute()
res = model_analysis.getResult()

print(res.get_layerDescriptors())
print(res.get_weightsAndBiases())

# Use oneAPI MKL to optimize the model
mkl_link()
MKL_CONTEXT(limit='threads', threads=4)

inference = neural_networks.inference.Batch()
inference.setModel(model)
inference.compute()
result = inference.getResult()

print(result.getValues())

# Use GAL to analyze the model's graph
graph_def = model.graph.as_graph_def()
graph = Graph(graph_def)

print(graph.get_nodes())
print(graph.get_edges())
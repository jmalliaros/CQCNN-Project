Phys 490: CQCNN project 
======================

## Description
For the Phys 490 final project, the paper "Predicting quantum advantage by quantum walk with convolutional neural networks"
was chosen as the topic to be recreated. In this repo, random undirected graphs are generated in the form of adjacency matrices
where a quantum and classical random walk is performed on the graph determining the labels for the graph. A convolutional neural network
called CQCNN was implemented which takes inputs of adjacency matrices and is trained to classify if random graphs are faster
with a classical walk or quantum walk.

## Dependencies
- numpy
- pygame
- matplotlib
- networkx
- pandas
- sklearn
- tqdm
- pytorch

 run `pip install -r requirements.txt ` to install dependencies

## Data Generation 
Run code in `data_generation` directory

### Generating graph data
To generate graph data, in `data_generation` directory, run the `main.py` code:

`python main.py -n <number of vertices in graph> -size <size of dataset> -shots <number of shots for walk>`

The data generated is saved under `/data`  as `graphs_n.csv`

Data is saved on each row where the first n^2 values represent the matrix, and following values represent 
the start, end, classical count, quantum count

Ex. of a row of a 3x3 matrix: `0 1 1 1 0 0 1 0 0 2 3 4 5`

Note: for graphs that cannot reach the endpoint from the starting point or takes too long, the count will return `-1`

### Running main.py

Run `python main.py -d data/graphs_10.csv` to run the main script that calls the model and forms the data

Sample: `>python main.py -n 10 -size 1000 -shots 10`

### classical random walk on graphs

To simulate classical random walk on adjacency matrices, the matrices are normalized along the rows which turned
it into a Markov chain with uniform probability distribution which was sampled from to determine the next step.

### quantum walk implementation 

To simulate quantum walk, the code from https://github.com/ctamon/quantum-walk-on-graphs was utilized and modified to
return the number of steps it took to reach a end node from some starting node


## Neural Network
To run the Classical-Quantum-Convolutional-Neural-Network (CQCNN) run:
 
`main.py -d <path to csv data> -param <path to json param file> -v <verbosity>`

sample: `python main.py -d data/graphs_8.csv -param param/param_1.json`

For help, use:

`python main.py --h`
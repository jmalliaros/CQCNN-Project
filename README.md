Phys 490: CQCNN project 
======================

## Dependencies
  run `pip install -r requirements.txt ` to install dependencies

## Data Generation 
Run code in `data_generation` directory

### Generating graph data
change variables `n` and `size` for different nxn matrices and data set size in `main.py`

run `python main.py` to generate data saved under `/data`  as `graphs_n.csv`

Data is saved on each row as nxn values representing the matrix, start, end, classical count, quantum count
Ex. of a 3x3 matrix: `0 1 1 1 0 0 1 0 0 2 3 4 5`

Note: for graphs that cannot reach the endpoint from the starting point, the count will return `-1`

### Running main.py

Run `python main.py -d data/graphs_10.csv` to run the main script that calls the model and forms the data


### quantum-walk-on-graphs 
Visual Simulation of Quantum Walk on Graphs.

Comments:
- Implemented in python.
- Required libraries:
  1. numpy: for linear algebra (eigenvalues and eigenvectors).
  2. pygame: for graphical visualization (basic drawing routines).
  3. networkx, matplotlib.pyplot: for drawing graphs.
  
To run: on Linux, type the following on the command line:

  `python qwalk.py`
  
The output will be a simulation of a continuous-time quantum walk on a path on three vertices.

To try the program on another graph, change the line containing

  `A = pathGraph(3)`

## Neural Network
Run code in `neural_net` directory 

run CQCNN: `main.py -d <path to data>`

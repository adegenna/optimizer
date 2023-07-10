![unittests](https://github.com/adegenna/optimizer/actions/workflows/unit_tests.yml/badge.svg)

Library for optimization algorithms. Allows user to choose from a variety of standard 
methods, both local (e.g., gradient descent) and global (e.g., particle swarm), for 
user-defined cost functions of arbitrary dimensionality. Results are written to text file 
and can be easily parsed/plotted with e.g. Python. Consult `examples/` for demonstrations.


Copyright 2023 by Anthony M. DeGennaro (ISC License).

**Requirements**

`Eigen` and `Gtest`, which may be installed like so:

```sh
sudo apt-get install libgtest-dev libeigen3-dev
```

**Installation**

```sh
cd [/PATH/TO/optimizer]
mkdir build
cd build
cmake ../
make
```

**Example Driver**

There are several example drivers located in `examples/` that are built with installation. Here is how to run one such example:

```sh
cd [/PATH/TO/optimizer]/build
./ex_var2d_func1d
```

**Output**

Results are written to a text file (name specified by the user). For gradient descent and simulated annealing, the output structure is this:

```sh
( x , cost(x) ) : 

x_11 , x_12 , [...] , x_1d ; C_1
x_21 , x_22 , [...] , x_2d ; C_2
[...]
x_N1 , x_N2 , [...] , x_Nd ; C_d
```

where d is the parameter space dimension and N is the number of epochs.

For particle swarm, an ensemble of N particles is computed and tracked every epoch, so the output structure is grouped into epoch ``blocks'', i.e.:

```sh
( x , cost(x) ) : 

epoch : 

x_11 , x_12 , [...] , x_1d ; C_1
x_21 , x_22 , [...] , x_2d ; C_2
[...]
x_N1 , x_N2 , [...] , x_Nd ; C_d

epoch : 

x_11 , x_12 , [...] , x_1d ; C_1
x_21 , x_22 , [...] , x_2d ; C_2
[...]
x_N1 , x_N2 , [...] , x_Nd ; C_d

[...]
```

**Example visualizations**

![](https://github.com/adegenna/optimizer/blob/master/figs/optimizer.png)
*(a) Gradient descent, (b) gradient descent with momentum, (c) simulated annealing*

![](https://github.com/adegenna/optimizer/blob/master/figs/swarm.png)
*Swarm optimization over 4 epochs*

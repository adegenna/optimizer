![unittests](https://github.com/adegenna/optimizer/actions/workflows/unit_tests.yml/badge.svg)

Library for optimization algorithms.

Copyright 2023 by Anthony M. DeGennaro (ISC License).

**Requirements**

Eigen
Gtest
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

```sh
cd [/PATH/TO/optimizer]/build
./ex_var2d_func1d
```

![](https://github.com/adegenna/optimizer/blob/master/figs/optimizer.png)
*(a) Gradient descent, (b) gradient descent with momentum, (c) simulated annealing*

![](https://github.com/adegenna/optimizer/blob/master/figs/swarm.png)
*Swarm optimization over 4 epochs*

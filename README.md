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
cd [/PATH/TO/root_finding]
mkdir build
cd build
cmake ../
make
```

**Example Driver**

```sh
cd [/PATH/TO/root_finding]/build
./ex_var2d_func1d
```

![](https://github.com/adegenna/root_finding/blob/master/figs/optimizer.png)
*(a) Gradient descent, (b) gradient descent with momentum, (c) simulated annealing*

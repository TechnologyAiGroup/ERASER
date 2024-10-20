# The ERASER Project

This is the website for the code and example data in the paper, "Efficient Reliability Attacks on Strong PUFs using Improved Evolution Strategies and Rank Correlation".

Now it contains the attack method for XOR APUF, MPUF, cMPUF and iPUF in the folders named after the type of PUF. We will continue updating this repository, adding code, data, and hardware implementations for other PUF types.

To start the attack, please excute

```
matlab {PUF_type}_ATTACK.m
```

The attack program can automatically generate CRPs and start attacking, if you want to use the example dataset, uncomment the lines about reading datasets from the file.

The available options for different methods include improvements to the CMA-ES algorithm and the selection of correlation coefficient. These are represented in the attack file by the  `flaga`, `flags`, and `flagp` respectively. You are free to choose and use them as needed.

For more details, please check the specific MATLAB file.

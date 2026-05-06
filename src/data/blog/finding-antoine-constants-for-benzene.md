---
author: Ever Vino 
pubDatetime: 2022-09-05
modDatetime: 2026-05-01
title: Finding Antoine Constants for Benzene
slug: finding-antoine-constants-for-benzene
featured: true
draft: false
tags:
  - engineering
  - chemical engineering
  - numerical methods
description:
  Here we calculate the Antoine constants for Benzene from experimental data using the numpy library, and we also plot the resulting curve with matplotlib.
---

## Table of contents

## Antoine Constants for Benzene

The vapor pressure data for Benzene at different temperatures are given in the following table. Determine the corresponding Antoine constants.

$$ 

\begin{array}{c|c|c|c} T(°C) & P(mmHg) & T(°C) & P(mmHg)\\ \hline -56.67 & 381.14 & -51.11 & 502.67\\ -45.56 & 651.61& -40.0 & 837.78\\ -34.44 & 1049.81& -28.89 & 1313.56\\ -23.33 & 1623.85& -17.78 & 1975.51\\ -12.22 & 2378.89& -6.67 & 2870.18 \\ -1.11 & 3428.70 & 4.44 & 4033.76 \\ 10.00 & 4747.43 & 15.56 & 5537.67 \\ 21.11 & 6412.65 & 26.67 & 7384.89 \\ 32.22 & 8481.25 & 37.78 & 9670.69 \\ 43.33 & 11015.28 & 48.89 & 12411.58 \\
\end{array} 

$$

_Source: [cheegineering](http://cheengineering.blogspot.com/2007_10_01_archive.html)_

The Antoine Equation is the following equation:

$$ \log_{10} P = A-\frac{B}{T+C} $$

Rearranging the equation to perform a multilinear regression:

$$T \cdot \log_{10}P +C\cdot \log_{10}P =A\cdot T+A\cdot C-B $$

$$ \log_{10}P=A+\frac{A\cdot C-B}{T}-\frac{C\cdot \log_{10}P}{T}$$

Using auxiliary constants: $\space \space b_0=A$ ;    $\space \space b_1=A\cdot C-B$ ;    $\space \space b_2=-C$

and the auxiliary variables: $\space \space y=\log_{10}P$; $\space \space x_1=\frac{1}{T}$; $\space \space x_2=\frac{\log_{10}P}{T}$

We obtain the following equation, which we will use to perform the multilinear regression.

$$y=b_0+b_1 x_1+b_2 x_2$$

## Some Theory Behind Multilinear Regression

It can be shown that the constants of any least-squares multilinear regression can be obtained from the following matrix relation:

$$ 
\begin{bmatrix} N \\ X_1 \\ X_2\\ \vdots \\ X_k \end{bmatrix} \times
\begin{bmatrix} N^T & X_1^T & X_2^T & \cdots & X_k^T \end{bmatrix} \times
\begin{bmatrix} b_0 \\ b_1 \\ b_2 \\ \vdots \\ b_k \end{bmatrix} =
\begin{bmatrix} N \\ X_1 \\ X_2\\ \vdots \\ X_k \end{bmatrix} \times
\begin{bmatrix} Y^T \end{bmatrix}\space\space\space\space\space (A)
$$

Where:

- $k$: number of variables
- $n$: number of data points available for each variable

_The transpose notation for $X$ is written as $X^T$. Note that $X$ is a row vector and $X^T$ is a column vector, and the same applies to the other variables._

The vectors mentioned above are composed of the experimental data used to perform the regression, except for $N$.

$$ N=\overbrace{(1, 1, ... , 1)}^{n\hspace{0.5em}ones} $$

$$ X_1=(x_{11}, x_{12}, ... , x_{1n})$$

$$ X_2=(x_{21}, x_{22}, ... , x_{2n})$$

$$...$$

$$ X_k=(x_{k1}, x_{k2}, ... , x_{kn})$$

$$ Y=(y_1, y_2, ... , y_n)$$

Simplifying equation $(A)$:

$$ \begin{bmatrix} N^T & X_1^T & X_2^T & \cdots & X_k^T \end{bmatrix}\times \begin{bmatrix} b_0 \\ b_1 \\ b_2 \\ \vdots \\ b_k \end{bmatrix}=  \begin{bmatrix} Y^T \end{bmatrix}$$

$$ 
\begin{bmatrix} b_0 \\ b_1 \\ b_2 \\ \vdots \\ b_k \end{bmatrix} = \begin{bmatrix} N^T & X_1^T & X_2^T & \cdots & X_k^T \end{bmatrix}^{-1} 
\begin{bmatrix} Y^T \end{bmatrix}
\space\space\space\space\space(B)
$$


Expression $(B)$ is much easier to handle and implement in Python.

To determine the degree of correlation, we can use the following formula known as the coefficient of determination, which is widely used to verify the quality of fit in multilinear regressions.

$$R^2 =\frac{\displaystyle\sum_{i=1}^{n}(\hat{y}_i-\bar{y})}{\displaystyle\sum_{i=1}^{n}(y_i - \bar{y})}$$

Where:

- $y =\log_{10}P$
- $y_i$: corresponds to the logarithm of each pressure value in our data
- $\bar{y}$: average of the logarithms of the pressure data
- $\hat{y}_i$: value obtained by applying our regression equation using the values $x_{1i}$ and $x_{2i}$

## Programming the Solution with Python

*To replicate this exercise in Python, you can download the Benzene data file [here](https://gist.githubusercontent.com/EverVino/026e9e745ddd078b64d3b643b1068a9c/raw/86ae999a475390b759b53b3e1d6d90014e10b802/datos_benceno_P_T.csv). Remember to save it as `datos_antoine.csv`, which is the filename used in the code.*

The Python implementation is shown below:

```python
import numpy as np
import csv
import matplotlib.pyplot as plt

P, T = [], []

with open("datos_antoine.csv", "r") as f:
    reader = csv.DictReader(f)

    for line in reader:
        P.append(float(line["P"].replace(",", ".")))
        T.append(float(line["T"].replace(",", ".")))

presion = np.array(P)
temp = np.array(T)

y = np.log10(presion)
x1 = 1 / temp
x2 = y * x1

n = len(T)
N = np.ones(n)
M = np.array([N, x1, x2])
b = np.linalg.pinv(M.T) @ y

A = b[0]
C = -b[2]
B = A * C - b[1]

# Results 6.867533528904103 832.260320713445 250.84525032625817
print(A, B, C)

data_temp = np.linspace(-60, 80, 100)
data_pres = 10**(A - B / (data_temp + C))

font = {'family': 'serif',
        'color': 'xkcd:dark red',
        'weight': 'normal',
        'size': 12,
        }

plt.plot(data_temp, data_pres, "-", label="Regression",
         color='xkcd:dark gray', alpha=0.7, zorder=2)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.scatter(T, P, marker="x", color='red', label="Experimental data")
plt.xlabel("Temperature (°C)", labelpad=10, fontdict=font)
plt.ylabel("Pressure (mmHg)", labelpad=10, fontdict=font)
plt.yticks(rotation=45)
plt.title("Pressure vs temperature \n Bencene",
          fontdict=font, pad=20)
plt.legend(loc=2, fontsize=10)
plt.show()

```

The Antoine constants for Benzene are:

$A = 6.8675$

$B = 832.2603 , °C$

$C = 250.8452 , °C$

The resulting plot is shown below:

![plot result](@/assets/images/antoine_bencene.png)


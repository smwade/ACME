# ACME Volume 2: Labs
### Sean Wade

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**

- [Python Essentials](#python-essentials)
  - [Lab1 - The Standard Library](#lab1---the-standard-library)
  - [Lab2 - Object-Oriented Programming](#lab2---object-oriented-programming)
- [Data Structures and Graph Algorithms](#data-structures-and-graph-algorithms)
  - [Lab3 - Public Key Cryptography](#lab3---public-key-cryptography)
  - [Lab4 - Data Structures I: Lists](#lab4---data-structures-i-lists)
  - [Lab5 - Data Structures II: Trees](#lab5---data-structures-ii-trees)
  - [Lab6 - Nearest Neighbor Search](#lab6---nearest-neighbor-search)
  - [Lab7 - Breadth-First Search and the Kevin Bacon Problem](#lab7---breadth-first-search-and-the-kevin-bacon-problem)
- [Probabilistic Algorithms](#probabilistic-algorithms)
  - [Lab8 - Markov Chains](#lab8---markov-chains)
- [Fourier Analysis](#fourier-analysis)
  - [Lab9 - Discrete Fourier Transform](#lab9---discrete-fourier-transform)
  - [Lab10 - Filtering and Convolution](#lab10---filtering-and-convolution)
  - [Lab11 - Introduction to Wavelets](#lab11---introduction-to-wavelets)
  - [Lab12 - Gaussian Quadrature](#lab12---gaussian-quadrature)
  - [Lab13 - Optimization Packages I (scipy.optimize)](#lab13---optimization-packages-i-scipyoptimize)
  - [Lab14 - Optimization Packages II (CVXOPT)](#lab14---optimization-packages-ii-cvxopt)
  - [Lab15 - Line Search](#lab15---line-search)
  - [Lab16 - Simplex](#lab16---simplex)
  - [Lab17 - Compressed Sensing](#lab17---compressed-sensing)
  - [Lab18 - Conjugate Gradient](#lab18---conjugate-gradient)
  - [Lab19 - Trust Region](#lab19---trust-region)
  - [Lab20 - Interior Point I](#lab20---interior-point-i)
  - [Lab21 - Interior Point II](#lab21---interior-point-ii)
  - [Lab22 - Dynamic Optimization](#lab22---dynamic-optimization)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Python Essentials
### Lab1 - The Standard Library

Python is designed to make it easy to implement complex tasks with little code. To that end, any Python distribution includes several built-in functions for accomplishing common tasks. In addition, Python is designed to import and reuse code written by others. A Python file that can be imported is called a module. All Python distributions include a collection of modules for accomplishing a variety of common tasks, collectively called the Python Standard Library. In this lab we become familiar with the Standard Library and learn how to create, import, and use modules.

### Lab2 - Object-Oriented Programming

Python is a class-based language. A class is a blueprint for an object that binds together specified variables and routines. Creating and using custom classes is often a good way to clean and speed up code. In this lab we learn how to define and use Python classes. In subsequents labs, we will create customized classes often for use in algorithms

## Data Structures and Graph Algorithms
### Lab3 - Public Key Cryptography

Implement the RSA cryptosystem as an example of public key cryptography and learn to use Python’s RSA implementation.

### Lab4 - Data Structures I: Lists

Implement linked lists as an introduction to data structures.

### Lab5 - Data Structures II: Trees

Implement tree data structures and understand their relative strengths and weaknesses.

### Lab6 - Nearest Neighbor Search

Nearest neighbor search is an optimization problem that arises in applications such as computer vision, pattern recognition, internet marketing, and data compression. In this lab we implement a K-D tree to solve the problem efficiently, then learn use scipy’s K-D tree in sklearn to implement a handwriting recognition algorithm.

### Lab7 - Breadth-First Search and the Kevin Bacon Problem

Graph theory has many practical applications. A graph may represent a complex system or network, and analyzing the graph often reveals critical information about the network. In this lab we learn to store graphs as adjacency dictionaries, implement a breadth-first search to identify the shortest path between two nodes, then use the NetworkX package to solve the so-called “Kevin Bacon problem.”

## Probabilistic Algorithms
### Lab8 - Markov Chains

A Markov chain is a finite collection of states with specified probabilities for transitioning from one state to another. They are characterized by the fact that future behavior of the system depends only on its current state. Markov chains have far ranging applications; in this lab, we create a Markov chain for generating random English sentences.

## Fourier Analysis
### Lab9 - Discrete Fourier Transform

The analysis of periodic functions has many applications in pure and applied mathematics, especially in settings dealing with sound waves. The Fourier transform provides a way to analyze such periodic functions. In this lab, we implement the discrete Fourier transform and explore digital audio signals.

### Lab10 - Filtering and Convolution

The Fourier transform reveals things about an audio signal that are not immediately apparent from the soundwave. In this lab we learn to filter noise out of a signal using the discrete Fourier transform, and explore the effect of convolution on sound files.

### Lab11 - Introduction to Wavelets

In the context of Fourier analysis, one seeks to represent a function as a sum of sinusoids. A drawback to this approach is that the Fourier transform only captures global frequency information, and local information is lost; we can know which frequencies are the most prevalent, but not when or where they occur. The Wavelet transform provides an alternative approach that avoids this shortcoming and is often a superior analysis technique for many types of signals and images.

### Lab12 - Gaussian Quadrature

Numerical quadrature is an important numerical integration technique. The popular Newton-Cotes quadrature uses uniformly spaced points to approximate the integral, but Gibbs phenomenon prevents Newton-Cotes from being effective for many functions. The Gaussian Quadrature method uses carefully chosen points and weights to mitigate this problem.

### Lab13 - Optimization Packages I (scipy.optimize)

Introduce some of the basic optimization functions available in scipy.optimize.

### Lab14 - Optimization Packages II (CVXOPT)

Introduce some of the basic optimization functions available in the CVXOPT package.

### Lab15 - Line Search

Investigate various Line-Search algorithms for numerical optimization.

### Lab16 - Simplex

Implement the Simplex Algorithm to solve linear constrained optimization problems.

### Lab17 - Compressed Sensing

Learn About Techniques in Compressed Sensing.

### Lab18 - Conjugate Gradient

Learn about the Conjugate-Gradient Algorithm and its uses.

### Lab19 - Trust Region

Explore Trust-Region methods for optimization.

### Lab20 - Interior Point I

For decades after its invention, the Simplex algorithm was the only competitive method for linear programming. The past 30 years, however, have seen the discovery and widespread adoption of a new family of algorithms that rival– and in some cases outperform–the Simplex algorithm, collectively called Interior Point methods. One of the major shortcomings of the Simplex algorithm is that the number of steps required to solve the problem can grow exponentially with the size of the linear system. Thus, for certain large linear programs, the Simplex algorithm is simply not viable. Interior Point methods offer an alternative approach and enjoy much better theoretical convergence properties. In this lab we implement an Interior Point method for linear programs, and in the next lab we will turn to the problem of solving quadratic programs.

### Lab21 - Interior Point II

Interior point methods originated as an alternative to the Simplex method for solving linear optimization problems. However, they can also be adapted to treat convex optimization problems in general. In this lab, we implement a primal-dual Interior Point method for convex quadratic constrained optimization and explore applications in elastic membrane theory and finance.

### Lab22 - Dynamic Optimization

This section teaches the fundamentals of Dynamic Programming using value function iteration.

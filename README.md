# Overview

NeVIS is a FPGA backend for running and deploying spiking neural networks.

# Summary

This repository contains the Python code for the compiler and prototype GUI interface. In its current iteration, it is designed to compile models made by the Nengo neural simulator. 

This repository also contains example code for running a simple spiking neural network, which is built using example code from Nengo's documentation (https://www.nengo.ai/nengo/examples/advanced/nef-algorithm.html).

# Project Development

The development of NeVIS will encompass adding several stages of functionality:

1. Integrate the current prototype architecture in with NengoGUI, with a constraint of 1 dimension to the inputs and outputs of the model.
2. Add support for higher input/output dimensions (2 and greater).
3. Add support for multiple ensembles to be autonomously compiled and deployed on a target FPGA platform.
4. Add support for learning rules (e.g. PES)
5. Add extra frameworks for system I/O besides UART in the hardware e.g. servo controllers and PWM outputs, along with an automated build tool (either in a GUI or command-line)
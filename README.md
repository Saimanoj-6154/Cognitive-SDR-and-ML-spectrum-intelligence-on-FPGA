# FPGA-Based Cognitive Radio for Real-Time Dynamic Spectrum Allocation

An FPGA-assisted Cognitive Software Defined Radio (SDR) framework for real-time dynamic spectrum allocation using spectrum sensing, FFT-based signal analysis, and lightweight neural network-assisted channel quality prediction.

---

## Overview

Traditional wireless systems use fixed spectrum allocation, leading to inefficient utilization of available RF spectrum. This project proposes a Cognitive Radio framework capable of sensing spectrum occupancy, analyzing channel conditions, and dynamically allocating communication channels in real time.

The system combines:

- FPGA-based signal processing
- FFT-based spectrum sensing
- RF feature extraction
- Lightweight ML-assisted channel quality prediction
- Dynamic spectrum allocation logic

The project is designed as a simulation and architecture validation framework without requiring physical FPGA or SDR hardware.

---

## Key Features

- Real-time RF spectrum sensing
- FFT-based spectral analysis
- Occupancy detection using energy sensing
- Lightweight ML-assisted channel quality prediction
- Dynamic spectrum allocation
- FPGA-oriented architecture design
- GNU Radio-based SDR simulation
- Vivado synthesis and FPGA resource analysis
- End-to-end simulation workflow

---

## System Pipeline

```text
RF Spectrum
    ↓
FFT Processing (FPGA)
    ↓
Spectrum Sensing
    ↓
Feature Extraction
    ↓
Lightweight Neural Network Classification
    ↓
Channel Quality Prediction
    ↓
Dynamic Spectrum Allocation
```

---

## Architecture Components

### FPGA Processing
Implemented using Verilog and simulated in Vivado.

Modules:
- FFT Processing
- Spectrum Sensing
- Threshold Detection
- Channel Occupancy Logic

### SDR Simulation
Implemented using GNU Radio.

Capabilities:
- RF signal generation
- Interference simulation
- Multi-channel spectrum emulation

### ML-Assisted Channel Analysis
Implemented using TensorFlow/Keras.

Purpose:
- Predict channel quality
- Assist adaptive spectrum allocation

---

## Technology Stack

| Category | Tool |
|---|---|
| FPGA Design & Synthesis | Vivado |
| HDL | Verilog |
| FPGA Simulation | Vivado Simulator |
| SDR Simulation | GNU Radio |
| RF Dataset Generation | Python + NumPy |
| Signal Analysis | SciPy |
| ML | TensorFlow/Keras |
| Visualization | Matplotlib |
| Version Control | Git + GitHub |

---

## Project Structure

```text
project/
│
├── fpga/
│   ├── fft/
│   ├── sensing/
│   ├── testbench/
│   └── synthesis_reports/
│
├── gnuradio/
│   ├── flowgraphs/
│   └── rf_simulation/
│
├── dataset/
│   ├── raw_iq_samples/
│   └── extracted_features/
│
├── ml/
│   ├── models/
│   ├── training/
│   └── inference/
│
├── simulation/
│   ├── fft_analysis/
│   ├── spectrum_sensing/
│   └── allocation_logic/
│
├── results/
│   ├── plots/
│   ├── metrics/
│   └── reports/
│
├── docs/
│
└── README.md
```

---

## Objectives

- Develop a cognitive radio architecture for efficient spectrum utilization
- Implement FFT-based spectrum sensing
- Perform RF feature extraction
- Design lightweight ML-assisted channel classification
- Simulate dynamic spectrum allocation
- Analyze FPGA resource utilization and timing

---

## Functional Workflow

### 1. RF Spectrum Generation
Synthetic RF environments are generated using GNU Radio.

### 2. FFT Processing
Frequency-domain analysis is performed using FFT modules.

### 3. Spectrum Sensing
Occupied and free channels are detected using energy-based sensing.

### 4. Feature Extraction
Features such as:
- Signal energy
- SNR
- Peak power
- Bandwidth occupancy

are extracted for classification.

### 5. Channel Quality Prediction
A lightweight neural network predicts channel quality.

### 6. Dynamic Spectrum Allocation
The system selects the optimal channel based on:
- interference level
- occupancy
- signal quality

---

## ML Model

A lightweight fully connected neural network is used.

### Architecture

```text
Input Layer
    ↓
Dense(16, ReLU)
    ↓
Dense(8, ReLU)
    ↓
Dense(4, Softmax)
```

### Output Classes

- Good
- Moderate
- Congested
- Noisy

---

## FPGA Design Goals

- Low-latency spectrum sensing
- Efficient FPGA resource utilization
- Parallel FFT processing
- Real-time adaptive allocation support

---

## Results and Metrics

The project evaluates:

- Spectrum sensing accuracy
- Channel prediction accuracy
- Allocation latency
- FFT performance
- FPGA resource utilization
- Timing analysis

---

## Future Improvements

- Hardware deployment on Zynq FPGA
- Real SDR integration using PlutoSDR/RTL-SDR
- Reinforcement learning-based allocation
- 5G/6G adaptive communication support
- FPGA-based neural inference acceleration

---

## Applications

- Cognitive Radio Networks
- Dynamic Spectrum Access
- Military Communication Systems
- Emergency Wireless Networks
- Smart RF Monitoring
- Adaptive IoT Communication

---

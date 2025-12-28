# Hopfield Neural Network Simulation

This repository contains a modular Python implementation of the **Hopfield Network**, a form of recurrent artificial neural network that serves as a content-addressable (associative) memory system. This project is developed for the **neuro-physics** organization to study attractor dynamics, energy minimization, and memory retrieval in neural lattices.

## 🧠 Theory Overview

The network consists of $N$ binary neurons $\sigma_i \in \{-1, 1\}$. Memory is stored in the synaptic weights $W_{ij}$ using the **Hebbian Learning Rule**:

$$W_{ij} = \frac{1}{N} \sum_{\mu=1}^{P} \xi_i^{\mu} \xi_j^{\mu}$$

where $\xi^{\mu}$ are the stored memory patterns. The network dynamics follow an energy minimization process defined by the Lyapunov (Energy) function:

$$E = -\frac{1}{2} \sum_{i,j} W_{ij} \sigma_i \sigma_j$$

## 🚀 Features

* **Multiple Update Rules**: Supports both Synchronous (parallel) and Sequential (asynchronous) updates.
* **Pattern Generation**: Preset patterns for 'H', 'X', '+', and striped lattices.
* **Energy Tracking**: Real-time calculation of the network's energy descent during retrieval.
* **Visualization Suite**: High-quality matplotlib plotting for lattice configurations and mathematical vector schematics.
* **Noise Robustness**: Tools to test memory retrieval against varying levels of stochastic noise.

## 🛠 Installation

1. Clone the repository:
```bash
git clone git@github.com:neuro-physics/hopfield-neural-network.git
cd hopfield-neural-network

```


2. Install dependencies:
```bash
pip install numpy matplotlib

```



## 📂 Core Functions

| Function | Description |
| --- | --- |
| `initialize_hopfield_model` | Computes the weight matrix  for a list of patterns. |
| `iterate_hopfield_synchronous` | Updates all neurons simultaneously; fast convergence. |
| `iterate_hopfield_sequential` | Updates neurons one-by-one; guaranteed energy descent. |
| `add_noise` | Corrupts a pattern by flipping a percentage of bits. |
| `plot_vector_lattice_schematic` | Visualizes the 10x10 lattice with  vector notation. |

## 📊 Simulation Examples

The current scripts simulate several complex behaviors of associative memory:

1. **Standard Retrieval**: Recovering 'H' and 'X' patterns from 30% noise.
2. **Anti-pattern Retrieval**: Demonstrating how the network can settle into $-\xi$ (the inverted memory).
3. **Spurious States (Hallucinations)**: Identifying local minima that do not correspond to stored memories (hallucinations), occurring when the network is initialized with hybrid patterns.

## 📈 Energy Minimization

The sequential update function tracks the energy after every single neuron flip. This results in a "staircase" descent toward the attractor, illustrating the physical principle of the model:

```python
s_final, energy_history = iterate_hopfield_sequential(W, s_init)
plt.plot(energy_history)
plt.title("System Energy over Neuron Updates")

```

## 📄 License

This project is licensed under the standard MIT License included in this repository.

---

**Developed by the Neuro-Physics Research Group.**
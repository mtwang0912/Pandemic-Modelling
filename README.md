# Pandemic-Modelling
BSc Project: A Network Science Approach to Pandemic Modelling

The full report can be found here: https://github.com/mtwang0912/Pandemic-Modelling/blob/main/PandemicModelling_02020640_DaveClements_TimEvans.pdf

The viva presentation can be found here: https://github.com/mtwang0912/Pandemic-Modelling/blob/main/BSc%20Project%20Viva.mp4 


# Overview of Achievements
This project aimed to simulate the spread of a disease within finite populations using small-world network models, employing the Susceptible-Infected-Recovered (SIR) epidemiological model. The study successfully:

- Investigated the phase transition phenomenon related to disease spread in network structures.
- Identified a critical point for disease containment or outbreak, determined by the average shortest path length (ASPL) of the network. 
- Explored how adopting mitigation measures influenced the network’s ASPL and the spread of the disease, showing that mitigation could shift the network state from an outbreak to a contained state by increasing ASPL.

# Key Features of the Code
- **Network Generation**: Utilized the NetworkX library to create small-world networks based on the Watts-Strogatz model, representing nodes (individuals) and edges (interactions).
- **Agent-Based Simulation**: Implemented an agent-based approach where nodes could transition between susceptible, infected, and recovered states based on infection probability and edge weights.
- **Temporal Analysis**: Captured the number of infected nodes and the overall propagation of the disease over time, with trials run multiple times to average out stochastic variations.
- **Mitigation Implementation**: Simulated the impact of different levels of mitigation adoption on infection probability and ASPL.

# Results
- Propagation Phases: Identified two distinct states within the network—contained (low infection spread) and outbreak (infection permeating the entire network).
  - Impact of Topology: ASPL dependencies on topology parameters (node count, average connections, probability of infection) followed power or exponential decay relationships.
- Mitigation Efficacy:
  - Demonstrated that high levels of mitigation (>40% adoption) significantly increased ASPL, effectively pushing the network to a contained state.
  - The relationship between mitigation and ASPL followed an exponential trend, indicating that more effective mitigation increased ASPL proportionally.
- Phase Transition: The probability of an outbreak sharply decreased as ASPL surpassed the critical threshold, indicating successful containment at higher ASPL values. 

# Hyperbolic Graph Neural Networks for Forensic Analytics

## Overview
This project explores the use of hyperbolic geometry for representing large scale financial transaction networks. Traditional Euclidean embeddings struggle with scale free graphs, while hyperbolic space provides exponential capacity for hierarchical structures.

## Key Idea
Graph nodes are embedded in hyperbolic space using the Lorentz model and visualized in the Poincare disk. The radial position reflects structural depth and connectivity patterns.

## Features
- Hyperbolic embedding pipeline
- Poincare disk visualization
- Analysis on Elliptic Bitcoin transaction dataset
- Structural interpretation of node hierarchy

## Output
- Nodes near center → highly connected hubs  
- Nodes near boundary → peripheral or anomalous entities  

## Limitations
Real world transaction graphs are not strictly hierarchical, so the resulting structure is not a perfect tree and may exhibit clustering and edge overlap.

## Run

```bash
pip install -r requirements.txt
python train.py

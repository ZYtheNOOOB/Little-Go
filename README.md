# Little-Go
CS561 project in USC. Implement a GO agent for 5x5 GO, no deep learning.

## Algorithm Applied

Feature extraction: Extract 1x1, 2x2 and 3x3 windows of location dependent and independent features on board.

Value estimation: Calculated by extracted features and their weights.

Learning algorithm: TD(0).

Search method: Monte-Carlo Tree Search.

## Efficiency

Incorporate weight sharing among features to accelarate learning.
Running efficiency improved by C++ in feature extraction and tree search.

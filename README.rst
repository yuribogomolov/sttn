=======
STTN
=======


STTN (spatio-temporal transactional network) is a generic data model to represent networks with spatial and temporal dimensions and a Python library that implements that standard.


Introduction
===========

Multiple real-world processes can be described by a list of transactions that include temporal and spatial dimensions, while transactions themself form a network between entities.
Examples include credit card transactions, phone calls, taxi trips, social network interactions, and the list goes on and on. Defining a generic data model to represent these datasets allows to:

1. Re-use the data retrieval, parser, and network construction logic.

2. Provide simplified API for network transformations (based on the knowledge of the network structure).

3. Implement advanced visualization, analysis, and modeling functionality (thanks to the consistent representation of spatial and temporal components).


Installation
===========
The latest library release and required dependencies can be installed from PyPI:

```
pip install sttn
```

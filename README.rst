=======
STTN
=======


STTN (spatio-temporal transactional network) is a generic data model to represent networks with spatial and temporal dimensions and a Python library that implements that standard.


Introduction
============

Multiple real-world processes can be described by a list of transactions that include temporal and spatial dimensions, while transactions themself form a network between entities.
Examples include credit card transactions, phone calls, taxi trips, social network interactions, and the list goes on and on. Defining a generic data model to represent these datasets allows to:

1. Re-use the data retrieval, parser, and network construction logic.

2. Provide simplified API for network transformations (based on the knowledge of the network structure).

3. Implement advanced visualization, analysis, and modeling functionality (thanks to the consistent representation of spatial and temporal components).


Installation
============
The latest library release and required dependencies can be installed from PyPI::

    pip install sttn

Getting started
===============

Import one of the included data providers, for example::

    from sttn.data.lehd import OriginDestinationEmploymentDataProvider

The latest list of included data providers can be found in the `data package <data_package_>`_. You can use available providers as an example to define your own parser. If the dataset is open we highly encourage you to open a Pull Request and contribute your provider to the community.

Now you can create an instance of the data provider and retrieve the data::

    lehd_provider = OriginDestinationEmploymentDataProvider()
    ny_lehd = lehd_provider.get_data(state='ny', year=2018)

Some data providers cache downloaded data on the local disk. The first run may take longer to download data from the Internet, while next runs will re-use the previously downloaded copy. The code above retrieves `LEHD Origin-Destination Employment Statistics <lodes>`_ for New York state based on 2018 census.
In addition to the origin-destination employment data the command above downloads shape files for census blocks and leverages both datasets to build the network.

Preview of node and edge attributes is helpful to understand the network structure::

    ny_lehd.nodes # to see network nodes
    ny_lehd.edges # to see network edges (or transactions)

.. _data_package: https://github.com/yuribogomolov/sttn/tree/main/src/sttn/data
.. _lodes: https://lehd.ces.census.gov/data/

A GUI for gamma spectroscopy using a PicoScope
==============================================

Introduction
------------

This is a data acquisition app for gamma spectroscopy. The Qt app requires a 5000-series PicoScope (e.g. the 5242D) for data acquisition.

Gamma spectroscopy of nuclear sources is commonly performed using either custom hardware (somewhat cheaper, easier to use, but less versatile) or modular hardware like NIM-crates (more expensive, harder to use, but more versatile). For the undergraduate physics lab at the Vrije Universiteit Amsterdam and the University of Amsterdam we require the versatility of modular hardware, but would like the ease-of-use of custom solutions. This way, students can focus on their research instead of working through a bunch of manuals.

Typically, a gamma detector (e.g. a NaI-scintillator/PMT or HPGe-detector) is connected to an amplifier which is connected to a multi-channel analyzer (MCA). Proprietary software must be used to read out the MCA. For more elaborate setups, a single-channel analyzer or pulse shaper is added to the mix. This can easily result in a time-consuming quest to find the cause of a very common situation: "it doesn't work."

We've decided to replace all NIM modules with a single PicoScope device. This is a digital oscilloscope which is connected over USB to a PC or laptop. Using the SDK it is easy to interface with the device and to write custom software for use by our students.

.. figure:: images/screenshot-spectrum.png
   :alt: screenshot showing a spectrum plot

   Screenshot of the GUI showing a plot of the gamma spectrum of a nuclear source (sodium-22, black line). Another detector at the opposite side of the source picks up coincident annihilation radiation gammas.


Installation
------------

In this section we will cover the installation of the gamma-spectroscopy package as well as the prerequisites (see below). On Windows, first make sure that the Anaconda distribution is installed and run all command-line commands from the Anaconda Prompt application.

Prerequisites
^^^^^^^^^^^^^

* PicoSDK
* Python wrappers for PicoSDK
* NumPy
* PyQt5
* PyQtGraph

To install the PicoSDK C libraries, follow the instructions at https://github.com/picotech/picosdk-python-wrappers. To install the wrapper itself, you can follow the instructions at that page or simply install from GitHub (you need to have Git installed):

.. code-block:: console

   $ pip install git+https://github.com/picotech/picosdk-python-wrappers

To install NumPy and the Qt5 packages you can use e.g. pip or conda, depending on your current Python setup. For example, using conda:

.. code-block:: console

   $ conda install numpy pyqt pyqtgraph

Gamma-spectroscopy
^^^^^^^^^^^^^^^^^^

You can install the latest release of this package directly from PyPI:

.. code-block:: console

   $ pip install gamma-spectroscopy

or, alternatively, you can install the latest development version from GitHub:

.. code-block:: console

   $ pip install git+https://github.com/davidfokkema/gamma-spectroscopy


Running the application
-----------------------

You can start the application GUI directly from the command-line (On Windows, first start the Anaconda Prompt application to get to the command-line):

.. code-block:: console

   $ gamma-spectroscopy

We will look into ways to include gamma-spectroscopy in Anaconda Navigator or as a stand-alone application so you don't first have to start a console.

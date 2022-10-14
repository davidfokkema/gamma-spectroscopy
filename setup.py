import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gamma-spectroscopy",
    version="1.1.1",
    author="David Fokkema",
    author_email="davidfokkema@icloud.com",
    description="A GUI for gamma spectroscopy using a PicoScope",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/davidfokkema/gamma-spectroscopy",
    packages=setuptools.find_packages(),
    package_data={
        "": ["*.ui"],
    },
    entry_points={
        "console_scripts": [
            "gamma-spectroscopy=gamma_spectroscopy.gamma_spectroscopy_gui:main",
        ],
    },
    install_requires=[
        "numpy",
        "pyqt5==5.15.2",
        "pyqtgraph",
        "picosdk @ git+https://github.com/picotech/picosdk-python-wrappers.git",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
    ],
)

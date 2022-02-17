from setuptools import setup, find_packages


PACKAGENAME = "SatGen"
VERSION = "0.1.0"


setup(
    name=PACKAGENAME,
    version=VERSION,
    author="Fangzhou Jiang",
    author_email="fzjiang@caltech.edu",
    description="A semi-analytical satellite galaxy and dark matter halo generator",
    long_description="A semi-analytical satellite galaxy and dark matter halo generator",
    install_requires=["numpy", "scipy", "lmfit", "fast_histogram"],
    packages=find_packages(),
    url="https://github.com/aphearin/SatGen",
)

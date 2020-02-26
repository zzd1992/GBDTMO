import setuptools

setuptools.setup(
    name="gbdtmo",
    version="0.0.1",
    author="Zhendong Zhang",
    author_email="zhd.zhang.ai@gmail.com",
    description="GBDT for multiple outputs",
    packages=['gbdtmo'],
    install_requires=['numpy', 'numba', 'graphviz'],
    classifiers=("Programming Language :: Python :: 3")
)

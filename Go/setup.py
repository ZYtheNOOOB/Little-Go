import setuptools
import numpy as np
from distutils.core import setup, Extension
from Cython.Build import cythonize

# setup(name='extract',
#       ext_modules=cythonize('extractor.pyx', language_level=3),
#       include_dirs=[np.get_include()])

setup(name='go',
      ext_modules=cythonize('go.pyx', language_level=3))

# setup(ext_modules=cythonize(Extension(
#     "go",  # the extension name
#     sources=["cgo.pyx", "go.cpp"],  # the Cython source and
#     # additional C++ source files
#     language="c++",  # generate and compile C++ code
# )))

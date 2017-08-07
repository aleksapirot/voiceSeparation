from distutils.core import setup
from Cython.Build import cythonize
import numpy

ext_options = {"compiler_directives": {"profile": True}, "annotate": True}
setup(
    ext_modules=cythonize(["diagfilt.pyx", "repet_speedup.pyx", "mfcc_speedup.pyx"], **ext_options),
    include_dirs=[numpy.get_include()]
)

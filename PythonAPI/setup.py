import setuptools
from simstar import __version__
setuptools.setup(
    name="simstar",
    version=__version__,
    author="Eatron Technologies",
    author_email="info@eatron.com",
    description="ADAS Simulator based on Unreal Engine",
   packages=setuptools.find_packages(),
    license='Proprietary',
    classifiers=(
        "Programming Language :: Python :: 3",
    ),
    install_requires=[
          'rpc-msgpack', 'numpy',
    ]
)
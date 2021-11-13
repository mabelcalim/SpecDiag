from setuptools import setup

setup(
    name='SpecDiag',
    version='0.1.0',
    description='A new tool for the comparison of time series in the frequency domain.',
    url='https://github.com/mabelcalim/Spectral-Diagram',
    author='Mabel Calim Costa',
    author_email='mabelcalim@gmail.com',
    license='BSD 3-clause',
    packages=['SpecDiag'],
    install_requires=['matplotlib',
                      'numpy',
                      ],
)

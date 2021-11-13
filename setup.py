from setuptools import setup

setup(
    name='Spectral-Diagram',
    version='0.1.0',
    description='A new tool for the comparison of time series in the frequency domain.',
    url='https://github.com/mabelcalim/Spectral-Diagram',
    author='Mabel Calim Costa',
    author_email='mabelcalim@gmail.com',
    license='BSD 3-clause',
    packages=['Spectral-Diagram'],
    install_requires=['matplotlib',
                      'numpy',
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD 3 License',
        'Operating System :: POSIX :: MacOSX',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)

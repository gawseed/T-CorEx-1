from setuptools import setup

with open("README.md") as f:
    long_description = f.read()

setup(name='gawseed-tcorex',
      version='1.1',
      description='Correlation Explanation Methods',
      long_description=long_description,
      long_description_content_type="text/markdown",
      author='Hrayr Harutyunyan',
      author_email='harhro@gmail.com',
      url='https://github.com/gawseed/T-CorEx',
      license='GNU Affero General Public License v3.0',
      install_requires=['numpy>=1.14.2',
                        'scipy>=1.1.0',
                        'torch>=0.4.1'],
      tests_require=['nose>=1.3.7',
                     'tqdm>=4.26'],
      entry_points={
          'console_scripts': [
              'tcorex = tcorex.tools.tcorex_cli:main',
              'tcorex-plot = tcorex.tools.tcorex_plot:main',
              'tcorex-changepoints = tcorex.tools.tcorex_changepoints:main',
          ]
      },
      classifiers=[
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Topic :: Scientific/Engineering :: Information Analysis',
          'Topic :: Scientific/Engineering :: Mathematics',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: GNU Affero General Public License v3',
      ],
      packages=['tcorex', 'tcorex.experiments', 'tcorex.tools'])

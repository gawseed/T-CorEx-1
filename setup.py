from setuptools import setup

long_description = '''
This package has implementations of two correlation explanation methods: linear CorEx and T-CorEx.

Linear CorEx searches for independent latent factors that explain all correlations between observed variables, while
also biasing the model selection towards modular latent factor models – directed latent factor graphical models where
each observed variable has a single latent variable as its only parent. The complete description of the method is presented
in NeurIPS 2019 [paper](https://arxiv.org/abs/1706.03353) *"Fast structure learning with modular regularization"*.

T-CorEx is designed for covariance estimation from temporal data. In its essence, T-CorEx trains a
[linear CorEx](https://arxiv.org/abs/1706.03353) for each time period, while employing two regularization techniques to
enforce temporal consistency of estimates.

Both methods have linear time and memory complexity with respect to the number of observed variables and can be applied to
truly high-dimensional datasets. It takes less than an hour on a moderate PC to run these methods on datasets with 100K variables.
Linear CorEx and T-CorEx are implemented in PyTorch and can run on both CPUs and GPUs.

This package also contains useful tools for working with high-dimensional low-rank plus diagonal matrices.
The code is compatible with Python 2.7-3.6 and is distributed under the GNU Affero General Public License v3.0.
'''

setup(name='T-CorEx',
      version='1.0',
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
              'tcorex = tcorex.tools.tcorex:main',
          ]
      },
      classifiers=[
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Topic :: Scientific/Engineering :: Information Analysis',
          'Topic :: Scientific/Engineering :: Mathematics',
          'Programming Language :: Python :: 3'
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 2'
          'Programming Language :: Python :: 2.7',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: GNU Affero General Public License v3'
      ],
      packages=['tcorex', 'tcorex.experiments', 'tcorex.tools'])

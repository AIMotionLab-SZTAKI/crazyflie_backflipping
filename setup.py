from setuptools import setup

setup(name='crazyflie_backflipping',
      version='1.0.0',
      install_requires=[
        'numpy',
        'mujoco',
        'glfw',
        'scipy',
        'gpytorch',
        'matplotlib',
        'bayesian-optimization',
        'scikit-optimize'
        ]
      )

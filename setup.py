# -*- coding: utf-8 -*-
"""
Setup file for hydesign
"""
import os
from setuptools import setup, find_packages

repo = os.path.dirname(__file__)
try:
    from git_utils import write_vers
    version = write_vers(vers_file='hydesign/__init__.py', repo=repo, skip_chars=1)
except Exception:
    version = '999'


try:
    from pypandoc import convert_file

    def read_md(f): return convert_file(f, 'rst', format='md')
except ImportError:
    print("warning: pypandoc module not found, could not convert Markdown to RST")

    def read_md(f): return open(f, 'r').read()


setup(name='hydesign',
      version=version,
      description='A tool for design and scontrol of utility scale wind-solar-storage based hybrid power plant.',
      long_description=read_md('README.md'),
      url='https://gitlab.windenergy.dtu.dk/TOPFARM/hydesign',
      project_urls={
          'Documentation': 'https://topfarm.pages.windenergy.dtu.dk/hydesign/',
          'Source': 'https://gitlab.windenergy.dtu.dk/TOPFARM/hydesign',
          'Tracker': 'https://gitlab.windenergy.dtu.dk/TOPFARM/hydesign/-/issues',
      },
      author='DTU Wind Energy',
      author_email='jumu@dtu.dk',
      license='MIT',
      packages=find_packages(),
      package_data={
          'hydesign': [
            'tests/test_files/sm.pkl',
            'tests/test_files/*.pickle',
            'tests/test_files/*.csv',
            'look_up_tables/*.nc',
            'examples/*.csv',
            'examples/*/*.csv',
            'examples/*/*/*.csv',
            'examples/*/*.yml',
            'examples/*.png',
            'examples/*/*.nc',
            ],},
      install_requires=[
          'dask',
          'numpy',
          'pandas',
          'scikit-learn',
          'scipy',
          'h5netcdf',
          'netcdf4',
          'xarray',
          'openmdao',
          'smt',
          'cplex',
          'docplex',
          'numpy-financial',
          'pvlib',
          'statsmodels',
          'rainflow',
          'pyyaml',
          'matplotlib',
          'zarr',
          'ortools',
          'NREL-PySAM',
          ],
      extras_require={
        'test': [
            'pytest',],
        'optional': [
            'seaborn',
            'jupyterlab',
            'finitediff',],
        'docs': [
            'pypandoc',
            'sphinx',
            'nbsphinx',
            'nbconvert',
            'sphinx_rtd_theme',],},
      include_package_data=True,
      zip_safe=True)

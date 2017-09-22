from setuptools import setup

setup(name='dk_tilted_model',
      version='0.1',
      description='reproduction of the Burton & Liszt (1978) Tilted Disk Model (Papers I-V)',
      url='www.astronomy.dk',
      author='Dhanesh Krishnarao (DK)',
      author_email='krishnarao@astro.wisc.edu',
      license='MIT',
      packages=['dk_tilted_model'],
      install_requires=[
          'numpy',
          'astropy',
          'spectral_cube',
          'aplpy',
          'seaborn',
          'matplotlib',
          'scipy',
          'numexpr'
      ],
      include_package_data=True,
      zip_safe=False)
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


# python setup.py sdist upload -r pypi
setup(
    name='ptfutils',
    version='0.0.9',
    packages=['tfutils', 'tfutils.data', 'tfutils.training'],
    url='https://github.com/PiotrDabkowski/',
    license='MIT',
    author='Piotr Dabkowski',
    author_email='piodrus@gmail.com',
    description='Useful modules for tensorflow',
)
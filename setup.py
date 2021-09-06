from setuptools import setup, find_namespace_packages

with open('README.md') as f:
    long_description = f.read()

setup(
    name='wembley',
    version=0.1,
    url='https://github.com/SamDuffield/vgpro',
    author='Sam Duffield',
    python_requires='>=3.6',
    install_requires=['jax',
                      'jaxlib',
                      'matplotlib',
                      'numpy'],
    packages=find_namespace_packages(),
    author_email='sddd2@cam.ac.uk',
    description='Variational Gaussian Processes with JAX',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='MIT'
)

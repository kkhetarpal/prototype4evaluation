from setuptools import setup
from prototype4evaluation import __version__
setup(
    name='prototype4evaluation',
    version=__version__,
    description='Code to demonstrate a minimal evaluation pipeline.',
    long_description=open('README.md', encoding='utf-8').read(),
    url='https://github.com/kkhetarpal/prototype4evaluation',
    author='Reasoning and Learning Lab, McGill University',
    author_email='name@email.com',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
    ],
    python_requires='>=3.5'
)

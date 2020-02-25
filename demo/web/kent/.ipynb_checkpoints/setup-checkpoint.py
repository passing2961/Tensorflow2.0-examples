from setuptools import setup, find_packages

setup(
    name='kent',
    version='1.0.0',
    author='Jonghwan Hyeon',
    author_email='jonghwanhyeon@kaist.ac.kr',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
    ],
    keywords='kent',
    packages=find_packages(),
    install_requires=[
        'pytest',
        'scipy',
        'numpy',
        'sklearn',
        'python-mecab-ko',
    ],
)


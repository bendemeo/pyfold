from setuptools import setup, find_packages

setup(
    name='pyfold',
    version='0.1.0',  # Start with a small version and increment it with new releases
    author='Your Name',
    author_email='your.email@example.com',
    description='A Python package for creating and manipulating origami crease patterns.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',  # This is important for READMEs in Markdown
    url='https://github.com/yourgithubusername/pyfold',  # Replace with your own GitHub URL
    packages=find_packages(),
    install_requires=[
        # List your project dependencies here.
        # For example, if you depend on numpy, add 'numpy'.
    ],
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.org/classifiers/
        'Development Status :: 3 - Alpha',  # Change as appropriate
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: MIT License',  # Change as appropriate
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.7',
    extras_require={
        'dev': [
            'pytest>=3.7',
        ],
    },
    include_package_data=True,
    package_data={
        # If any package contains *.txt or *.rst files, include them:
        '': ['*.txt', '*.md', '*.rst'],
    },
    # Entry points for creating executable scripts or command-line utilities
    entry_points={
        'console_scripts': [
            # Define console scripts here, like
            # 'pyfold-cli=pyfold.cli:main',
        ],
    },
)
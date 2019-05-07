import os
import sys

from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand


class PyTest(TestCommand):
    user_options = [("pytest-args=", "a", "Arguments to pass to pytest")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = ""

    def run_tests(self):
        import shlex

        # import here, cause outside the eggs aren't loaded
        import pytest

        errno = pytest.main(shlex.split(self.pytest_args))
        sys.exit(errno)


with open("README.rst", "r") as readme_file:
    readme = readme_file.read()

requirements = ["pytz>=2018", "numpy>=1.16.0", "matplotlib>=3.0.0"]

# upload to pypi:
#   python setup.py sdist bdist_wheel
#   python -m twine upload dist/*

setup(
    name="pydiagnostics",
    version="0.3.9",
    author="Timo Lesterhuis",
    author_email="timolesterhuis@gmail.com",
    description="A toolbox to analyse diagnostic data!",
    long_description=readme,
    long_description_content_type="text/x-rst",
    url="https://github.com/tim00w/diagnostics/",
    packages=find_packages(),
    install_requires=requirements,
    tests_require=["pytest", "pytest-cov", "pytest-mpl"],
    cmdclass={"pytest": PyTest},
    license="MIT License",
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    project_urls={"Documentation": "https://diagnostics.readthedocs.io/en/latest/",
                  "Source": "https://github.com/tim00w/diagnostics"},
)

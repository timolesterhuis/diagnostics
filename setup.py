import io
import sys
from glob import glob

from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext

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


def read(*names, **kwargs):
    with io.open(
        join(dirname(__file__), *names),
        encoding=kwargs.get('encoding', 'utf8')
    ) as fh:
        return fh.read()


requirements = ["pytz>=2018", "numpy>=1.16.0", "matplotlib>=3.0.0"]

# upload to pypi:
#   python setup.py sdist bdist_wheel
#   python -m twine upload dist/*

setup(
    name="pydiagnostics",
    version="0.6.0",
    author="Timo Lesterhuis",
    author_email="timolesterhuis@gmail.com",
    description="A toolbox to analyse diagnostic data!",
    long_description=read("README.rst"),
    long_description_content_type="text/x-rst",
    url="https://github.com/tim00w/diagnostics/",
    project_urls={
        'Documentation': 'https://diagnostics.readthedocs.io/',
        #'Changelog': 'https://python-nameless.readthedocs.io/en/latest/changelog.html',
        "Source": "https://github.com/tim00w/diagnostics",
        'Issue Tracker': 'https://github.com/timolesterhuis/diagnostics/issues',
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
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
)

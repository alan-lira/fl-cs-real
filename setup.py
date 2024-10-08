from pathlib import Path
from setuptools import find_packages, setup

from fl_cs_real.utils.setup_tools_util import get_readme, get_requirements_list, get_version

# Paths.
__BASE_PATH = Path(__file__).parent.resolve()
__VERSION_FILE = __BASE_PATH.joinpath("fl_cs_real/VERSION")
__README_FILE = __BASE_PATH.joinpath("README.md")
__REQUIREMENTS_FILE = __BASE_PATH.joinpath("requirements.txt")


setup(name="fl_cs_real",
      version=get_version(__VERSION_FILE),
      description="",
      long_description=get_readme(__README_FILE),
      long_description_content_type="text/markdown",
      url="https://github.com/alan-lira/fl_cs_real",
      author="Alan L. Nunes",
      author_email="",
      license="",
      platforms=["Operating System :: POSIX :: Linux"],
      classifiers=["Development Status :: 1 - Planning"],
      packages=find_packages(),
      include_package_data=True,
      install_requires=get_requirements_list(__REQUIREMENTS_FILE),
      python_requires=">=3.10",
      entry_points={"console_scripts": ["fl-cs-real=fl_cs_real.main:main"]})

from pathlib import Path


def get_version(version_file: Path) -> str:
    major = 0
    minor = 0
    patch = 0
    pre_release = None
    with open(file=version_file, encoding="utf-8") as version_file:
        for line in version_file:
            line = line.strip()
            if "major" in line:
                major = int(line.rsplit("=", 1)[1])
            elif "minor" in line:
                minor = int(line.rsplit("=", 1)[1])
            elif "patch" in line:
                patch = int(line.rsplit("=", 1)[1])
            if "pre_release" in line:
                # PEP 440, Pre-releases: Alpha release: aN | Beta release: bN | Release Candidate: rcN,
                # Where N stands for sequential number of pre-release.
                pre_release = str(line.rsplit("=", 1)[1])
    return str(major) + "." + str(minor) + "." + str(patch) + pre_release


def get_readme(readme_file: Path) -> str:
    with open(file=readme_file, encoding="utf-8") as readme_file:
        readme = readme_file.read()
    return readme


def get_requirements_list(requirements_file: Path) -> list:
    with open(file=requirements_file, encoding="utf-8") as requirements_file:
        requirements_list = requirements_file.read().splitlines()
    return requirements_list

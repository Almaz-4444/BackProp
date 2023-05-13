from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = []

setup(
    name="BackProp",
    version="0.0.6",
    author="Almaz4444",
    author_email="salmanowa2309@gmail.com",
    description="Package for working with BackProp neural networks",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/Almaz-4444/BackProp",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)
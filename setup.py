from setuptools import find_packages, setup


extras = {}
extras["quality"] = ["black ~= 22.0", "ruff>=0.0.241", "urllib3<=2.0.0"]
extras["docs_specific"] = ["hf-doc-builder"]
extras["dev"] = extras["quality"] + extras["docs_specific"]

setup(
    name="neurocache",
    version="0.0.1",
    description="Neurocache: A library for augmenting language models with external memory.",
    license_files=["LICENSE"],
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="deep learning",
    license="Apache",
    author="Ali Safaya",
    author_email="alisafaya@gmail.com",
    url="https://github.com/alisafaya/neurocache",
    package_dir={"": "src"},
    packages=find_packages("src"),
    entry_points={},
    python_requires=">=3.8.0",
    install_requires=[
        "peft>=0.5.0",
        "numpy>=1.17",
        "packaging>=20.0",
        "psutil",
        "pyyaml",
        "torch>=1.13.0",
        "transformers>=4.25.0",
        "tqdm",
        "accelerate>=0.21.0",
        "safetensors",
    ],
    extras_require=extras,
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

# Release checklist
# 1. Change the version in __init__.py and setup.py.
# 2. Commit these changes with the message: "Release: VERSION"
# 3. Add a tag in git to mark the release: "git tag VERSION -m 'Adds tag VERSION for pypi' "
#    Push the tag to git: git push --tags origin main
# 4. Run the following commands in the top-level directory:
#      python setup.py bdist_wheel
#      python setup.py sdist
# 5. Upload the package to the pypi test server first:
#      twine upload dist/* -r pypitest
#      twine upload dist/* -r pypitest --repository-url=https://test.pypi.org/legacy/
# 6. Check that you can install it in a virtualenv by running:
#      pip install -i https://testpypi.python.org/pypi peft
# 7. Upload the final version to actual pypi:
#      twine upload dist/* -r pypi
# 8. Add release notes to the tag in github once everything is looking hunky-dory.
# 9. Update the version in __init__.py, setup.py to the new version "-dev" and push to master

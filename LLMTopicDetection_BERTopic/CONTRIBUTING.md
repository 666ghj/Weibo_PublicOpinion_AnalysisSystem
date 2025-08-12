# Contributing to BERTopic

Hi! Thank you for considering contributing to BERTopic. With the modular nature of BERTopic, many new add-ons, backends, representation models, sub-models, and LLMs, can quickly be added to keep up with the incredibly fast-pacing field.

Whether contributions are new features, better documentation, bug fixes, or improvement on the repository itself, anything is appreciated!

## 📚 Guidelines

### 🤖 Contributing Code

To contribute to this project, we follow an `issue -> pull request` approach for main features and bug fixes. This means that any new feature, bug fix, or anything else that touches on code directly needs to start from an issue first. That way, the main discussion about what needs to be added/fixed can be done in the issue before creating a pull request. This makes sure that we are on the same page before you start coding your pull request. If you start working on an issue, please assign it to yourself but do so after there is an agreement with the maintainer, [@MaartenGr](https://github.com/MaartenGr).

When there is agreement on the assigned approach, a pull request can be created in which the fix/feature can be added. This follows a  ["fork and pull request"](https://docs.github.com/en/get-started/quickstart/contributing-to-projects) workflow.
Please do not try to push directly to this repo unless you are a maintainer.

There are exceptions to the `issue -> pull request` approach that are typically small changes that do not need agreements, such as:
* Documentation
* Spelling/grammar issues
* Docstrings
* etc.

There is a large focus on documentation in this repository, so please make sure to add extensive descriptions of features when creating the pull request.

Note that the main focus of pull requests and code should be:
* Easy readability
* Clear communication
* Sufficient documentation

## 🚀 Quick Start

To start contributing, make sure to first start from a fresh environment. Using an environment manager, such as `conda` or `pyenv` helps in making sure that your code is reproducible and tracks the versions you have in your environment.

If you are using conda, you can approach it as follows:

1. Create and activate a new conda environment (e.g., `conda create -n bertopic python=3.9`)
2. Install requirements (e.g., `pip install .[dev]`)
  * This makes sure to also install documentation and testing packages
3. (Optional) Run `make docs` to build your documentation
4. (Optional) Run `make test` to run the unit tests and `make coverage` to check the coverage of unit tests

❗Note: Unit testing the package can take quite some time since it needs to run several variants of the BERTopic pipeline.

## 🧹 Linting and Formatting

We use [Ruff](https://docs.astral.sh/ruff/) to ensure code is uniformly formatted and to avoid common mistakes and bad practices.

* To automatically re-format code, run `make format`
* To check for linting issues, run `make lint` - some issues may be automatically fixed, some will not be

When a pull request is made, the CI will automatically check for linting and formatting issues. However, it will not automatically apply any fixes, so it is easiest to run locally.

If you believe an error is incorrectly flagged, use a [`# noqa:` comment to suppress](https://docs.astral.sh/ruff/linter/#error-suppression), but this is discouraged unless strictly necessary.

## 🤓 Collaborative Efforts

When you run into any issue with the above or need help to start with a pull request, feel free to reach out in the issues! As with all repositories, this one has its particularities as a result of the maintainer's view. Each repository is quite different and so will their processes.

## 🏆 Recognition

If your contribution has made its way into a new release of BERTopic, you will be given credit in the changelog of the new release! Regardless of the size of the contribution, any help is greatly appreciated.

## 🎈 Release

BERTopic tries to mostly follow [semantic versioning](https://semver.org/) for its new releases. Even though BERTopic has been around for a few years now, it is still pre-1.0 software. With the rapid chances in the field and as a way to keep up, this versioning is on purpose. Backwards-compatibility is taken into account but integrating new features and thereby keeping up with the field takes priority. Especially since BERTopic focuses on modularity, flexibility is necessary.

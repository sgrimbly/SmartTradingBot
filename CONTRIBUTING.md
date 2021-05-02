# How to Contribute

I'm very happy to accept your patches and contributions to this project. If you would
like to contribute, please follow the following guidelines.

## Installing Pre-Commit Hooks and Testing Dependencies

Install the pre-commit hooks and testing dependencies:
```bash
pip install .[testing_formatting]
pre-commit install
```
You can run all the pre-commit hooks without making a commit as follows:
```bash
pre-commit run --all-files
```

## Naming Conventions
### Branch Names
We name our feature and bugfix branches as follows - `feature/[BRANCH-NAME]` or `bugfix/[BRANCH-NAME]`. Please ensure `[BRANCH-NAME]` is hyphen delimited.
### Commit Messages
We follow the conventional commits [standard](https://www.conventionalcommits.org/en/v1.0.0/).

## Code reviews

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests.

When making a Pull Request with a proposed change, please use this [format](.github/pull_request_template.md).

## Community Guidelines

This project follows
[Google's Open Source Community Guidelines](https://opensource.google.com/conduct/).

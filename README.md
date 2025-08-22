
<!-- TODO: create a gym-like environment called "cybergod-gym" which we can remote into other machines and act upon them -->

<!-- TODO: create human labeling environment for vimgolf-gym and cybergod-gym as web application -->

<!-- TODO: create a dedicated cybergod_vimgolf_gym docker image, separate from cybergod_worker_terminal and so on -->

# vimgolf-gym

OpenAI gym like environment and benchmark for Vimgolf.

## Installation

```bash
pip install vimgolf-gym
```

## Usage

```python
import vimgolf_gym

env = vimgolf_gym.make("vimgolf-v0")
env.act("")
img = env.screenshot() # output a PIL image
env.render() # preview screenshot
env.reset()

if env.solved:
   env.last_solution
```

## Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit helps, and credit will always be given.

You can contribute in many ways:

### Types of Contributions

#### Report Bugs

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

#### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help wanted" is open to whoever wants to implement it.

#### Implement Features

Look through the GitHub issues for features. Anything tagged with "enhancement" and "help wanted" is open to whoever wants to implement it.

#### Write Documentation

vimgolf-gym could always use more documentation, whether as part of the official vimgolf-gym docs, in docstrings, or even on the web in blog posts, articles, and such.

#### Submit Feedback

The best way to send feedback is to file an issue at https://github.com/james4ever0/vimgolf-gym/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions are welcome :)

### Get Started!

Ready to contribute? Here's how to set up vimgolf-gym for local development.

1. Fork the vimgolf-gym repo on GitHub.
2. Clone your fork locally:

   ```
   $ git clone git@github.com:your_name_here/vimgolf-gym.git
   ```

3. Create a branch for local development:

   ```
   $ git checkout -b name-of-your-bugfix-or-feature
   ```

4. Now you can make your changes locally.

5. When you're done making changes, check that your changes pass the tests:

   ```
   $ make test
   ```

6. Commit your changes and push your branch to GitHub:

   ```
   $ git add .
   $ git commit -m "Your detailed description of your changes."
   $ git push origin name-of-your-bugfix-or-feature
   ```

7. Submit a pull request through the GitHub website.

### Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put your new functionality into a function with a docstring, and add the feature to the list in README.md.
3. The pull request should work for Python 3.5, 3.6, 3.7 and 3.8. Check [Travis CI](https://travis-ci.org/your_name_here/vimgolf-gym/pull_requests) to make sure that the tests pass for all supported Python versions.
# BrainStorm
An agent that helps you to learn anything on your computer.

## Documentation

The `brainstorm_llm` docs under `src/brainstorm_llm/docs` are built with MkDocs
(Material theme) and published on Read the Docs.

```bash
uv run task docs        # serve at http://127.0.0.1:8000
uv run task docs-build  # strict build, same check as CI
```

To publish, import the repository at [readthedocs.com](https://readthedocs.com)
and point the project at the default branch. Read the Docs reads
`.readthedocs.yaml`, builds `mkdocs.yml`, and rebuilds on every push to `main`;
the `docs` GitHub Actions workflow validates the build on pushes and pull requests.

## Local Installation

#### 1. Clone the repo

```bash
git clone https://github.com/MahirHamiAbrar/BrainStorm.git
```

#### 2. Setup dependencies

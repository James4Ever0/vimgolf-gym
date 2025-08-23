uv pip install .
env PDOC_PROCESS=1 pdoc -o ./docs vimgolf_gym
cd docs
python3 -m http.server
# TODO: unify the documentation style
# pdoc supported styles: markdown,google,numpy,restructuredtext 
# commandline option: pdoc -d <style>
uv pip install .
env PDOC_PROCESS=1 pdoc -d google -o ./docs vimgolf_gym
cd docs
python3 -m http.server

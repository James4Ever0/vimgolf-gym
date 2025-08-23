# TODO: unify the documentation style
# pdoc supported styles: markdown,google,numpy,restructuredtext 
# commandline option: pdoc -d <style>

# TODO: annotate class attributes next to the definition
uv pip install .
env PDOC_PROCESS=1 pdoc -d google -o ./docs vimgolf_gym '!vimgolf_gym.*.*.model_config' '!vimgolf_gym.*.*.model_extra'
cd docs
python3 -m http.server

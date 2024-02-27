test:
	pytest tests/test_back.py
	pytest tests/test_masks.py
	pytest tests/test_tools.py
	pytest tests/test_basic_beam_size.py
	pytest tests/test_no_noise.py
	pytest tests/test_noise.py
	pytest tests/test_iso_noise.py
	pytest tests/test_gaussian.py

html:
	cd docs && python -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build
	open docs/_build/index.html

lint:
	-pylint laserbeamtools/__init__.py
	-pylint laserbeamtools/analysis.py
	-pylint laserbeamtools/background.py
	-pylint laserbeamtools/display.py
	-pylint laserbeamtools/gaussian.py
	-pylint laserbeamtools/masks.py
	-pylint laserbeamtools/image_tools.py
	-pylint laserbeamtools/m2_fit.py
	-pylint laserbeamtools/m2_display.py
	-pylint laserbeamtools/rayfile_gen.py
	-pylint tests/test_all_notebooks.py


doccheck:
	-pydocstyle laserbeamtools/__init__.py
	-pydocstyle laserbeamtools/analysis.py
	-pydocstyle laserbeamtools/background.py
	-pydocstyle laserbeamtools/display.py
	-pydocstyle laserbeamtools/gaussian.py
	-pydocstyle laserbeamtools/image_tools.py
	-pydocstyle laserbeamtools/masks.py
	-pydocstyle laserbeamtools/m2_fit.py
	-pydocstyle laserbeamtools/m2_display.py
	-pydocstyle laserbeamtools/rayfile_gen.py

rstcheck:
	-rstcheck README.rst
	-rstcheck CHANGELOG.rst
	-rstcheck docs/index.rst
	-rstcheck docs/changelog.rst
	-rstcheck --ignore-directives automodapi docs/analysis.rst
	-rstcheck --ignore-directives automodapi docs/background.rst
	-rstcheck --ignore-directives automodapi docs/display.rst
	-rstcheck --ignore-directives automodapi docs/image_tools.rst
	-rstcheck --ignore-directives automodapi docs/m2_display.rst
	-rstcheck --ignore-directives automodapi docs/m2_fit.rst
	-rstcheck --ignore-directives automodapi docs/masks.rst

rcheck:
	make clean
	make test
	make lint
	make doccheck
	make rstcheck
	touch docs/*ipynb
	touch docs/*rst
	make html
	check-manifest
	pyroma -d .
	pytest --verbose tests/test_all_notebooks.py


clean:
	rm -rf .eggs
	rm -rf .pytest_cache
	rm -rf .virtual_documents
	rm -rf __pycache__
	rm -rf dist
	rm -rf laserbeamtools.egg-info
	rm -rf laserbeamtools/__pycache__
	rm -rf docs/_build
	rm -rf docs/api
	rm -rf docs/.ipynb_checkpoints
	rm -rf tests/__pycache__
	rm -rf build


.PHONY: clean rcheck html notecheck pycheck doccheck test rstcheck
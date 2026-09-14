.PHONY: install demo train dashboard test

install:
	pip install -e .

demo:
	python scripts/generate_demo_data.py
	python scripts/train_model.py --data data/raw/demo_ibm_aml.csv

train:
	python scripts/train_model.py --data data/raw/HI-Small_Trans.csv

dashboard:
	streamlit run app/streamlit_app.py

test:
	pytest -q

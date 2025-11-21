.PHONY: install train eval clean

install:
	pip install -r requirements.txt

train:
	python3 src/train.py

eval:
	python3 src/evaluate.py

clean:
	rm -rf checkpoint/ logs/


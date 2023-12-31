# help
help:
	@echo " "
	@echo "Targets:"
	@echo " "
	@echo "- make black"
	@echo "- make flake8"
	@echo "- make generate_badge"
	@echo "- make check_all "

black:
	pre-commit run black
flake8:
	pre-commit run flake8
generate_badge:
	bash -c 'flake8 ./ --exit-zero --format=html --htmldir ./reports/flake8 --statistics --output-file ./reports/flake8/flake8stats.txt && genbadge flake8 -o ./reports/flake8/flake8-badge.svg'
check_all:
	pre-commit
	bash -c 'flake8 ./ --exit-zero --format=html --htmldir ./reports/flake8 --statistics --output-file ./reports/flake8/flake8stats.txt && genbadge flake8 -o ./reports/flake8/flake8-badge.svg'

.PHONY: help  Makefile

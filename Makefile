PYTHON ?= python3
TECTONIC ?= tectonic

.PHONY: help reproduce paper figures check all

help:
	@echo "make reproduce  recompute the main numbers from data/ (numpy only)"
	@echo "make paper      build paper/main.pdf (Tectonic)"
	@echo "make figures    redraw the figures that need only analyses/out (the others need the archived traces)"
	@echo "make check      page budget, provenance hygiene and a clean tree"

reproduce:
	$(PYTHON) reproduce/reproduce_tables.py data/episodes

paper:
	cd paper && $(TECTONIC) main.tex

figures:
	cd analyses && $(PYTHON) p272b_learning_curve_plot.py out/p272

check:
	$(PYTHON) tools/check_repo.py --pdf paper/main.pdf

all: reproduce paper check

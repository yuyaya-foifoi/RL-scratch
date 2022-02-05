SRC := src

style:
	isort $(SRC)
	black $(SRC)
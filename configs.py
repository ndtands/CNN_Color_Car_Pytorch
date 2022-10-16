from utils import read_json

IDX2TAG_NAME = 'IDX2TAG.json'
TAGS = ["black", "blue", "brown", "green", "orange", "pink", "purple", "red", "white", "yellow"]
IDX2TAG = {str(idx):tag for idx, tag in enumerate(TAGS)}
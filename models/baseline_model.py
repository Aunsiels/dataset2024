import csv

import requests
from loguru import logger

from models.abstract_model import AbstractModel
from bs4 import BeautifulSoup


class BaselineModel(AbstractModel):
    def __init__(self):
        super().__init__()

    def generate_predictions(self, inputs):
        raise NotImplementedError

    @staticmethod
    def read_prompt_templates_from_csv(file_path) -> dict:
        """Read prompt templates from a CSV file."""
        logger.info(
            f"Reading prompt templates from `{file_path}`..."
        )

        with open(file_path) as csvfile:
            reader = csv.DictReader(csvfile)
            prompt_templates = {
                row["Relation"]: row["PromptTemplate"] for row in reader
            }
        return prompt_templates

    @staticmethod
    def disambiguation_baseline(item) -> str:
        """A simple disambiguation function that returns the Wikidata ID of an item."""
        item = str(item).strip()
        res = disambiguation_autocomplete(item)
        if res == item:
            res = disambiguation_search(item)
        return res


def disambiguation_autocomplete(item) -> str:
    item = str(item).strip()

    if not item or item == "None":
        return ""

    try:
        # If item can be converted to an integer, return it directly
        return str(int(item))
    except ValueError:
        # If not, proceed with the Wikidata search
        try:
            url = (f"https://www.wikidata.org/w/api.php"
                   f"?action=wbsearchentities"
                   f"&search={item}"
                   f"&language=en"
                   f"&format=json")
            data = requests.get(url).json()
            # Return the first id (Could upgrade this in the future)
            return str(data["search"][0]["id"])
        except Exception as e:
            logger.error(f"Error getting Wikidata ID for `{item}`: {e}")
            return item


def disambiguation_search(item) -> str:
    item = str(item).strip()
    if not item or item == "None":
        return ""
    try:
        url = f"https://www.wikidata.org/w/index.php?go=Go&search={item}&title=Special%3ASearch&ns0=1&ns120=1"
        data = requests.get(url).content.decode("utf-8")
        soup = BeautifulSoup(data, "html.parser")
        res = []
        for item in soup.find_all("li", {"class": "mw-search-result mw-search-result-ns-0"}):
            title = item.find("span", {"class": "wb-itemlink-label"}).text
            description = item.find("span", {"class": "wb-itemlink-description"}).text
            id = item.find("span", {"class": "wb-itemlink-id"}).text.strip("(").strip(")")
            stats = item.find("div", {"class": "mw-search-result-data"}).text
            statements = stats.split(" ")[0]
            res.append((title, description, id, statements))
        res = sorted(res, key=lambda x: int(x[-1]), reverse=True)
        if res:
            return res[0][2]
    except Exception as e:
        logger.error(f"Error getting Wikidata ID for `{item}`: {e}")
        raise
        return item


if __name__ == '__main__':
    print(disambiguation_autocomplete("Ray Bradbury,"))
    print(disambiguation_search("Ray Bradbury,"))
    print(BaselineModel.disambiguation_baseline("Ray Bradbury,"))

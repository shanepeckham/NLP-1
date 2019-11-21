
import mendeleev
from mendeleev import element
from mendeleev import get_all_elements


import pandas as pd
from pprint import pprint
from collections import defaultdict as DD


class Element:
    def __init__(self,number,symbol,name):
        self.number = number
        self.symbol = symbol
        self.name = name


class ElementTable:

    def __init__(self):
        self.chemical_elements = DD(Element)
        self.chemical_dict = DD(list)
        self.chemical_table = None
        self._we_create_chemistry_()

    def _we_create_chemistry_(self):

        for el in get_all_elements():
            self.chemical_elements[el.atomic_number] = Element(
                el.atomic_number, el.symbol, el.name)

        for el_id, el_values in self.chemical_elements.items():
            self.chemical_dict[el_id] = [el_values.number, el_values.symbol, el_values.name]

        self.chemical_table = pd.DataFrame.from_dict(self.chemical_dict).T
        self.chemical_table.columns = ['number','symbol','name']
        return

if __name__ == '__main__':
    element_table = ElementTable()
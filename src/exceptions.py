class NotEnoughWords(Exception):
    """raised when number of passed words for a label is less than 5"""

    pass


class NoLabelsDefined(Exception):
    """raised when predict is called before add_labels"""

    pass


class InvalidClassificationMethod(Exception):
    """raised when an invalid classification method is passed"""

    pass

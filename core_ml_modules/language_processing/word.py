class Word:
    """
    Word object containing its original form, canonical form (head of the cluster it belongs to), and part-of-speech tag.
    :type original_form: str
    :type canonical_form: str
    :type pos_tag: str
    """
    original_form = None
    canonical_form = None
    pos_tag = None

    def __init__(self, original_form):
        self.original_form = original_form

    def set_canonical_form(self, canonical_form):
        """
        Sets the canonical form. Raises ValueError if it attempts to set a different canonical form from an existing one.
        """
        if self.canonical_form != None and self.canonical_form != canonical_form:
            raise ValueError("Word is already has the canonical form {}".format(self.canonical_form))

        self.canonical_form = canonical_form

    def set_pos_tag(self, pos_tag):
        """
        Sets the canonical form. Raises ValueError if it attempts to set a different part-of-speech tag from an existing one.
        """
        if self.pos_tag != None and self.pos_tag != pos_tag:
            raise ValueError("Word is already has the canonical form {}".format(self.pos_tag))

        self.pos_tag = pos_tag


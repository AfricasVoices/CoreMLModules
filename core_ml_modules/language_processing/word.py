class Word:
    """
    Word object containing its original form, canonical form (head of the cluster it belongs to), and part-of-speech tag.
    :type original_form: str
    :type canonical_form: str
    :type pos_tag: str
    """
    _original_form = None
    _canonical_form = None
    _pos_tag = None

    def __init__(self, original_form):
        self._original_form = original_form

    def get_original_form(self):
        return self._original_form

    def get_canonical_form(self):
        return self._canonical_form

    def get_pos_tag(self):
        return self._pos_tag

    def set_canonical_form(self, canonical_form):
        """
        Sets the canonical form. Raises ValueError if it attempts to set a different canonical form from an existing one.
        """
        if self._canonical_form != None and self._canonical_form != canonical_form:
            raise ValueError("Word is already has the canonical form {}".format(self._canonical_form))

        self._canonical_form = canonical_form

    def set_pos_tag(self, pos_tag):
        """
        Sets the canonical form. Raises ValueError if it attempts to set a different part-of-speech tag from an existing one.
        """
        if self._pos_tag != None and self._pos_tag != pos_tag:
            raise ValueError("Word is already has the canonical form {}".format(self._pos_tag))

        self._pos_tag = pos_tag

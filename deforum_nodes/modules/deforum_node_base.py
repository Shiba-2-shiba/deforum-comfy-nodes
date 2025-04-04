class DeforumDataBase:

    # @classmethod
    # def INPUT_TYPES(s):
    #     return s.params

    RETURN_TYPES = (("deforum_data",))
    FUNCTION = "get"
    OUTPUT_NODE = True
    CATEGORY = f"deforum/parameters"
    @classmethod
    # def IS_CHANGED(cls, text, autorefresh, **kwargs):
    def IS_CHANGED(cls, **kwargs): # 再改訂後のシグネチャ
        # Force re-evaluation of the node
        autorefresh = kwargs.get('autorefresh', 'No')
        if autorefresh == "Yes":
            return float("NaN")
        return None
    def get(self, deforum_data=None, *args, **kwargs):

        if deforum_data:
            deforum_data.update(**kwargs)
        else:
            deforum_data = kwargs
        return (deforum_data,)

class Query:
    def __init__(self, query: str):
        self._query = query

    @property
    def query(self) -> str:
        return self._query

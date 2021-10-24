class NetworkException(Exception):
    def __init__(self, content: str, status_code: int):
        self.message = f"Network request returned {status_code} code: {content}"
        super(NetworkException, self).__init__(self.message)

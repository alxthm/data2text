class WarningsFilter:
    def __init__(self, stream):
        """
        Filter some repetitive warnings to keep a clean stdout

        Args:
            stream: can be sys.stdout or sys.stderr for instance
        """
        self.stream = stream
        self.strings_to_filter = [
            "Check your hdfs-site.xml file",
            "pyarrow.hdfs.connect is deprecated",
        ]

    def __getattr__(self, attr_name):
        return getattr(self.stream, attr_name)

    def write(self, data):
        # check there is no forbidden string in the output data
        if all(s not in data for s in self.strings_to_filter):
            self.stream.write(data)
            self.stream.flush()

    def flush(self):
        self.stream.flush()


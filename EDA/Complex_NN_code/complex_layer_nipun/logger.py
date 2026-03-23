import datetime

class Logger:
    def __init__(self, filepath):
        self.filepath = filepath
        with open(self.filepath, "w", encoding="utf-8") as f:
            f.write(f"=== LOG START ===\n")

    def log(self, msg):
        print(msg)
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    def separator(self):
        self.log("-" * 80)

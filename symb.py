class SymbDict:
    def __init__(self):
        self.symbol_dict = []
        self.current_index = 0

    def symbol_exists(self, s):
        for e in self.symbol_dict:
            if e[1] == s:
                return True

    def insert_symbol(self, s):
        if not self.symbol_exists(s):
            self.symbol_dict.append([self.current_index, s])
            self.current_index += 1

    def get_dict_length(self):
        return len(self.symbol_dict)

    def get_index_by_symbol(self, s):
        for e in self.symbol_dict:
            if e[1] == s:
                return e[0]

    def get_symbol_by_index(self, i):
        for e in self.symbol_dict:
            if e[0] == i:
                return e[1]

    def clear(self):
        self.symbol_dict = []
        self.current_index = 0



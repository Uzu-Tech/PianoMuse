from dataclasses import dataclass

@dataclass
class Node():
    value: object
    idx: int
    previous: 'Node' = None
    next: 'Node' = None

class LinkedArray():
    def __init__(self, items):
        self._items = []
        previous = None
        # Create nodes for each item that reference the next and previous node
        for idx, item in enumerate(items):
            node = Node(item, idx, previous)
            if previous is not None:
                previous.next = node
            previous = node
            self._items.append(node)
            
    def __getitem__(self, idx):
        if self._items[idx] is None:
            raise ValueError(f"No node at index '{idx}'")
        return self._items[idx]
    
    def __setitem__(self, idx, value):
        if self._items[idx] is None:
            raise ValueError(f"No node at index '{idx}'")
        self._items[idx] = value
    
    def get_previous_idx(self, idx):
        previous = self[idx].previous
        if previous is None:
            return None
        return previous.idx
    
    def get_next_idx(self, idx):
        next = self[idx].next
        if next is None:
            return None
        return next.idx
    
    def get_second_next_idx(self, idx):
        next = self[idx].next
        if next is None or next.next is None:
            return None
        return next.next.idx
    
    def get_second_previous_idx(self, idx):
        previous = self[idx].previous
        if previous is None or previous.previous is None:
            return None
        return previous.previous.idx
    
    def __len__(self):
        return len(self._items)
    
    def get_list(self):
        if len(self._items) == 0:
            return []
        
        item = self._items[0]
        list = []
        while item is not None:
            list.append(item.value)
            if item.next is not None:
                item = self._items[item.next.idx]
            else:
                item = None
        
        return list
    
    def merge_pair(self, idx, new_value):
        if idx > len(self._items) - 2 or self[idx].next is None:
            raise ValueError("Invalid Index")

        self[idx].value = new_value
        self[self[idx].next.idx] = None
        self[idx].next = self[idx].next.next
        if self[idx].next is not None:
            self[idx].next.previous = self[idx]

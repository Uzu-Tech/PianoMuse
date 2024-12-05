class MaxPriorityMap:
    def __init__(self, heap_key, map_key):
        self._heap_key = heap_key
        self._map_key = map_key
        self._heap = []  # List to store heap elements
        self._map = {}  # Dictionary to map keys to their indices in the heap

    def peek(self):
        if len(self) == 0:
            raise ValueError("Empty heap")
        # Return the top element of the heap
        return self._heap[0]

    def push(self, item):
        # Add the new item to the end of the heap
        self._heap.append(item)
        map_key = self._map_key(item)
        if map_key in self._map:
            raise ValueError("Duplicate item in map")
        # Update the map
        self._map[map_key] = len(self) - 1
        # Restore the heap property
        self._heapify_up(len(self) - 1)

    def pop(self):
        if len(self) == 0:
            raise ValueError("Empty heap")

        # Swap bottom and top of heap
        self._swap(0, len(self) - 1)
        # Remove the max from the heap and the map
        max = self._heap.pop()
        map_key = self._map_key(max)
        self._map.pop(map_key)

        # Restore heap property
        if len(self) != 0:
            self._heapify_down(0)

        return max

    def __contains__(self, map_key):
        return map_key in self._map

    def __len__(self):
        return len(self._heap)

    def pop_by_map_key(self, map_key):
        if map_key not in self._map:
            raise ValueError(f"Map key {map_key} not found")
        # Get the position on the heap from the map
        item_pos = self._map[map_key]
        # Pop item at the position
        self._swap(item_pos, len(self) - 1)
        item = self._heap.pop()
        self._map.pop(map_key)
        # Heapify if necessary
        if item_pos < len(self):
            self._heapify_up(item_pos)
            self._heapify_down(item_pos)
        return item

    def _heapify_up(self, idx):
        # Continuously bubble up the tree until it's smaller than its parent
        while idx > 0 and self._heap_key(self._heap[idx]) > self._heap_key(
            self._heap[self._get_parent(idx)]
        ):
            self._swap(idx, self._get_parent(idx))
            idx = self._get_parent(idx)

    def _heapify_down(self, idx):
        # Get children branches
        max_idx = idx
        left_idx = self._get_left_child(idx)
        right_idx = self._get_right_child(idx)

        # Compare each branch with parent to get the largest between them
        if left_idx < len(self) and self._heap_key(
            self._heap[left_idx]
        ) > self._heap_key(self._heap[max_idx]):
            max_idx = left_idx

        if right_idx < len(self) and self._heap_key(
            self._heap[right_idx]
        ) > self._heap_key(self._heap[max_idx]):
            max_idx = right_idx

        # Swap and repeat again if needed
        if max_idx != idx:
            self._swap(idx, max_idx)
            self._heapify_down(max_idx)

    def _get_parent(self, idx):
        return (idx - 1) // 2

    def _get_right_child(self, idx):
        return 2 * idx + 2

    def _get_left_child(self, idx):
        return 2 * idx + 1

    def _swap(self, i, j):
        # Swap values in map
        self._map[self._map_key(self._heap[i])] = j
        self._map[self._map_key(self._heap[j])] = i
        # Swap on heap
        self._heap[i], self._heap[j] = self._heap[j], self._heap[i]

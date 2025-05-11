# supermarket_model.py
import numpy as np

# --- ID Constants ---
PATHWAY_ID = 0
OBSTACLE_ID = 1
AP_ID = 2
STALL_ID_START = 50
STALL_ID_END = 99
ITEM_ID_START = 100

class SupermarketMap:
    def __init__(self, width_m, height_m, resolution_m):
        self.width_m = width_m
        self.height_m = height_m
        self.resolution_m = resolution_m

        self.num_cols = int(width_m / resolution_m)
        self.num_rows = int(height_m / resolution_m)

        self.grid_map = np.full((self.num_rows, self.num_cols), PATHWAY_ID, dtype=int)
        
        self.stall_definitions = {}  # stall_id -> {"name": name, "coords": (r,c,h,w)}
        self.item_definitions = {}   # item_id -> {"name": name}
        self.item_locations = {}     # (row, col) -> item_id (grid cells occupied by items)
        self.access_points = []      # list of (r, c) tuples

        self._next_stall_id = STALL_ID_START
        self._next_item_id = ITEM_ID_START

    def _get_next_stall_id(self):
        if self._next_stall_id > STALL_ID_END:
            raise ValueError("Ran out of Stall IDs!")
        current_id = self._next_stall_id
        self._next_stall_id += 1
        return current_id

    def _get_next_item_id(self):
        current_id = self._next_item_id
        self._next_item_id += 1
        return current_id

    def _is_within_bounds(self, r, c, h=1, w=1):
        return 0 <= r < self.num_rows and \
               0 <= c < self.num_cols and \
               r + h <= self.num_rows and \
               c + w <= self.num_cols

    def add_general_obstacle(self, r_start, c_start, obs_height, obs_width):
        if not self._is_within_bounds(r_start, c_start, obs_height, obs_width):
            print(f"Error: Obstacle at ({r_start},{c_start}) size ({obs_height}x{obs_width}) "
                  f"is out of bounds. Not added.")
            return False
        
        # Check for overlap with critical elements (e.g., APs, existing items)
        # For simplicity, allow overwriting pathways. More complex checks can be added.
        self.grid_map[r_start : r_start + obs_height, c_start : c_start + obs_width] = OBSTACLE_ID
        print(f"Added general obstacle at ({r_start},{c_start}) size ({obs_height}x{obs_width}).")
        return True

    def add_stall_area(self, r_start, c_start, stall_height, stall_width, stall_name):
        if not self._is_within_bounds(r_start, c_start, stall_height, stall_width):
            print(f"Error: Stall '{stall_name}' at ({r_start},{c_start}) size ({stall_height}x{stall_width}) "
                  f"is out of bounds. Not added.")
            return -1

        target_area = self.grid_map[r_start : r_start + stall_height, c_start : c_start + stall_width]
        if np.any(target_area != PATHWAY_ID):
            print(f"Warning: Stall '{stall_name}' at ({r_start},{c_start}) overlaps with existing objects. "
                  f"Not added. Affected values: {np.unique(target_area[target_area != PATHWAY_ID])}")
            return -1
            
        stall_id = self._get_next_stall_id()
        self.grid_map[r_start : r_start + stall_height, c_start : c_start + stall_width] = stall_id
        self.stall_definitions[stall_id] = {
            "name": stall_name,
            "coords": (r_start, c_start, stall_height, stall_width)
        }
        print(f"Added stall '{stall_name}' (ID: {stall_id}) at ({r_start},{c_start}) size ({stall_height}x{stall_width}).")
        return stall_id

    def add_item_to_grid(self, r_grid, c_grid, item_height, item_width, item_name):
        if not self._is_within_bounds(r_grid, c_grid, item_height, item_width):
            print(f"Error: Item '{item_name}' at ({r_grid},{c_grid}) size ({item_height}x{item_width}) "
                  f"is out of bounds. Not added.")
            return -1

        target_cells_values = self.grid_map[r_grid : r_grid + item_height, c_grid : c_grid + item_width]
        
        # Items can be placed on Stalls or general Obstacles (if obstacles are shelves)
        allowed_base_values = ((target_cells_values >= STALL_ID_START) & (target_cells_values <= STALL_ID_END)) | \
                              (target_cells_values == OBSTACLE_ID)
        
        if not np.all(allowed_base_values):
            print(f"Warning: Item '{item_name}' at ({r_grid},{c_grid}) cannot be placed. "
                  f"Target cells are not valid stall/obstacle areas. "
                  f"Invalid values: {np.unique(target_cells_values[~allowed_base_values])}")
            return -1

        item_id_to_assign = -1
        for id_val, props in self.item_definitions.items():
            if props["name"] == item_name:
                item_id_to_assign = id_val
                break
        if item_id_to_assign == -1:
            item_id_to_assign = self._get_next_item_id()
            self.item_definitions[item_id_to_assign] = {"name": item_name}

        self.grid_map[r_grid : r_grid + item_height, c_grid : c_grid + item_width] = item_id_to_assign
        for r_offset in range(item_height):
            for c_offset in range(item_width):
                self.item_locations[(r_grid + r_offset, c_grid + c_offset)] = item_id_to_assign
        
        print(f"Placed item '{item_name}' (ID: {item_id_to_assign}) at ({r_grid},{c_grid}) size ({item_height}x{item_width}).")
        return item_id_to_assign

    def add_access_point(self, r_ap, c_ap):
        if not self._is_within_bounds(r_ap, c_ap):
            print(f"Warning: AP at ({r_ap},{c_ap}) is out of bounds. Not added.")
            return False
        
        if self.grid_map[r_ap, c_ap] == PATHWAY_ID:
            self.grid_map[r_ap, c_ap] = AP_ID
            self.access_points.append((r_ap, c_ap))
            # print(f"Added AP at ({r_ap},{c_ap}).")
            return True
        else:
            print(f"Warning: Cannot place AP at ({r_ap},{c_ap}). Location occupied by ID {self.grid_map[r_ap, c_ap]}.")
            return False

    def get_item_locations_by_name(self, item_name):
        """Returns a list of (r,c) tuples for all cells containing the item."""
        item_id_to_find = -1
        for id_val, props in self.item_definitions.items():
            if props["name"] == item_name:
                item_id_to_find = id_val
                break
        if item_id_to_find == -1:
            return []
        
        locations = []
        for (r,c), id_val in self.item_locations.items():
            if id_val == item_id_to_find:
                locations.append((r,c))
        return locations

    def get_item_name_at_location(self, r, c):
        if self._is_within_bounds(r,c):
            value = self.grid_map[r,c]
            if value >= ITEM_ID_START:
                return self.item_definitions.get(value, {}).get("name", f"Unknown Item ({value})")
        return None
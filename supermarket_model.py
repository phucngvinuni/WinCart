# supermarket_model.py
import numpy as np
# Không cần import config ở đây nữa nếu config không dùng ID từ đây

# --- ID Constants ---
PATHWAY_ID = 0
OBSTACLE_ID = 1 # Vật cản chung (tường, cột, ...)
AP_ID = 2       # Ô chứa AP (có thể là lối đi được đánh dấu)
STALL_ID_START = 50
STALL_ID_END = 99
ITEM_ID_START = 100 # ID cho các ô chứa mặt hàng cụ thể

class SupermarketMap:
    def __init__(self, width_m, height_m, resolution_m):
        self.width_m = width_m
        self.height_m = height_m
        self.resolution_m = resolution_m

        self.num_cols = int(width_m / resolution_m)
        self.num_rows = int(height_m / resolution_m)

        self.grid_map = np.full((self.num_rows, self.num_cols), PATHWAY_ID, dtype=int)

        self.stall_definitions = {}  # stall_id -> {"name": name, "r": r, "c": c, "h": h, "w": w}
        self.item_definitions = {}   # item_id -> {"name": name}
        # item_locations_on_grid: (row, col) -> item_id (ô nào chứa item_id nào)
        self.item_locations_on_grid = {}
        # item_to_stall_map: item_name -> stall_id (mặt hàng nào thuộc gian hàng nào - tùy chọn)
        self.item_to_stall_map = {}
        # approachable_item_locations: item_name -> list of (r,c) for pathfinding target
        self.approachable_item_locations = {}
        self.access_points = []      # list of (r, c) tuples

        self._next_stall_id = STALL_ID_START
        self._next_item_id = ITEM_ID_START
        print(f"Supermarket initialized: {self.num_rows} rows, {self.num_cols} cols.")

    def _get_next_stall_id(self):
        if self._next_stall_id > STALL_ID_END:
            # Tự động mở rộng nếu cần, hoặc raise lỗi nghiêm ngặt hơn
            print(f"Warning: Exceeded STALL_ID_END ({STALL_ID_END}). Extending range.")
            # STALL_ID_END = self._next_stall_id # Hoặc một cơ chế khác
            # Hoặc đơn giản là raise lỗi nếu không muốn mở rộng
            # raise ValueError("Ran out of Stall IDs!")
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
            print(f"Error: Obstacle at ({r_start},{c_start}) size ({obs_height}x{obs_width}) out of bounds.")
            return False
        self.grid_map[r_start : r_start + obs_height, c_start : c_start + obs_width] = OBSTACLE_ID
        # print(f"Added general obstacle at ({r_start},{c_start}) size ({obs_height}x{obs_width}).")
        return True

    def add_stall_area(self, r_start, c_start, stall_height, stall_width, stall_name):
        if not self._is_within_bounds(r_start, c_start, stall_height, stall_width):
            print(f"Error: Stall '{stall_name}' at ({r_start},{c_start}) size ({stall_height}x{stall_width}) out of bounds.")
            return -1

        target_area = self.grid_map[r_start : r_start + stall_height, c_start : c_start + stall_width]
        if np.any(target_area != PATHWAY_ID):
            occupied_ids = np.unique(target_area[target_area != PATHWAY_ID])
            print(f"Warning: Stall '{stall_name}' at ({r_start},{c_start}) overlaps. Occupied by IDs: {occupied_ids}. Not added.")
            return -1

        stall_id = self._get_next_stall_id()
        self.grid_map[r_start : r_start + stall_height, c_start : c_start + stall_width] = stall_id
        self.stall_definitions[stall_id] = {
            "name": stall_name,
            "r": r_start, "c": c_start, "h": stall_height, "w": stall_width
        }
        print(f"Added stall '{stall_name}' (ID: {stall_id}) at ({r_start},{c_start}).")
        return stall_id

    def add_item_to_grid(self, r_item_area_start, c_item_area_start, item_area_height, item_area_width, item_name, on_stall_id=None):
        if not self._is_within_bounds(r_item_area_start, c_item_area_start, item_area_height, item_area_width):
            print(f"Error: Item area '{item_name}' at ({r_item_area_start},{c_item_area_start}) out of bounds.")
            return -1

        target_cells_values = self.grid_map[r_item_area_start : r_item_area_start + item_area_height,
                                           c_item_area_start : c_item_area_start + item_area_width]

        # Items must be placed on a stall or a general obstacle (acting as a shelf)
        is_on_stall = (target_cells_values >= STALL_ID_START) & (target_cells_values <= STALL_ID_END)
        is_on_obstacle_shelf = (target_cells_values == OBSTACLE_ID)
        allowed_base = is_on_stall | is_on_obstacle_shelf

        if not np.all(allowed_base):
            invalid_values = np.unique(target_cells_values[~allowed_base])
            print(f"Warning: Item '{item_name}' at ({r_item_area_start},{c_item_area_start}) cannot be placed. "
                  f"Base cells are not valid stalls or obstacles. Invalid base IDs: {invalid_values}")
            return -1

        # Get or create item_id
        item_id_to_assign = -1
        for id_val, props in self.item_definitions.items():
            if props["name"] == item_name:
                item_id_to_assign = id_val
                break
        if item_id_to_assign == -1:
            item_id_to_assign = self._get_next_item_id()
            self.item_definitions[item_id_to_assign] = {"name": item_name}

        # Assign item_id to the grid cells
        self.grid_map[r_item_area_start : r_item_area_start + item_area_height,
                      c_item_area_start : c_item_area_start + item_area_width] = item_id_to_assign

        # Store individual cell locations for this item
        for r_offset in range(item_area_height):
            for c_offset in range(item_area_width):
                r_abs, c_abs = r_item_area_start + r_offset, c_item_area_start + c_offset
                self.item_locations_on_grid[(r_abs, c_abs)] = item_id_to_assign
        
        if on_stall_id and on_stall_id in self.stall_definitions:
            self.item_to_stall_map[item_name] = on_stall_id

        print(f"Placed item '{item_name}' (ID: {item_id_to_assign}) in area starting at ({r_item_area_start},{c_item_area_start}).")
        self._update_approachable_location(item_name, r_item_area_start, c_item_area_start, item_area_height, item_area_width)
        return item_id_to_assign

    def _update_approachable_location(self, item_name, item_area_r, item_area_c, item_area_h, item_area_w, preferred_side=None):
        """Finds and stores an approachable pathway cell for an item area."""
        # Create a list of all cells belonging to this item's area on the shelf
        item_shelf_cells = []
        for r_offset in range(item_area_h):
            for c_offset in range(item_area_w):
                r, c = item_area_r + r_offset, item_area_c + c_offset
                # Check if (r,c) is actually part of the item (already set on grid_map)
                if self._is_within_bounds(r,c) and self.grid_map[r,c] >= ITEM_ID_START : # or it's on a STALL_ID that will become item
                    item_shelf_cells.append((r,c))

        if not item_shelf_cells:
             # This might happen if add_item_to_grid failed to place the item.
             # Or if called before the grid_map is updated with item_id.
             # Let's try to find approachable spots based on the intended item area.
            for r_offset in range(item_area_h):
                for c_offset in range(item_area_w):
                     item_shelf_cells.append((item_area_r + r_offset, item_area_c + c_offset))


        approachable_spot = self.find_accessible_spot_near_generic_area(item_shelf_cells, preferred_side)
        if approachable_spot:
            if item_name not in self.approachable_item_locations:
                self.approachable_item_locations[item_name] = []
            if approachable_spot not in self.approachable_item_locations[item_name]: # Avoid duplicates
                self.approachable_item_locations[item_name].append(approachable_spot)
                print(f"  Approachable spot for '{item_name}': {approachable_spot}")
        else:
            print(f"  Warning: Could not find approachable spot for '{item_name}' in area ({item_area_r},{item_area_c}).")


    def find_accessible_spot_near_generic_area(self, area_cells, preferred_side=None):
        """Generic function to find pathway near a list of cells (shelf, item area, stall)."""
        if not area_cells: return None
        candidate_spots_with_side_info = []
        for r_area, c_area in area_cells:
            # Check 4 neighbors
            for dr, dc, side_name in [(-1,0,'top'), (1,0,'bottom'), (0,-1,'left'), (0,1,'right')]:
                nr, nc = r_area + dr, c_area + dc
                if self._is_within_bounds(nr, nc) and self.grid_map[nr, nc] == PATHWAY_ID:
                    candidate_spots_with_side_info.append(((nr, nc), side_name))
        
        if not candidate_spots_with_side_info: return None

        # Apply preferred_side logic
        if preferred_side:
            sides_to_check = [preferred_side] if isinstance(preferred_side, str) else preferred_side
            for p_side in sides_to_check:
                preferred_options = [spot for spot, side in candidate_spots_with_side_info if side == p_side]
                if preferred_options:
                    # Return a "central" spot among preferred options
                    avg_r = sum(r for r,c in preferred_options) / len(preferred_options)
                    avg_c = sum(c for r,c in preferred_options) / len(preferred_options)
                    return min(preferred_options, key=lambda s: ((s[0]-avg_r)**2 + (s[1]-avg_c)**2))
        
        # Fallback: return a "central" spot from all candidates
        # relative to the center of the area_cells
        if not area_cells: return None # Should have been caught earlier
        area_center_r = sum(r for r,c in area_cells) / len(area_cells)
        area_center_c = sum(c for r,c in area_cells) / len(area_cells)
        all_actual_candidate_spots = list(set([spot for spot, side in candidate_spots_with_side_info])) # Unique spots
        return min(all_actual_candidate_spots, key=lambda s: ((s[0]-area_center_r)**2 + (s[1]-area_center_c)**2))


    def add_access_point(self, r_ap, c_ap):
        if not self._is_within_bounds(r_ap, c_ap):
            print(f"Warning: AP at ({r_ap},{c_ap}) out of bounds.")
            return False
        if self.grid_map[r_ap, c_ap] == PATHWAY_ID:
            self.grid_map[r_ap, c_ap] = AP_ID # Mark the cell as AP
            self.access_points.append((r_ap, c_ap))
            return True
        # Allow placing AP on an existing AP_ID cell (idempotent)
        elif self.grid_map[r_ap, c_ap] == AP_ID:
            if (r_ap, c_ap) not in self.access_points: # Should not happen if logic is correct
                self.access_points.append((r_ap,c_ap))
            return True
        else:
            print(f"Warning: Cannot place AP at ({r_ap},{c_ap}). Location occupied by ID {self.grid_map[r_ap, c_ap]}.")
            return False

    def get_item_locations_by_name(self, item_name_query):
        """Returns a list of (r,c) for cells occupied by the item on the grid_map."""
        item_id_to_find = -1
        for id_val, props in self.item_definitions.items():
            if props["name"].lower() == item_name_query.lower(): # Case-insensitive search
                item_id_to_find = id_val
                break
        if item_id_to_find == -1: return []
        
        return [loc for loc, item_id in self.item_locations_on_grid.items() if item_id == item_id_to_find]

    def get_approachable_item_location_by_name(self, item_name_query, current_cart_pos=None):
        """Returns the most suitable approachable (pathway) (r,c) for an item."""
        item_name_found = None
        for defined_id, props in self.item_definitions.items():
            if props["name"].lower() == item_name_query.lower():
                item_name_found = props["name"] # Get the canonical name
                break
        
        if not item_name_found or item_name_found not in self.approachable_item_locations:
            return None
            
        possible_targets = self.approachable_item_locations[item_name_found]
        if not possible_targets: return None

        if current_cart_pos and len(possible_targets) > 1:
            return min(possible_targets, key=lambda target_pos:
                       ((target_pos[0] - current_cart_pos[0])**2 +
                        (target_pos[1] - current_cart_pos[1])**2))
        return possible_targets[0]

    def get_stall_approachable_location(self, stall_name_query, current_cart_pos=None):
        stall_id_found = -1
        stall_props_found = None
        for id_val, props in self.stall_definitions.items():
            if props["name"].lower() == stall_name_query.lower():
                stall_id_found = id_val
                stall_props_found = props
                break
        
        if not stall_props_found: return None

        # Find an approachable spot for this stall
        # Stall itself is defined by r,c,h,w
        stall_r, stall_c, stall_h, stall_w = stall_props_found['r'], stall_props_found['c'], \
                                             stall_props_found['h'], stall_props_found['w']
        stall_cells = []
        for r_offset in range(stall_h):
            for c_offset in range(stall_w):
                stall_cells.append((stall_r + r_offset, stall_c + c_offset))
        
        # For stalls, maybe prefer sides?
        approachable_spot = self.find_accessible_spot_near_generic_area(stall_cells, preferred_side=['left', 'right','top','bottom']) # Try all sides
        return approachable_spot


    def get_item_name_at_grid_location(self, r, c): # Renamed from get_item_name_at_location
        if self._is_within_bounds(r,c):
            value = self.grid_map[r,c]
            if value >= ITEM_ID_START:
                return self.item_definitions.get(value, {}).get("name", f"Unknown Item ({value})")
            elif STALL_ID_START <= value <= STALL_ID_END:
                return self.stall_definitions.get(value, {}).get("name", f"Unknown Stall ({value})")
            elif value == AP_ID: return "Access Point"
            elif value == OBSTACLE_ID: return "Vật cản"
            elif value == PATHWAY_ID: return "Lối đi"
        return None
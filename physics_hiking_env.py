# physics_hiking_env.py
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
import rasterio
import gpxpy
import gpxpy.gpx
from pyproj import CRS, Transformer


class RealisticHikingEnv(gym.Env):
    """
    Physics-driven hiking environment.

    Key properties:
    - Start at trailhead (GPX first point), goal at trail end (last point).
    - Movement constrained by terrain physics: step height, slope limit, impassables.
    - Rewards ONLY for survival + reaching the goal (no trail bias).
    - Episode ends ONLY on goal/health<=0/energy<=0 (no step cap unless you pass max_steps).
    - Exposes `transform` and `raster_crs` for GPX export in visualisation.
    """

    metadata = {"render_modes": ["rgb_array"]}

    # -------------------- Init --------------------
    def __init__(
        self,
        processed_data_dir: str | Path = "data/processed",
        patch_size: int = 64,
        max_steps: int | None = None,
        auto_save_gpx: bool = True,
        rng_seed: int | None = None,
        curriculum_learning: bool = False,
        start_distance_meters: float = 500.0,  # Start this close to goal for curriculum
        include_goal_in_obs: bool = False,     # Add goal direction to observations
    ):
        super().__init__()
        self.processed_dir = Path(processed_data_dir)
        self.patch_size = int(patch_size)
        self.max_steps = max_steps  # if None -> no truncation on steps
        self.auto_save_gpx = auto_save_gpx
        self.np_rng = np.random.default_rng(rng_seed)
        
        # Curriculum learning parameters
        self.curriculum_learning = curriculum_learning
        self.start_distance_meters = start_distance_meters
        self.include_goal_in_obs = include_goal_in_obs
        self.curriculum_successes = 0  # Track successful goal reaches
        self.curriculum_attempts = 0   # Track total attempts

        # ---- Physics params (tuned) ----
        self.MAX_SAFE_SLOPE = 45.0        # deg baseline; eased by grip (increased from 35.0)
        self.MAX_STEP_HEIGHT = 1.0        # m vertical step limit (increased from 0.6)
        self.ENERGY_MAX = 1e9             # Disabled energy system (was 200.0)
        self.HEALTH_MAX = 100.0

        # Energy drain scales
        self.ELEVATION_ENERGY_SCALE = 0.2   # per meter climbed (reduced from 0.5)
        self.SLOPE_ENERGY_SCALE = 0.02      # per degree at next cell (reduced from 0.05)
        self.VEG_ENERGY_SCALE = 0.2         # proportional to veg cost (reduced from 0.5)

        # Slip / damage
        self.SLOPE_SLIP_THRESHOLD = 60.0    # deg (increased from 20.0)
        self.SLIP_DAMAGE_SCALE = 0.05       # HP lost ~ slope*scale (reduced from 0.2)

        # ---- Load terrain layers ----
        self._load_maps()

        self.map_h, self.map_w = self.elevation_map.shape

        # ---- Agent state ----
        self.current_pos = np.zeros(2, dtype=np.float32)  # (row, col), subpixel
        self.velocity = np.zeros(2, dtype=np.float32)
        self.energy = self.ENERGY_MAX
        self.health = self.HEALTH_MAX
        self.step_count = 0
        self.trajectory: list[np.ndarray] = []
        self.goal: np.ndarray = np.zeros(2, dtype=np.float32)

        # ---- Gym spaces ----
        self.action_space = spaces.Discrete(9)  # 8 dirs + rest
        
        # Base observation space
        obs_dict = {
            "terrain_rgb": spaces.Box(
                low=0, high=255, shape=(3, self.patch_size, self.patch_size), dtype=np.uint8
            ),
            "physical_state": spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32),
        }
        
        # Add goal information if requested
        if self.include_goal_in_obs:
            obs_dict["goal_info"] = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
            
        self.observation_space = spaces.Dict(obs_dict)

    # -------------------- Data loading --------------------
    def _load_maps(self):
        """Load DEM + processed layers. Also expose transform/CRS for GPX export."""
        # DEM from raw to get true geo transform/CRS
        raw_dem = self.processed_dir.parent / "raw" / "dem_st_helens.tif"
        with rasterio.open(raw_dem) as src:
            self.elevation_map = src.read(1).astype(np.float32)
            self.transform = src.transform       # for pixel->map coords
            self.raster_crs = src.crs
            raw_cell_size = float(src.res[0])
            
            # Convert degrees to meters if DEM is in geographic coordinates (EPSG:4326)
            if src.crs.to_epsg() == 4326:
                # Convert geographic coordinates to ground distance
                # St. Helens area is approximately 46.2°N latitude
                latitude = 46.2
                lat_rad = np.radians(latitude)
                
                # 1 degree longitude = cos(latitude) * 111,320 meters
                # 1 degree latitude = 111,320 meters (constant)
                meters_per_degree_lon = np.cos(lat_rad) * 111320
                meters_per_degree_lat = 111320
                
                # Use longitude conversion (more conservative)
                self.cell_size = raw_cell_size * meters_per_degree_lon
            else:
                self.cell_size = raw_cell_size

        # Slope
        with rasterio.open(self.processed_dir / "slope.tif") as src:
            self.slope_map = src.read(1).astype(np.float32)

        # Vegetation movement cost (includes grip adjustments)
        with rasterio.open(self.processed_dir / "vegetation_cost.tif") as src:
            self.vegetation_cost = src.read(1).astype(np.float32)

        # RGB terrain for observation
        try:
            with rasterio.open(self.processed_dir / "terrain_rgb.tif") as src:
                rgb = src.read().transpose(1, 2, 0)  # HWC
            self.terrain_rgb = rgb.astype(np.uint8)
        except Exception:
            # Elevation shaded grayscale fallback
            e = (self.elevation_map - self.elevation_map.min()) / (self.elevation_map.ptp() + 1e-9)
            gray = (e * 255).astype(np.uint8)
            self.terrain_rgb = np.repeat(gray[..., None], 3, axis=2)

        # Optional trail coords (pixel space)
        tr_path = self.processed_dir / "trail_coordinates.npy"
        self.trail_coords = np.load(tr_path) if tr_path.exists() else None

        # Transformer for lon/lat export convenience (for internal use if needed)
        self.to_lonlat = Transformer.from_crs(self.raster_crs, CRS.from_epsg(4326), always_xy=True)

    # -------------------- Helpers --------------------
    def _get_terrain_at(self, pos: np.ndarray):
        r = int(np.clip(pos[0], 0, self.map_h - 1))
        c = int(np.clip(pos[1], 0, self.map_w - 1))
        return {
            "elev": self.elevation_map[r, c],
            "slope": self.slope_map[r, c],
            "veg": self.vegetation_cost[r, c],
            "r": r,
            "c": c,
        }

    def _movement_is_valid(self, cur: np.ndarray, nxt: np.ndarray):
        # 1) Hard out-of-bounds block
        if nxt[0] < 0 or nxt[0] >= self.map_h or nxt[1] < 0 or nxt[1] >= self.map_w:
            return False, "out_of_bounds"

        cur_t = self._get_terrain_at(cur)
        nx_t = self._get_terrain_at(nxt)

        # 2) Impassable (e.g., water/very high veg)
        if nx_t["veg"] >= 100.0:
            return False, "impassable"

        # 3) Step height block
        dh = abs(nx_t["elev"] - cur_t["elev"])
        if dh > self.MAX_STEP_HEIGHT:
            return False, "step_too_high"

        # 4) Slope limit (relaxed by grip)
        dist_m = np.linalg.norm(nxt - cur) * self.cell_size
        if dist_m > 0:
            move_slope = np.degrees(np.arctan(dh / (dist_m + 1e-9)))
            grip = min(nx_t["veg"], 5.0) / 5.0  # lower veg cost -> higher grip factor
            limit = self.MAX_SAFE_SLOPE + 10.0 * grip
            if move_slope > limit:
                return False, "too_steep"

        return True, "ok"

    def _apply_slip_if_needed(self, nxt: np.ndarray):
        """Occasional slip on steep cells, with bounded damage and clamped slide."""
        nx = self._get_terrain_at(nxt)
        if nx["slope"] <= self.SLOPE_SLIP_THRESHOLD:
            return nxt, "normal"

        # Probability increases with slope; reduced by veg (grip).
        slip_p = (nx["slope"] - self.SLOPE_SLIP_THRESHOLD) / 30.0
        slip_p *= 1.0 / max(1.0, nx["veg"])
        slip_p = np.clip(slip_p, 0.0, 1.0)

        if self.np_rng.random() < slip_p:
            r, c = nx["r"], nx["c"]
            # local gradient (edge-safe)
            up = self.elevation_map[max(r - 1, 0), c]
            down = self.elevation_map[min(r + 1, self.map_h - 1), c]
            left = self.elevation_map[r, max(c - 1, 0)]
            right = self.elevation_map[r, min(c + 1, self.map_w - 1)]
            grad = np.array([up - down, left - right], dtype=np.float32)
            n = np.linalg.norm(grad)
            if n > 0:
                dirn = grad / n
                dist_pix = min(3.0, nx["slope"] / 10.0)
                slid = nxt + dirn * dist_pix
                # clamp inside
                slid[0] = np.clip(slid[0], 0, self.map_h - 1)
                slid[1] = np.clip(slid[1], 0, self.map_w - 1)
                # bounded damage (no energy penalty for easier training)
                self.health = max(0.0, self.health - min(20.0, nx["slope"] * self.SLIP_DAMAGE_SCALE))
                # self.energy = max(0.0, self.energy - 5.0)  # Disabled energy penalty
                return slid, "slipped"
        return nxt, "normal"

    def _energy_cost(self, cur: np.ndarray, nxt: np.ndarray) -> float:
        # Energy system disabled for easier training
        return 0.0
        # Old energy calculation (disabled):
        # cur_t = self._get_terrain_at(cur)
        # nx_t = self._get_terrain_at(nxt)
        # gain = max(0.0, nx_t["elev"] - cur_t["elev"])
        # elev = self.ELEVATION_ENERGY_SCALE * gain
        # slope = self.SLOPE_ENERGY_SCALE * abs(nx_t["slope"])
        # veg = self.VEG_ENERGY_SCALE * nx_t["veg"]
        # return float(min(8.0, 0.25 + elev + slope + veg))

    def _goal_reached(self) -> bool:
        return np.linalg.norm(self.current_pos - self.goal) * self.cell_size < 3.0  # within 3 m

    def _obs(self):
        # RGB patch centered on agent, padded if near edges
        r, c = int(self.current_pos[0]), int(self.current_pos[1])
        hp = self.patch_size // 2
        r0, r1 = max(0, r - hp), min(self.map_h, r + hp)
        c0, c1 = max(0, c - hp), min(self.map_w, c + hp)

        patch = np.zeros((self.patch_size, self.patch_size, 3), dtype=np.uint8)
        view = self.terrain_rgb[r0:r1, c0:c1]
        pr = (self.patch_size - view.shape[0]) // 2
        pc = (self.patch_size - view.shape[1]) // 2
        patch[pr : pr + view.shape[0], pc : pc + view.shape[1]] = view
        terrain_rgb = patch.transpose(2, 0, 1)  # CHW

        cur_t = self._get_terrain_at(self.current_pos)
        dist_goal = np.linalg.norm(self.current_pos - self.goal) * self.cell_size
        physical = np.array(
            [
                self.current_pos[0] / self.map_h,
                self.current_pos[1] / self.map_w,
                self.velocity[0],
                self.velocity[1],
                self.energy / self.ENERGY_MAX,
                self.health / self.HEALTH_MAX,
                cur_t["slope"] / 45.0,
                cur_t["veg"] / 10.0,
                dist_goal / 100.0,
            ],
            dtype=np.float32,
        )
        
        obs = {"terrain_rgb": terrain_rgb, "physical_state": physical}
        
        # Add goal information if requested
        if self.include_goal_in_obs:
            # Goal direction and distance
            goal_direction = self.goal - self.current_pos
            goal_distance = np.linalg.norm(goal_direction)
            if goal_distance > 0:
                goal_direction = goal_direction / goal_distance  # Normalize
            
            goal_info = np.array([
                goal_direction[0],  # Normalized direction to goal (row)
                goal_direction[1],  # Normalized direction to goal (col) 
                dist_goal / 4000.0,  # Distance to goal (normalized)
            ], dtype=np.float32)
            
            obs["goal_info"] = goal_info
        
        return obs

    # -------------------- Gym API --------------------
    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        # Spawn at trailhead if available; goal at highest point on trail (summit).
        if self.trail_coords is not None and len(self.trail_coords) >= 2:
            # Find the highest elevation point along the trail (summit)
            max_elevation = -np.inf
            summit_idx = 0
            for i, trail_point in enumerate(self.trail_coords):
                r, c = int(trail_point[0]), int(trail_point[1])
                if 0 <= r < self.map_h and 0 <= c < self.map_w:
                    elevation = self.elevation_map[r, c]
                    if elevation > max_elevation:
                        max_elevation = elevation
                        summit_idx = i
            
            self.goal = self.trail_coords[summit_idx].astype(np.float32)
            
            if self.curriculum_learning:
                # Curriculum learning: start closer to goal, gradually increase distance
                self.curriculum_attempts += 1
                
                # Calculate current curriculum distance based on success rate
                success_rate = self.curriculum_successes / max(1, self.curriculum_attempts)
                if success_rate > 0.75 and self.curriculum_attempts > 10:  # More conservative advancement
                    # If doing well, DOUBLE the difficulty for faster progression
                    self.start_distance_meters = min(self.start_distance_meters * 2.0, 3748.0)  # Double distance!
                    print(f"Curriculum: DOUBLED difficulty to {self.start_distance_meters:.0f}m from goal (success rate: {success_rate:.1%})")
                    # Reset counters for new difficulty level
                    self.curriculum_successes = 0
                    self.curriculum_attempts = 0
                
                # Find trail point approximately start_distance_meters from summit
                target_distance_pixels = self.start_distance_meters / self.cell_size
                
                # Find all trail points within reasonable range of target distance
                candidates = []
                for i, trail_point in enumerate(self.trail_coords):
                    distance_to_summit = np.linalg.norm(trail_point - self.goal)
                    distance_diff = abs(distance_to_summit - target_distance_pixels)
                    
                    # Accept points within ±30% of target distance
                    if distance_diff <= target_distance_pixels * 0.3:
                        candidates.append((i, distance_diff))
                
                if candidates:
                    # Randomly select from valid candidates to add variety
                    best_idx = self.np_rng.choice([idx for idx, _ in candidates])
                else:
                    # Fallback: find closest trail point to target distance
                    best_idx = 0
                    best_distance_diff = float('inf')
                    
                    for i, trail_point in enumerate(self.trail_coords):
                        distance_to_summit = np.linalg.norm(trail_point - self.goal)
                        distance_diff = abs(distance_to_summit - target_distance_pixels)
                        if distance_diff < best_distance_diff:
                            best_distance_diff = distance_diff
                            best_idx = i
                
                self.current_pos = self.trail_coords[best_idx].astype(np.float32)
                
                actual_distance = np.linalg.norm(self.current_pos - self.goal) * self.cell_size
                
                # Adjust step limit based on actual distance
                # Allow ~10 steps per meter of distance as a generous upper bound
                min_steps = max(200, int(actual_distance * 10))  # at least 200 steps
                if self.max_steps and self.max_steps < min_steps:
                    print(f"Warning: Step limit ({self.max_steps}) may be too low for {actual_distance:.0f}m distance, recommend {min_steps}")
                
                if not hasattr(self, '_curriculum_info_printed') or self.curriculum_attempts % 100 == 0:
                    print(f"Curriculum: Starting {actual_distance:.0f}m from summit (target: {self.start_distance_meters:.0f}m)")
                    print(f"Success rate: {success_rate:.2f} ({self.curriculum_successes}/{self.curriculum_attempts})")
                    self._curriculum_info_printed = True
            else:
                # Standard training: start at trailhead
                self.current_pos = self.trail_coords[0].astype(np.float32)
            
            # Store summit info for visualization
            self.summit_elevation = max_elevation
            self.summit_idx = summit_idx
            
            # Print trail info once
            if not hasattr(self, '_summit_info_printed'):
                distance_to_goal = np.linalg.norm(self.current_pos - self.goal) * self.cell_size
                print(f"Trail: Goal at summit (point {summit_idx}/{len(self.trail_coords)-1}, elevation {max_elevation:.1f}m)")
                print(f"Starting distance to goal: {distance_to_goal:.0f}m")
                self._summit_info_printed = True
        else:
            # Fallback: corners
            self.current_pos = np.array([self.map_h // 10, self.map_w // 10], dtype=np.float32)
            self.goal = np.array(
                [self.map_h - self.map_h // 10, self.map_w - self.map_w // 10], dtype=np.float32
            )

        self.velocity[:] = 0.0
        self.energy = self.ENERGY_MAX
        self.health = self.HEALTH_MAX
        self.step_count = 0
        self.trajectory = [self.current_pos.copy()]
        
        # Action diversity tracking
        self.recent_actions = []  # Track last 20 actions for diversity bonus
        self.action_counts = np.zeros(9)  # Count of each action type

        return self._obs(), {}

    def step(self, action: int):
        self.step_count += 1
        prev = self.current_pos.copy()

        # 8-connected movement + rest
        if action == 8:  # rest
            intended = self.current_pos.copy()
            self.energy = min(self.ENERGY_MAX, self.energy + 2.0)
            result = "rest"
        else:
            dirs = {
                0: (-1, 0),
                1: (-1, 1),
                2: (0, 1),
                3: (1, 1),
                4: (1, 0),
                5: (1, -1),
                6: (0, -1),
                7: (-1, -1),
            }
            # Handle both scalar and array actions
            action_idx = int(action.item()) if hasattr(action, 'item') else int(action)
            d = np.array(dirs[action_idx], dtype=np.float32)
            intended = self.current_pos + d

            valid, reason = self._movement_is_valid(self.current_pos, intended)
            if valid:
                nxt, slip_state = self._apply_slip_if_needed(intended)
                self.current_pos = nxt
                # Energy drain (disabled for easier training)
                # ec = self._energy_cost(prev, self.current_pos)
                # self.energy = max(0.0, self.energy - ec)
                result = slip_state
            else:
                # small penalty energy when bumping into blocked cell (disabled)
                # self.energy = max(0.0, self.energy - 0.3)
                result = f"blocked_{reason}"

        self.trajectory.append(self.current_pos.copy())

        # ---- Reward (survival + goal + progress) ----
        reached = self._goal_reached()
        reward = 0.0
        
        # Goal reach bonus (MASSIVELY INCREASED)
        if reached:
            reward += 10000.0  # Increased from 1000.0 - make goal extremely attractive
            # Track curriculum success
            if self.curriculum_learning:
                self.curriculum_successes += 1
            
        # Death penalty (health only, energy system disabled)
        if self.health <= 0.0:
            reward -= 1000.0
            
        # Distance-based progress reward (HEAVILY INCREASED for goal motivation)
        prev_dist = np.linalg.norm(prev - self.goal) * self.cell_size
        cur_dist = np.linalg.norm(self.current_pos - self.goal) * self.cell_size
        progress = prev_dist - cur_dist  # Positive = moving toward goal
        
        # Massive progress rewards to motivate goal-seeking behavior
        reward += progress * 50.0  # 50x reward for every meter closer to goal
        
        # Additional distance-based shaping (closer = better base reward)
        max_distance = 4000.0  # Approximate max distance on map
        distance_bonus = (max_distance - cur_dist) / max_distance * 10.0  # 0-10 bonus based on closeness
        reward += distance_bonus
        
        # Exponential bonus for getting very close to goal
        if cur_dist < 500.0:  # Within 500m of summit
            proximity_bonus = (500.0 - cur_dist) / 500.0 * 100.0  # 0-100 bonus
            reward += proximity_bonus
            
        if cur_dist < 100.0:  # Within 100m of summit  
            final_approach_bonus = (100.0 - cur_dist) / 100.0 * 500.0  # 0-500 bonus
            reward += final_approach_bonus
        
        # Light penalties for unsafe moves (reduced to not discourage exploration)
        if "slip" in result:
            reward -= 2.0  # Reduced from 5.0
        if "blocked" in result:
            reward -= 0.5  # Reduced from 1.0
            
        # Very small time penalty (reduced to not discourage long journeys)
        reward -= 0.001  # Reduced from 0.01
            
        # Alive shaping (encourage staying safe)
        reward += (self.health / self.HEALTH_MAX)

        # ---- Done flags ----
        terminated = reached or (self.health <= 0.0)  # Energy system disabled
        truncated = False
        if self.max_steps is not None and self.step_count >= self.max_steps and not terminated:
            truncated = True  # neutral truncation (we don't change reward)

        info = {
            "result": result,
            "energy": float(self.energy),
            "health": float(self.health),
            "position": self.current_pos.copy(),
            "goal": self.goal.copy(),
            "reached_goal": bool(reached),
            "trajectory": [(float(pos[0]), float(pos[1])) for pos in self.trajectory],
        }

        # Auto-save GPX only when summit goal is reached
        if (terminated or truncated) and self.auto_save_gpx and reached:
            try:
                self.save_gpx(Path("training_output_physics") / "summit_reached")
            except Exception:
                pass

        return self._obs(), float(reward), bool(terminated), bool(truncated), info

    def render(self, mode="rgb_array"):
        vis = self.terrain_rgb.copy()
        r, c = map(int, self.current_pos)
        if 2 <= r < self.map_h - 2 and 2 <= c < self.map_w - 2:
            vis[r - 2 : r + 3, c - 2 : c + 3] = [255, 0, 0]  # agent
        gr, gc = map(int, self.goal)
        if 2 <= gr < self.map_h - 2 and 2 <= gc < self.map_w - 2:
            vis[gr - 2 : gr + 3, gc - 2 : gc + 3] = [0, 0, 255]  # goal
        return vis

    # -------------------- GPX export --------------------
    def _pix_to_map_xy(self, r: float, c: float):
        # rasterio expects (row, col) -> (x, y) via transform.xy, but we can also apply the affine directly
        x, y = rasterio.transform.xy(self.transform, int(r), int(c))
        return float(x), float(y)

    def _map_xy_to_lonlat(self, x: float, y: float):
        lon, lat = self.to_lonlat.transform(x, y)
        return float(lon), float(lat)

    def save_gpx(self, out_dir: Path) -> str:
        """
        Save current episode trajectory to GPX in WGS84 (EPSG:4326).
        Returns the filepath.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        gpx = gpxpy.gpx.GPX()
        tr = gpxpy.gpx.GPXTrack()
        gpx.tracks.append(tr)
        seg = gpxpy.gpx.GPXTrackSegment()
        tr.segments.append(seg)

        for r, c in self.trajectory:
            x, y = self._pix_to_map_xy(r, c)
            lon, lat = self._map_xy_to_lonlat(x, y)
            ele = float(self.elevation_map[int(r), int(c)])
            seg.points.append(gpxpy.gpx.GPXTrackPoint(latitude=lat, longitude=lon, elevation=ele))

        tag = "goal" if self._goal_reached() else "ended"
        path = out_dir / f"episode_{tag}_{np.random.randint(1_000_000_000)}.gpx"
        with open(path, "w") as f:
            f.write(gpx.to_xml())
        return str(path)

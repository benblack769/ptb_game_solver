import json
import os
from .vislibity import LibVisibility

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def env_filename(fname):
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(cur_dir,"environments",fname)

def load_env(json_fname):
    env_values = Struct(**json.load(open(env_filename(json_fname))))

    map_info = Struct(**json.load(open(env_filename(env_values.map_fname))))

    print(map_info.blocker_polygons,map_info.width,map_info.height,flush=True)
    for x in map_info.blocker_polygons:
        x.append(x[0])

    libvis = LibVisibility(map_info.blocker_polygons,map_info.width,map_info.height)

    guard = env_values.guard_locations
    agent = env_values.agent_location
    rewards = map_info.reward_points

    return libvis, env_values, guard, agent, rewards

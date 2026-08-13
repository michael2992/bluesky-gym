from bluesky_gym.envs.competition_env import CompetitionEnv
import bluesky as bs
from core.tools import kwikqdrdist
import bluesky_gym.envs.common.functions as fn
import numpy as np


class prototypeSingleAgentV1Env(CompetitionEnv):
    """A prototype reward function for the competition."""

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)

    def _get_reward(self, ac_id):
        """New reward function based on the initial base function instead with a new component based on distance moved towards target and with the drift component removed, 
        computed from the metric deltas over this env step (action-frequency invariant).
        Reads only committed sim/metric state, never state produced by _get_obs."""
        m = self.metrics[ac_id]
        base = self._reward_baseline[ac_id]
        ac_idx = bs.traf.id2idx(ac_id)

        # retrieve the goal lat and long from the scenario for the single agent
        goal_lat, goal_lon =  self.scenario.agents[0].goal 

        qdr, _ = kwikqdrdist(bs.traf.lat[ac_idx], bs.traf.lon[ac_idx], goal_lat, goal_lon)
        drift_angle = np.deg2rad(fn.bound_angle_positive_negative_180(bs.traf.hdg[ac_idx] - qdr))
        drift = abs(drift_angle) 

        reached = m["waypoint_reached"] - base["waypoint_reached"]
        d_intrusion = m["intrusion_time"] - base["intrusion_time"]
        d_restricted = m["time_in_restricted_area"] - base["time_in_restricted_area"]
        d_outside = m["time_outside_sector"] - base["time_outside_sector"]

        airspeed= bs.traf.tas[ac_idx]

        return (self.reach_reward * reached
                + airspeed * np.cos(drift) /150
                + self.drift_penalty * drift
                + self.intrusion_penalty * d_intrusion
                + self.restricted_area_penalty * d_restricted
                + self.sector_exit_penalty * d_outside)

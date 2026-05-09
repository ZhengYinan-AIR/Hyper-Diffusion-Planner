import numpy as np
from tqdm import tqdm

from nuplan.common.actor_state.state_representation import Point2D

from hdp_nuplan.data_process.roadblock_utils import route_roadblock_correction
from hdp_nuplan.data_process.agent_process import (
    agent_past_process,
    sampled_tracked_objects_to_array_list,
    sampled_static_objects_to_array_list,
    agent_future_process
)
from hdp_nuplan.data_process.map_process import get_neighbor_vector_set_map, map_process
from hdp_nuplan.data_process.ego_process import (
    get_ego_past_array_from_scenario,
    get_ego_future_array_from_scenario,
    calculate_additional_ego_states,
    sampled_past_ego_states_to_array,
)
from hdp_nuplan.data_process.utils import convert_to_model_inputs, convert_absolute_quantities_to_relative


class DataProcessor(object):
    def __init__(self, config):

        self._save_dir = getattr(config, "save_path", None)

        self.past_time_horizon = 2 # [seconds]
        self.num_past_poses = 10 * self.past_time_horizon
        self.future_time_horizon = 8 # [seconds]
        self.num_future_poses = 10 * self.future_time_horizon

        self.num_agents = config.agent_num
        self.num_static = config.static_objects_num
        self.max_ped_bike = 10 # Limit the number of pedestrians and bicycles in the agent.
        self._radius = 100 # [m] query radius scope relative to the current pose.

        self._map_features = ['LANE', 'LEFT_BOUNDARY', 'RIGHT_BOUNDARY', 'ROUTE_LANES'] # name of map features to be extracted.
        self._max_elements = {'LANE': config.lane_num, 'LEFT_BOUNDARY': config.lane_num, 'RIGHT_BOUNDARY': config.lane_num, 'ROUTE_LANES': config.route_num} # maximum number of elements to extract per feature layer.
        self._max_points = {'LANE': config.lane_len, 'LEFT_BOUNDARY': config.lane_len, 'RIGHT_BOUNDARY': config.lane_len, 'ROUTE_LANES': config.route_len} # maximum number of points per feature to extract per feature layer.

    # Use for inference
    def observation_adapter(self, history_buffer, traffic_light_data, map_api, route_roadblock_ids, device='cpu'):

        '''
        ego
        '''
        ego_state_history = list(history_buffer.ego_states)
        ego_state = history_buffer.current_state[0]
        ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
        anchor_ego_state = np.array([ego_state.rear_axle.x, ego_state.rear_axle.y, ego_state.rear_axle.heading], dtype=np.float64)

        if len(ego_state_history) >= 2:
            ego_agent_past = sampled_past_ego_states_to_array(ego_state_history)
            ego_agent_past = convert_absolute_quantities_to_relative(ego_agent_past, anchor_ego_state)
            time_stamps_past = np.array([state.time_point.time_us for state in ego_state_history], dtype=np.int64)
            ego_current_state = calculate_additional_ego_states(ego_agent_past, time_stamps_past)
        else:
            ego_agent_past = None
            ego_current_state = np.array([0., 0., 1., 0., 0., 0., 0., 0., 0., 0.], dtype=np.float32)

        '''
        neighbor
        '''
        observation_buffer = history_buffer.observation_buffer # Past observations including the current
        neighbor_agents_past, neighbor_agents_types = sampled_tracked_objects_to_array_list(observation_buffer)
        static_objects, static_objects_types = sampled_static_objects_to_array_list(observation_buffer[-1])
        _, neighbor_agents_past, _, static_objects = \
            agent_past_process(ego_agent_past, neighbor_agents_past, neighbor_agents_types, self.num_agents, static_objects, static_objects_types, self.num_static, self.max_ped_bike, anchor_ego_state)

        '''
        Map
        '''
        # Simply fixing disconnected routes without pre-searching for reference lines
        route_roadblock_ids = route_roadblock_correction(
            ego_state, map_api, route_roadblock_ids
        )
        coords, traffic_light_data, speed_limit, lane_route = get_neighbor_vector_set_map(
            map_api, self._map_features, ego_coords, self._radius, traffic_light_data
        )
        vector_map = map_process(route_roadblock_ids, anchor_ego_state, coords, traffic_light_data, speed_limit, lane_route, self._map_features,
                                    self._max_elements, self._max_points)


        data = {"neighbor_agents_past": neighbor_agents_past[:, -21:],
                "ego_current_state": ego_current_state,
                "static_objects": static_objects}
        data.update(vector_map)
        data = convert_to_model_inputs(data, device)

        return data

    # Use for data preprocess
    def work(self, scenarios):

        for scenario in tqdm(scenarios):
            map_name = scenario._map_name
            token = scenario.token
            map_api = scenario.map_api

            '''
            ego & agents past
            '''
            ego_state = scenario.initial_ego_state
            ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
            anchor_ego_state = np.array([ego_state.rear_axle.x, ego_state.rear_axle.y, ego_state.rear_axle.heading], dtype=np.float64)
            ego_agent_past, time_stamps_past = get_ego_past_array_from_scenario(scenario, self.num_past_poses, self.past_time_horizon)

            present_tracked_objects = scenario.initial_tracked_objects.tracked_objects
            past_tracked_objects = [
                tracked_objects.tracked_objects
                for tracked_objects in scenario.get_past_tracked_objects(
                    iteration=0, time_horizon=self.past_time_horizon, num_samples=self.num_past_poses
                )
            ]
            sampled_past_observations = past_tracked_objects + [present_tracked_objects]
            neighbor_agents_past, neighbor_agents_types = \
                sampled_tracked_objects_to_array_list(sampled_past_observations)
            static_objects, static_objects_types = sampled_static_objects_to_array_list(present_tracked_objects)

            ego_agent_past, neighbor_agents_past, neighbor_indices, static_objects = \
                agent_past_process(ego_agent_past, neighbor_agents_past, neighbor_agents_types, self.num_agents, static_objects, static_objects_types, self.num_static, self.max_ped_bike, anchor_ego_state)
            '''
            Map
            '''
            route_roadblock_ids = scenario.get_route_roadblock_ids()
            traffic_light_data = list(scenario.get_traffic_light_status_at_iteration(0))

            if route_roadblock_ids != ['']:
                route_roadblock_ids = route_roadblock_correction(
                    ego_state, map_api, route_roadblock_ids
                )

            coords, traffic_light_data, speed_limit, lane_route = get_neighbor_vector_set_map(
                map_api, self._map_features, ego_coords, self._radius, traffic_light_data
            )

            vector_map = map_process(route_roadblock_ids, anchor_ego_state, coords, traffic_light_data, speed_limit, lane_route, self._map_features, self._max_elements, self._max_points)

            '''
            ego & agents future
            '''
            ego_agent_future = get_ego_future_array_from_scenario(scenario, ego_state, self.num_future_poses, self.future_time_horizon)

            present_tracked_objects = scenario.initial_tracked_objects.tracked_objects
            future_tracked_objects = [
                tracked_objects.tracked_objects
                for tracked_objects in scenario.get_future_tracked_objects(
                    iteration=0, time_horizon=self.future_time_horizon, num_samples=self.num_future_poses
                )
            ]

            sampled_future_observations = [present_tracked_objects] + future_tracked_objects
            future_tracked_objects_array_list, _ = sampled_tracked_objects_to_array_list(sampled_future_observations)
            neighbor_agents_future = agent_future_process(anchor_ego_state, future_tracked_objects_array_list, self.num_agents, neighbor_indices)


            '''
            ego current
            '''
            ego_current_state = calculate_additional_ego_states(ego_agent_past, time_stamps_past)

            # gather data
            data = {"map_name": map_name, "token": token, "ego_current_state": ego_current_state, "ego_agent_future": ego_agent_future,
                    "neighbor_agents_past": neighbor_agents_past, "neighbor_agents_future": neighbor_agents_future, "static_objects": static_objects}
            data.update(vector_map)

            self.save_to_disk(self._save_dir, data)

    def save_to_disk(self, dir, data):
        np.savez(f"{dir}/{data['map_name']}_{data['token']}.npz", **data)
        
        
def get_filter_parameters(num_scenarios_per_type=None, limit_total_scenarios=None, shuffle=True, scenario_tokens=None, log_names=None):
    scenario_types = None
    map_names = None
    timestamp_threshold_s = None
    ego_displacement_minimum_m = None
    expand_scenarios = True
    remove_invalid_goals = False
    ego_start_speed_threshold = None
    ego_stop_speed_threshold = None
    speed_noise_tolerance = None

    return (
        scenario_types,
        scenario_tokens,
        log_names,
        map_names,
        num_scenarios_per_type,
        limit_total_scenarios,
        timestamp_threshold_s,
        ego_displacement_minimum_m,
        expand_scenarios,
        remove_invalid_goals,
        shuffle,
        ego_start_speed_threshold,
        ego_stop_speed_threshold,
        speed_noise_tolerance,
    )


def build_history_buffer(processor, scenario):
    ego_states = list(
        scenario.get_ego_past_trajectory(
            iteration=0,
            num_samples=processor.num_past_poses,
            time_horizon=processor.past_time_horizon,
        )
    ) + [scenario.initial_ego_state]

    observation_buffer = list(
        scenario.get_past_tracked_objects(
            iteration=0,
            time_horizon=processor.past_time_horizon,
            num_samples=processor.num_past_poses,
        )
    ) + [scenario.initial_tracked_objects]

    return SimpleNamespace(
        ego_states=ego_states,
        observation_buffer=observation_buffer,
        current_state=(scenario.initial_ego_state, scenario.initial_tracked_objects),
    )


def summarize_npz(npz_path):
    with np.load(npz_path, allow_pickle=False) as data:
        print(f"Saved sample: {npz_path}")
        for key in sorted(data.files):
            value = data[key]
            print(f"{key}: shape={value.shape}, dtype={value.dtype}")


def compare_observation_adapter(processor, scenario, npz_path, atol=1e-5):
    history_buffer = build_history_buffer(processor, scenario)
    traffic_light_data = list(scenario.get_traffic_light_status_at_iteration(0))
    route_roadblock_ids = scenario.get_route_roadblock_ids()

    adapter_output = processor.observation_adapter(
        history_buffer,
        traffic_light_data,
        scenario.map_api,
        route_roadblock_ids,
        device="cpu",
    )
    
    breakpoint()

    keys_to_check = [
        "ego_current_state",
        "neighbor_agents_past",
        "static_objects",
        "lanes",
        "lanes_speed_limit",
        "lanes_has_speed_limit",
        "route_lanes",
        "route_lanes_speed_limit",
        "route_lanes_has_speed_limit",
    ]

    with np.load(npz_path, allow_pickle=False) as expected:
        mismatches = []
        for key in keys_to_check:
            actual = adapter_output[key]
            if isinstance(actual, torch.Tensor):
                actual = actual.detach().cpu().numpy()
            expected_value = expected[key][None, ...]

            if actual.shape != expected_value.shape:
                mismatches.append(f"{key}: shape mismatch actual={actual.shape} expected={expected_value.shape}")
                continue

            if not np.allclose(actual, expected_value, atol=atol, rtol=0.0):
                max_abs = np.max(np.abs(actual - expected_value))
                mismatches.append(f"{key}: value mismatch max_abs={max_abs}")

    if mismatches:
        raise AssertionError("observation_adapter verification failed:\n" + "\n".join(mismatches))

    print("observation_adapter verification passed")
    for key in keys_to_check:
        print(f"verified: {key}")


def main():
    parser = argparse.ArgumentParser(description="Test raw nuPlan -> HDP input conversion and verify observation_adapter")
    parser.add_argument("--data_path", type=str, default="/data/nuplan-v1.1/trainval", help="Path to raw nuPlan logs")
    parser.add_argument("--map_path", type=str, default="/data/nuplan-v1.1/maps", help="Path to nuPlan maps")
    parser.add_argument("--save_path", type=str, default="./tmp/hdp_cache_test", help="Directory to save generated HDP npz files")
    parser.add_argument("--log_names_json", type=str, default=None, help="Optional JSON file containing a list of nuPlan log names")
    parser.add_argument("--scenario_token", type=str, default=None, help="Optional single scenario token to process")
    parser.add_argument("--scenarios_per_type", type=int, default=None, help="Number of scenarios per type")
    parser.add_argument("--total_scenarios", type=int, default=1, help="Limit total scenarios to process")
    parser.add_argument("--shuffle_scenarios", action="store_true", help="Shuffle selected scenarios")
    parser.add_argument("--agent_num", type=int, default=32, help="Number of dynamic agents")
    parser.add_argument("--static_objects_num", type=int, default=5, help="Number of static objects")
    parser.add_argument("--lane_len", type=int, default=20, help="Number of points per lane")
    parser.add_argument("--lane_num", type=int, default=70, help="Number of lanes")
    parser.add_argument("--route_len", type=int, default=20, help="Number of points per route lane")
    parser.add_argument("--route_num", type=int, default=25, help="Number of route lanes")
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    log_names = None
    if args.log_names_json is not None:
        with open(args.log_names_json, "r", encoding="utf-8") as file:
            log_names = json.load(file)

    scenario_tokens = [args.scenario_token] if args.scenario_token else None

    builder = NuPlanScenarioBuilder(
        args.data_path,
        args.map_path,
        sensor_root=None,
        db_files=None,
        map_version="nuplan-maps-v1.0",
    )
    scenario_filter = ScenarioFilter(
        *get_filter_parameters(
            num_scenarios_per_type=args.scenarios_per_type,
            limit_total_scenarios=args.total_scenarios,
            shuffle=args.shuffle_scenarios,
            scenario_tokens=scenario_tokens,
            log_names=log_names,
        )
    )

    worker = SingleMachineParallelExecutor(use_process_pool=True)
    scenarios = builder.get_scenarios(scenario_filter, worker)
    print(f"Total number of scenarios: {len(scenarios)}")
    if not scenarios:
        raise RuntimeError("No scenarios matched the provided filters.")

    processor = DataProcessor(args)
    processor.work(scenarios)

    npz_files = sorted(
        os.path.join(args.save_path, name)
        for name in os.listdir(args.save_path)
        if name.endswith(".npz")
    )
    if not npz_files:
        raise RuntimeError("No HDP npz files were generated.")

    summarize_npz(npz_files[0])
    compare_observation_adapter(processor, scenarios[0], npz_files[0])


if __name__ == "__main__":
    import argparse
    import json
    import os
    import sys
    from types import SimpleNamespace

    import numpy as np
    import torch

    from hdp_nuplan.data_process.data_processor import DataProcessor

    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
    from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
    from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
    main()

import yaml
import time
import networkx as nx
import osmnx as ox
import stlcg


from nuplane.core import XPlaneCore
from nuplane.core import kill_all_servers
from nuplane.utils.transform import get_bearing


from nuplane.env import XPlaneEnv

from experiments.taxiing.control.controllers import PathController
from experiments.taxiing.navigation.path_planner import PathPlanner
from experiments.taxiing.taxi_experiment import TaxiingExperiment
from experiments.STL_taxiing.control.controllers import PathController
from experiments.STL_taxiing.SpeedLimitExperiment import SpeedLimitExperiment
from experiments.LLM_Taxiing.LLMExperiment import LLMExperiment
from experiments.LLM_Taxiing.LLM2STL import LLMCheck
from experiments.STL_taxiing.STL_formulas import StateChecker
from experiments.agents import Hero, AIAircraft


import matplotlib.pyplot as plt

from utils import skip_run

# Run the simulation
config = yaml.load(open('config/xplane_config.yaml'), Loader=yaml.SafeLoader)

with skip_run('skip', 'collect_data') as check, check():
    kill_all_servers()
    core = XPlaneCore(config)   
    """
    Class responsible of handling all the different XPlane functionalities, such as server-client connecting,
    actor spawning and getting the sensors data.
    """

    core.client.sendDREF('sim/weather/cloud_type[0]', 1)
    time.sleep(5)
    core.client.sendDREF('sim/weather/cloud_type[0]', 0)
    time.sleep(5)
    core.client.sendDREF('sim/weather/cloud_type[0]', 1)

    core.client.sendDREF('sim/time/zulu_time_sec', 43200)

    time.sleep(10)
    kill_all_servers()

with skip_run('skip', 'explore_positioning') as check, check():
    # kill_all_servers()
    core = XPlaneCore(config)
    # core.client.pauseSim(False)
    start = 0
    end = 1062
    experiment_config = {'gate': 'B6'}
    position = core.map.taxi_network.gates[experiment_config['gate']]
    position_1 = core.map.node_graph.nodes[1038]
    position_2 = core.map.node_graph.nodes[1037]
    position_3 = core.map.node_graph.nodes[1058]

    bearing = get_bearing(
        position_1['lat'], position_1['lon'], position_2['lat'], position_2['lon']
    )
    core.hero.set_position(position_2['lat'], position_2['lon'], heading=bearing, ac=0)
    core.hero.set_position(position_1['lat'], position_1['lon'], heading=bearing, ac=1)
    core.hero.set_position(position_3['lat'], position_3['lon'], heading=bearing, ac=3)
    pos = position_2['lon']
    for i in range(100):
        pos = pos + 0.00001
        core.hero.set_position(position_2['lat'], pos, heading=bearing, ac=0)

with skip_run('skip', 'explore_route_network') as check, check():
    '''
     This block basically generates a path between a start node and an end node. 
     It takes in the x-y location of the AC, find the nearest node from the generated
     node graph, finds the nearest node from AC location as the start
     Initializes an instance of the class Path Planner and finally finds the path as X, Y coordinates (reversed)

    '''

    core = XPlaneCore(config, debug=True)
    G = core.map.get_node_graph() 

    street_per_node = ox.stats.count_streets_per_node(G) # Number of undirected edges per nodes
    # Returns node a dictionary with nodes ID and undirected edges count per nodes in the graph G

    start = config['hero_config']['spawn_location'][0]  #Initial node where AC is located 
    end = 1034 # Goal node
    node_info = core.map.get_node_info(start)  # Returns a node graph of the world map with info about them (X, Y) and their connections
    start = ox.distance.nearest_nodes(G, X=node_info['x'], Y=node_info['y']) # Find nearest node from the current position (X,Y)
    route = nx.shortest_path(G, start, end, weight='length') # Find the shortest path from the start (nearest node) to the end (Goal node)
    path_planner = PathPlanner(core.map)
    new_lat_lon = path_planner.find_path(start, end, n_splits=1) # Finds path between start and end using path graph 

    fig, ax = ox.plot_graph_route(G, route, show=False, close=False)
    ax.plot(new_lat_lon[:, 1], new_lat_lon[:, 0], marker='o') # X and Y are rerversed in X-plane
    plt.show()

    

with skip_run('skip', 'explore_new_route_network') as check, check():

    '''
     This block basically generates a path between a start node and an end node. 
     It takes in the x-y location of the AC, find the nearest node from the generated
     node graph, finds the nearest node from AC location as the start
     Initializes an instance of the class Path Planner and finally finds the path as X, Y coordinates (reversed)

    '''
    
    core = XPlaneCore(config, debug=True)
    G = core.map.get_node_graph()

    # Get start and end points from 6th spawn point
    spawn_points = yaml.load(open('config/spawn_points.yaml'), Loader=yaml.SafeLoader)
    num = 6
    start = spawn_points[num]['start']
    end = spawn_points[num]['end']

    start_node = ox.distance.nearest_nodes(G, X=start['x'], Y=start['y'])
    print(start_node)
    end_node = ox.distance.nearest_nodes(G, X=end['x'], Y=end['y'])
    route = nx.shortest_path(G, start_node, end_node, weight='length')

    path_planner = PathPlanner(core.map)
    new_lat_lon = path_planner.find_path(start_node, end_node, n_splits=1)

    path_planner.set_route(new_lat_lon) 
    '''
    path_planner.set_route initializes and prepares a route for navigation. 
    It calculates a distance tolerance, sets up a deque to store the route, 
    and identifies the next waypoint to be reached by the navigation entity

    '''
    print(path_planner.distance_tol)

    fig, ax = ox.plot_graph_route(G, route, show=False, close=False)
    ax.plot(new_lat_lon[:, 1], new_lat_lon[:, 0], marker='o')
    plt.show()

with skip_run('skip', 'explore_path_planning') as check, check():
    config['experiment']['type'] = TaxiingExperiment
    config['experiment']['experiment_config'] = yaml.load(
        open('experiments/taxiing/experiment_config.yaml'), Loader=yaml.SafeLoader
    )

    core = XPlaneCore(config)
    G = core.map.get_node_graph()

    spawn_points = yaml.load(open('config/spawn_points.yaml'), Loader=yaml.SafeLoader)
    num = 2
    start = spawn_points[num]['start']
    end = spawn_points[num]['end']

    start_node = ox.distance.nearest_nodes(G, X=start['x'], Y=start['y'])
    end_node = ox.distance.nearest_nodes(G, X=end['x'], Y=end['y'])

    config['hero_config']['spawn_location'] = [start_node]

    # Path planning
    path_planner = PathPlanner(core.map)
    route_lat_lon = path_planner.find_path(start_node, end_node, n_splits=1)
    path_planner.set_route(route_lat_lon)

    hero = Hero(core.client, config['hero_config'])
    core.setup_experiment(
        experiment_config=config['experiment']['experiment_config'], hero=hero
    )
    core.reset(path_planner.get_orientation())

    # Controller
    controller = PathController(agent=hero)
    hero.apply_action([0, 0, 0, 0, 0, 0, 1.5, 1])

    while not path_planner.get_done_status():
        obs = hero.get_observation()

        # Get heading and distance to next waypoint
        if obs is not None:
            heading = path_planner.get_heading(obs)
            distance_to_next_pos = path_planner.get_distance_to_next_pos()
            control = controller.get_control(distance_to_next_pos, heading)

            # Apply control
            hero.apply_action(control)

    hero.apply_action([0, 0, 0, 0, 0, 0, 1.5, 1])

with skip_run('skip', 'taxiing_experiment') as check, check():
    # Setup the environment and experiment
    config['experiment']['type'] = TaxiingExperiment
    config['experiment']['experiment_config'] = yaml.load(
        open('experiments/taxiing/experiment_config.yaml'), Loader=yaml.SafeLoader
    )
    xplane_env = XPlaneEnv(config, debug=True) # Creates an Instance of the class XPlaneEnv
    obs, reward, done, info = xplane_env.reset()

    # Controller or RL
    controller = PathController(agent=xplane_env.experiment.hero) # Creates an instance of the controller

    while not done:
        control = controller.get_control(obs[0], obs[1])

        # Check speed

        # Apply control
        obs, reward, done, info = xplane_env.step(control)


with skip_run('skip', 'speed_limit_experiment') as check, check():
    # Setup the environment and experiment
    config['experiment']['type'] = SpeedLimitExperiment
    config['experiment']['experiment_config'] = yaml.load(
        open('experiments/STL_taxiing/experiment_config.yaml'), Loader=yaml.SafeLoader
    )
    xplane_env = XPlaneEnv(config, debug=True) # Creates an Instance of the class XPlaneEnv
    obs, reward, done, info = xplane_env.reset()

    # Controller or RL
    controller = PathController(agent=xplane_env.experiment.hero) # Creates an instance of the controller
    state_checker = StateChecker(agent=xplane_env.experiment.hero)

    while not done:
        
        control = controller.get_control(obs[0], obs[1])

        # Update speed trace
        state_checker.get_state_trace()

        # Apply control
        obs, reward, done, info = xplane_env.step(control)

    # Check Speed Limit Violation
    state_checker.speed_limit()

with skip_run('run', 'LLM_experiment') as check, check():
    # Setup the environment and experiment
    config['experiment']['type'] = LLMExperiment
    config['experiment']['experiment_config'] = yaml.load(
        open('experiments/LLM_Taxiing/experiment_config.yaml'), Loader=yaml.SafeLoader
    )
    xplane_env = XPlaneEnv(config, debug=True) # Creates an Instance of the class XPlaneEnv
    obs, reward, done, info = xplane_env.reset()

    # Controller or RL
    controller = PathController(agent=xplane_env.experiment.hero) # Creates an instance of the controller
    LLMChecker = LLMCheck(agent = xplane_env.experiment.hero)

    # Define the specific aircraft command and get the code
    user_input = "The speed of the car should always be less than 2.0"

    code = LLMChecker.code_generation(user_input)
    print(code)

    while not done:
        control = controller.get_control(obs[0], obs[1])

        # Check if the command is violated
        
        LLMChecker.get_state()
        LLMChecker.command_evaluation(code)

        # Apply control
        obs, reward, done, info = xplane_env.step(control)
    
    
with skip_run('skip', 'image_data_feed') as check, check():
    config['experiment']['type'] = TaxiingExperiment
    config['experiment']['experiment_config'] = yaml.load(
        open('experiments/taxiing/experiment_config.yaml'), Loader=yaml.SafeLoader
    )

    core = XPlaneCore(config, debug=True)
    data = core.client.getDREF("nuplane/camera_rgb")
    print(data)

    print(len(data))

with skip_run('skip', 'explore_functionality') as check, check():
    # kill_all_servers()
    core = XPlaneCore(config)
    core.map.draw_map(with_labels=True)

    core.client.pauseSim(True)

    core.client.sendDREFs(["sim/operation/fix_all_systems"], [1])

    hero = Hero(core.client)
    ai_carft = AIAircraft(core.client, id=1)

    experiment_config = config['experiment_config']
    # Position
    position = core.map.gates[experiment_config['gate']]

    position = core.map.node_graph.nodes[1063]
    position['heading'] = 270.0
    hero.reset()
    hero.set_position(position)

    position = core.map.node_graph.nodes[1062]
    position['heading'] = 270.0
    ai_carft.reset()
    ai_carft.set_position(position, hero.get_altitude())
    core.client.pauseSim(False)

    for i in range(5):
        ai_carft.apply_action([0, 0, 0, 0.5, False])
        hero.apply_action([0, 0, 0, 0.5, False])
        time.sleep(10)
        hero.apply_action([0, 0, 0, 0, False])
        ai_carft.apply_action([0, 0, 0, 0, False])

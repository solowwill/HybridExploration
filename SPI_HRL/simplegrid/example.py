# NOTE: to run this you must install additional dependencies

if __name__=='__main__':
    import logging, os, sys
    from gym_simplegrid.envs import SimpleGridEnv
    from datetime import datetime as dt
    import gymnasium as gym
    from gymnasium.utils.save_video import save_video

    # Folder name for the simulation
    FOLDER_NAME = dt.now().strftime('%Y-%m-%d %H:%M:%S')
    os.makedirs(f"log/{FOLDER_NAME}")

    # Logger to have feedback on the console and on a file
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)

    logger.info("-------------START-------------")

    options ={
        'start_loc': 0,
        # goal_loc is not specified, so it will be randomly sampled
    }

    obstacle_map = [
                "0000000001000220000000000",
                "0010022001000000001000000",
                "0010000001001110002200001",
                "0010000001000000002200001",
                "0010000000000200000000001",
                "0010001020000200000000000",
                "0010001020000001111110000",
                "0000001000000000000000200",
                "0022001000222000000000200",
                "0000001000000000000000200",
                "0220000000002000011100000",
                "0000000000001100020000000",
                "0000112211002000000020003",
                "0000000000002000100020000",
                "0000200000000000100000000",
                "0000200000010000100022200",
                "0100000000010000000000000",
                "0100022200010000200010000",
                "0100022200010000200010000",
                "0100002100000000000010000",
                "0100000000111000000010000",
                "0000000000200000000010000",
                "0000100022000000000000000",
                "0000100022000211111000000",
                "0000100000000200000000000",
            ]
    
    env = gym.make(
        'SimpleGrid-v0',
        start = (13,0), 
        obstacle_map=obstacle_map, 
        render_mode='human'
    )

    obs, info = env.reset(seed=1, options=options)
    rew = env.unwrapped.reward
    done = env.unwrapped.done

    logger.info("Running action-perception loop...")
    
    with open(f"log/{FOLDER_NAME}/history.csv", 'w') as f:
        f.write(f"step,x,y,reward,done,action\n")
        
        for t in range(20):
            #img = env.render(caption=f"t:{t}, rew:{rew}, pos:{obs}")
            
            action = env.action_space.sample()
            f.write(f"{t},{info['agent_xy'][0]},{info['agent_xy'][1]},{rew},{done},{action}\n")
            
            if done:
                logger.info(f"...agent is done at time step {t}")
                break
            
            obs, rew, done, _, info = env.step(action)
            
    env.close()
    if env.render_mode == 'rgb_array_list':
        frames = env.render()
        save_video(frames, f"log/{FOLDER_NAME}", fps=env.fps)
    logger.info("...done")
    logger.info("-------------END-------------")
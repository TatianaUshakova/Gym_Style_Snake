import pygame
import random
import numpy as np
#import torch
#import torch.nn as nn
#import os

width, height = 640, 480  
blocksize = 20
white = (255, 255,255)
blue = (0,0,155)
black = (0,0,0)
yellow = (0, 255, 0)
red = (255, 0,0)

from collections import namedtuple

Point = namedtuple('point', 'x,y')

import gym
from gym import spaces

#gym style snake:
#gym uses the commands:
#env = gym.make("CartPole-v1", render_mode="human")  
#instead it should be env = GymStyleSnake() or env = GymStyleSnake(aware_length)
#all the other commands match gym commmands


class GymStyleSnake(gym.Env):
    
    metadata = {"render_modes": ["human", "none"]}
    
    def __init__(self, aware_length = 10, disallow_backward = False, render_mode="human", max_iter=10000):
        super().__init__()
        self.render_mode = render_mode
        pygame.init()
        if self.render_mode == "human":
            self.screen = pygame.display.set_mode((width, height))
        else:
            self.screen = None
        self.clock = pygame.time.Clock()
        self.aware_length = aware_length
        self.disallow_backward = disallow_backward

        # Always enforce a time limit, and cap it to 5 minutes at min speed if set too high
        cap_max_steps_5min = 1500
        if max_iter is None or max_iter > cap_max_steps_5min:
            print(f"max_iter set to {max_iter}, capping to {cap_max_steps_5min} steps (~5 minutes at min speed).")
            self.max_iter = cap_max_steps_5min
        else:
            self.max_iter = max_iter

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10+2*self.aware_length,), dtype=np.float32
        )
        if self.disallow_backward:
            self.action_space = spaces.Discrete(3)  # relative to snake: left, straight, right
        else:
            self.action_space = spaces.Discrete(4)  # absolute directions
        self.reset() 
           
    def reset(self, *, seed=None, options=None): 
        #if seed is not None: np.random.seed(seed)
        self.head = Point(width//2, height//2)
        self.snake = [self.head, Point(self.head.x - blocksize, self.head.y), Point(self.head.x - 2*blocksize, self.head.y)]
        self.__place_food()
        self.action = Point(1,0)
        self.score = 0
        self.speed = 5
        self.terminated, self.truncated = False, False
        self.frame_iteration = 0
        state, info = self._make_state(), {}
        return state, info

    def __place_food(self):
        while True:
            self.food = Point(random.randint(0, width//blocksize -1)*blocksize, random.randint(0, height//blocksize -1)*blocksize)
            if self.food not in self.snake:
                break
                #not a recursion since then might run to the internal python recursion limit - though a problem only if snake is so long that covers most of the screen

    def __is_collision(self):
        return (self.head.x < 0 or self.head.x>= width or self.head.y<0 or self.head.y>= height or self.head in self.snake[1:]) #self.snake[0] =self.head by default
        
    def render(self, mode='human'): 
        #to do? add mode='rgb_array'

        if self.render_mode != 'human':
            return

        self.screen.fill(white)
        pygame.draw.rect(self.screen, blue, pygame.Rect(0, 0, width, height))
        
        for point in self.snake:
            pygame.draw.rect(self.screen, black, pygame.Rect(point.x, point.y, blocksize, blocksize))
        pygame.draw.rect(self.screen, yellow, pygame.Rect(self.food.x, self.food.y, blocksize, blocksize))
        
        if self.terminated or self.truncated:
            #print gameover and scores
            font = pygame.font.Font(None, 50)
            gameover_text = font.render(f"Game Over, Your Score: {self.score}", True, red)
            gameover_text_rect = gameover_text.get_rect(center = (width//2, height//2-30))
            
            font1 = pygame.font.Font(None, 30)
            newgame_text = font1.render(f"Play again? (y/n):", True, red) #not really needed - it was from human mode version
            newgame_text_rect = newgame_text.get_rect(center = (width//2, height//2+30))
            
            self.screen.blit(gameover_text, gameover_text_rect)
            self.screen.blit(newgame_text, newgame_text_rect)
        
        pygame.display.flip()#update the screen

    def _point_to_idx(self, point):
        # Convert Point action to index (right=0, up=1, left=2, down=3)
        if point.x == 1: return 0    # right
        if point.y == -1: return 1   # up
        if point.x == -1: return 2   # left
        if point.y == 1: return 3    # down
        raise ValueError(f"Invalid point action: {point}")

    def _idx_to_point(self, idx):
        # Convert index to Point action
        actions = [Point(1,0), Point(0,-1), Point(-1,0), Point(0,1)]
        return actions[idx]
    
    def _snake_to_game_frame(self, relative_action):
        """
        Converts a relative action (0: left, 1: straight, 2: right)
        to an absolute direction in the game frame.
        """
        directions = [Point(1,0), Point(0,-1), Point(-1,0), Point(0,1)]  # right, up, left, down
        current_idx = self._point_to_idx(self.action)
        if relative_action == 0:  # left turn
            new_idx = (current_idx - 1) % 4
        elif relative_action == 1:  # straight
            new_idx = current_idx
        elif relative_action == 2:  # right turn
            new_idx = (current_idx + 1) % 4
        else:
            raise ValueError("Invalid relative action")
        return directions[new_idx]

    def _make_state(self):
        '''create state to be passed to nn consisting of 
         - distance from the nearest wall;
         - distance from the food;
         - direction of motion;
         - position of aware_lenght of the body relative to the head
         '''
        state = []

        # 1. Distance to wall (normalized) -- change to the nearest only?
        left_dist   = self.head.x / width
        right_dist  = (width - self.head.x) / width
        top_dist    = self.head.y / height
        bottom_dist = (height - self.head.y) / height
        state.extend([left_dist, right_dist, top_dist, bottom_dist])

        # 2. Direction to food (normalized)
        food_dx = (self.food.x - self.head.x) / width
        food_dy = (self.food.y - self.head.y) / height
        state.extend([food_dx, food_dy])

        # 3. Direction of motion 
        direction = [0, 0, 0, 0]  # right, up, left, down 
        if self.action == Point(1, 0):   direction[0] = 1
        elif self.action == Point(0, -1): direction[1] = 1
        elif self.action == Point(-1, 0): direction[2] = 1
        elif self.action == Point(0, 1):  direction[3] = 1
        state.extend(direction)

        # 4. Body positions relative to head (for aware_length segments)
        # Each segment: (dx, dy) normalized
        for i in range(1, self.aware_length + 1):
            if i < len(self.snake):
                dx = (self.snake[i].x - self.head.x) / width
                dy = (self.snake[i].y - self.head.y) / height
            else:
                dx, dy = 2.0, 2.0  # value outside the possible range to pad if snake is short
            state.extend([dx, dy])

        return np.array(state, dtype=np.float32)

    def step(self, agents_action):
        if self.disallow_backward: #not allowing snake to go backwards that kills snake, -> nn doesnt have to learn it
            # Use relative action and convert to absolute direction
            action = self._snake_to_game_frame(agents_action)
        else:
            # Use absolute direction directly
            action = self._idx_to_point(agents_action)
        self.action = action #play step read this - to do - change to be passed as arg
        reward = self.__play_step()
        next_state = self._make_state()
        return next_state, reward, self.terminated, self.truncated, {} #last one is info 

    def __play_step(self, render = True): #returns reward #to do - prob rescale reward?
        '''do game step and returns the reward for the step'''
        self.frame_iteration += 1
        rl_reward = 0# -0.01  -  was a time penalty
                    #----
                    #there is an interesting story about this reward: it was added as snake, not knowing its full body position, 
                    #learned to move in a zigzag way to avoid hitting the body. Time penalty intended to prevent it, enforcing snake
                    #to move straight. However, when the time penalty was chosen too big, snake at some point learned to go to the 
                    #nearest wall and hit it, reducing accumulated negative reward
        
        current_distance = ((self.head.x - self.food.x)**2 + (self.head.y - self.food.y)**2)**0.5
        
        # Then move the snake
        self.head = Point(self.head.x+self.action.x*blocksize, self.head.y+self.action.y*blocksize)
        self.snake.insert(0, self.head)

        # if exiding max steps
        if self.frame_iteration > self.max_iter:
            self.truncated = True
            return rl_reward

        if self.__is_collision():
            rl_reward = -20 
            self.terminated = True
            return rl_reward
         
        elif self.head == self.food:
            self.score += 1 
            self.speed = min(15, self.speed + 1)
            rl_reward = 100
            self.__place_food()
        
        else: #nothing happen - continue going
            new_distance = ((self.head.x - self.food.x)**2 + (self.head.y - self.food.y)**2)**0.5
            
            distance_change = current_distance - new_distance
            rl_reward += distance_change/blocksize   # will be at max +-1 so good comparatiely to other distances we have
                                                    #this reward is present in general to better show to snake to go towards the food
            
            self.snake.pop()

        if render:
            self.render()
        self.clock.tick(self.speed)
        return rl_reward
    
    def close(self):
        pygame.quit()


def main():
    from stable_baselines3 import PPO

    env = GymStyleSnake(aware_length=10)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)

    # Wait dor user to react before showing the performance game
    input("Training finished! Ready to see the performance of the trained agent? (Press Enter to continue)")

    # Test the trained agent
    obs, _ = env.reset()
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated: #=done, restart
            obs, _ = env.reset()

    env.close()

if __name__ == "__main__":
    main()

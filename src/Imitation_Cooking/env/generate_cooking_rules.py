import sys
import os
sys.path.append(os.getcwd())
from src.Imitation_Cooking.env.env_cooking import Env_Cooking

env = Env_Cooking()
env.reset()

# for i in range(10):
#     print (env.step(env.action_space.sample()))

env.extract_rules('./src/Imitation_Cooking/data/cooking_rules/')


import numpy as np
import pandas as pd

# Load the data
bws_clip = np.load('./bws_clip.npy')
bws_gpt4 = np.load('./bws_gpt4.npy')
bws_gpt4_color = np.load('./bws_gpt4_color.npy')
bws_gpt4_color2 = np.load('./bws_gpt4_color2.npy')
bws_gpt4_only_prompt = np.load('./bws_gpt4_only_prompt.npy')

sim_clip = np.load('./sim_clip.npy')
sim_gpt4 = np.load('./sim_gpt4.npy')
sim_gpt4_color = np.load('./sim_gpt4_color.npy')
sim_gpt4_color2 = np.load('./sim_gpt4_color2.npy')
sim_gpt4_only_prompt = np.load('./sim_gpt4_only_prompt.npy')

# Correct the missing 12th image in bws_gpt4 and sim_gpt4
bws_gpt4 = np.insert(bws_gpt4, 11, bws_clip[11])
sim_gpt4 = np.insert(sim_gpt4, 11, sim_clip[11])

# Combine all arrays into a dataframe
data = {
    "bws_clip": bws_clip,
    "bws_gpt4": bws_gpt4,
    "bws_gpt4_color": bws_gpt4_color,
    "bws_gpt4_color2": bws_gpt4_color2,
    "bws_gpt4_only_prompt": bws_gpt4_only_prompt,
    "sim_clip": sim_clip,
    "sim_gpt4": sim_gpt4,
    "sim_gpt4_color": sim_gpt4_color,
    "sim_gpt4_color2": sim_gpt4_color2,
    "sim_gpt4_only_prompt": sim_gpt4_only_prompt
}

df = pd.DataFrame(data)
df.head()

df.to_csv('results.csv')

import random
import pandas as pd
import matplotlib.pyplot as plt
from satoriengine.veda.pipelines.meta.o import optimal_circle_placement,optimal_circle_placement_gravity, optimal_circle_placement_antigravity

from scipy.stats import norm


def get_radius(df, a, b, score, sigma=5.0, R_max=1.0):
    """
    Compute a radius for placing a new point based on its score using a normal distribution mapping.

    Conceptual approach:
    - We assume that the best score is 0 and that as score increases, we move further out.
    - We convert the score into a "z-score" by dividing by sigma.
    - We use the standard normal CDF (Φ) to map this z-score to a probability value.
    - At score=0, Φ(0)=0.5, and we shift and scale so that radius=0 at score=0.
    - As score increases, Φ(score/σ) moves from 0.5 towards 1.
    - We define radius = R_max * 2 * (Φ(score/σ) - 0.5).
      This ensures radius=0 at score=0 and radius asymptotically approaches R_max as score grows large.

    Parameters:
    - df: DataFrame containing data (not currently used in the calculation here, but included for compatibility).
    - a, b: Values of A and B for the new point (not used directly, but kept in signature if needed for future logic).
    - score: The score for the point. Lower is better; zero is best.
    - sigma: Controls how quickly radius grows with increasing score.
    - R_max: The maximum radius as score grows very large.

    Returns:
    - radius: A radius value for placing the point.
    """
    z = score / sigma
    p = norm.cdf(z)               # Maps z to [0,1], with score=0 -> p=0.5
    radius = R_max * 2 * (p - 0.5) # Shift and scale so radius=0 at score=0, grows to R_max as score -> ∞
    return radius

def generate_small_dataset(data=None, a=None, b=None, score=None):
    data = pd.DataFrame({
        # encode points
        'x': [0,],
        'y': [0,],
        # features
        'a': [0,],
        'b': [0,],
        # score (radius)
        'score': [0,],
        'radius': [0,],
    }) if data is None else data
    data['radius'] = data['score']
    # Given A and B values for the new point
    a = a or 100
    b = b or 100
    # Circle radius
    radius = score or 1
    return data, a, b, radius

def generate_sample_dataset():
    data = pd.DataFrame({
        'x': [0, 1, 2, -1, 1.5,  -.9, -.8, ],
        'y': [0, 1, -1, -2, .5,  -1.9, -1.8,],
        'a': [15, 10, 20, 30, 12, 31,  32,  ],
        'b': [20, 15, 25, 35, 18, 34,  33,  ],
        's': [0, .25, .33, .15, .16, .28, .12],
    })
    # Given A and B values for the new point
    a = 33
    b = 30
    # Circle radius
    radius = 1
    return data, a,b,radius

def visualizeToPng(df):
    # Plot the 'value' column against the 'date_time' index
    plt.figure(figsize=(12, 6))
    plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=df.iloc[:, 2], cmap='viridis')
    #plt.plot(df['y'], df['x'], label=str(df['a'])+str(df['b']), linewidth=1)
    for i in range(len(df)):
        plt.text(df.iloc[i, 0], df.iloc[i, 1], str(df.iloc[i, 2])+','+str(df.iloc[i, 3]), fontsize=12, ha='center', va='center')
    # Formatting the plot
    plt.title("Time Series Visualization", fontsize=16)
    plt.xlabel("Date Time", fontsize=14)
    plt.ylabel("Value", fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    # Save the plot to a file
    output_path = './visualize.png'
    plt.savefig(output_path)
    # Close the plot to free memory
    plt.close()


# Generate sample data
def run():
    data, a, b, radius = generate_small_dataset()
    visualizeToPng(data)
    x, y = optimal_circle_placement_gravity(data, a, b, radius)
    input(f'point ({x, y})')
    ## combine data and a and be
    df = pd.concat([data, pd.DataFrame({'a':[a], 'b':[b], 'x':[x], 'y':[y]})])
    visualizeToPng(df)

def run_loop():
    df = None
    a = None
    b = None
    radius = None
    while True:
        data, a, b, radius = generate_small_dataset(df, a, b, radius)
        visualizeToPng(data)
        x, y = optimal_circle_placement_gravity(data, a, b, radius)
        input(f'point {x, y}')
        ## combine data and a and be
        df = pd.concat([data, pd.DataFrame({'x':[x], 'y':[y], 'a':[a], 'b':[b], 'score':[radius]})], ignore_index=True)
        # for row in dataframe clear x and y values, and recalculate them
        new_df = pd.DataFrame({'x':[], 'y':[], 'a':[], 'b':[], 'score':[] })
        for ix, row in df.iterrows():
            df_excluded = df.drop(ix)
            a = row['a']
            b = row['b']
            score = row['score']
            x, y = optimal_circle_placement_gravity(df_excluded, a, b, score)
            new_df = pd.concat([new_df, pd.DataFrame({'x': [x], 'y': [y], 'a': [a], 'b': [b], 'score': [score]})], ignore_index=True)
            visualizeToPng(new_df)
            input(ix)
        df = new_df
        visualizeToPng(df)
        a = random.randint(0,200)
        b = random.randint(0,200)
        radius = random.random()

run_loop()

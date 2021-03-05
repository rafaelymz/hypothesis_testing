import matplotlib.pyplot as plt
import seaborn as sns
from hyppo.tools import SIMULATIONS
from hyppo.independence import MGC, Dcorr
import time

# make plots look pretty
sns.set(color_codes=True, style="white", context="talk", font_scale=2)
PALETTE = sns.color_palette("Greys", n_colors=9)
sns.set_palette(PALETTE[2::2])

linear = list(SIMULATIONS.values())[0]
spiral =list(SIMULATIONS.values())[7]
indep = list(SIMULATIONS.values())[-1]
sim_data = [linear, spiral, indep]
test_data = {}

# constants
NOISY = 100  # sample size of noisy simulation
NO_NOISE = 1000  # sample size of noise-free simulation

# simulation titles
SIM_TITLES = [
    "Linear",
    "Spiral",
    "Independence",
]


def plot_sims():
    """Plot simulations"""
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(16, 10))

    plt.suptitle("Independence Simulations", y=0.93, va="baseline")

    for i, col in enumerate(ax):
        print(col)
        count = i
        sim_title = SIM_TITLES[count]
        sim = sim_data[count]

        # the multiplicative noise and independence simulation don't have a noise
        # parameter
        if sim_title in ["Noise", "Independence"]:
            x, y = sim(NO_NOISE, 1)
            x_no_noise, y_no_noise = x, y
            test_data[sim_title] = (x, y, x, y)
        else:
            x, y = sim(NOISY, 1, noise=True)
            x_no_noise, y_no_noise = sim(NO_NOISE, 1)
            test_data[sim_title] = (x, y, x_no_noise, y_no_noise)

        # plot the noise and noise-free sims
        col.scatter(x, y, label="Noisy")
        col.scatter(x_no_noise, y_no_noise, label="No Noise")

        # make the plot look pretty
        col.set_title("{}".format(sim_title))
        col.set_xticks([])
        col.set_yticks([])
        if count == 16:
            col.set_ylim([-1, 1])
        sns.despine(left=True, bottom=True, right=True)

    leg = plt.legend(
        bbox_to_anchor=(0.5, 0.1),
        bbox_transform=plt.gcf().transFigure,
        ncol=5,
        loc="upper center",
    )
    leg.get_frame().set_linewidth(0.0)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(5.0)
    plt.subplots_adjust(hspace=0.75)


# run the created function for the simultions
plot_sims()


start = time.time()
for key, v in test_data.items():
    x, y, x_noise, y_noise = v
    stat, pvalue = Dcorr().test(x, y, workers=-1)
    print(f'The pvalue for {key} no noise data is {pvalue}')
    if key != 'Independence':
        stat, pvalue = Dcorr().test(x_noise, y_noise, workers=-1)
        print(f'The pvalue for {key} noisy data is {pvalue}')
print(f'Total processing time is {time.time()-start}')

start = time.time()
for key, v in test_data.items():
    x, y, x_noise, y_noise = v
    stat, pvalue, mgc_dict = MGC().test(x, y, workers=-1)
    print(f'The pvalue for {key} no noise data is {pvalue}. The optimal scale is {mgc_dict["opt_scale"]}')
    if key != 'Independence':
        stat, pvalue, mgc_dict = MGC().test(x_noise, y_noise, workers=-1)
        print(f'The pvalue for {key} noisy data is {pvalue}. The optimal scale is {mgc_dict["opt_scale"]}')
print(f'Total processing time is {time.time()-start}')



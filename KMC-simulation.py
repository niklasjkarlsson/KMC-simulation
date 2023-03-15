import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
from IPython.display import HTML#, image
from threading import Thread
from queue import Queue
from matplotlib import rc



# List of parameters:

step_height = 1                     # Height difference of terraces for the periodic boundary condition
T = 1750                            # Temperature
k = 1.380649*(10**(-23))            # Boltzmann constant
N = 100                             # Size of lattice
v = 10**(13)                        # Adatoms' vibration frequency
na =  0 #N**2 // 100                    # Initial number of adatoms
ac = na / (N**2)                    # Initial adatom concentration
E_S = 1.3*1.60217663*(10**(-19))    # Substrates potential energy
E_N = 1.0*1.60217663*(10**(-19))    # Interaction energy
ads_rate = 20                       # Rate of adsorption
total_ads_rate = ads_rate * N**2    # Total adsorption rate
steps = 1000000                     # Number of simulation steps
animationskip = 3000                # simulation steps skipped between frames in animation
timeskip = 0.00005                 # time skipped between frames in animation
adslist = np.array([50, 100, 150, 200])

# a class to efficiently store adatom coordinates for each event (diffusion or desorption) and amount of neighbouring adatoms
class RateList():
    def __init__(self, rate, event_type):
        self.adatom_to_position = {}
        self.adatoms = []
        self.rate = rate
        self.event_type = event_type

    def add_adatom(self, item):
        if item in self.adatom_to_position:
            return
        self.adatoms.append(item)
        self.adatom_to_position[item] = len(self.adatoms)-1

    def remove_adatom(self, item):
        if item not in self.adatom_to_position:
            return
        position = self.adatom_to_position.pop(item)
        last_item = self.adatoms.pop()
        if position != len(self.adatoms):
            self.adatoms[position] = last_item
            self.adatom_to_position[last_item] = position

    def choose_random_adatom(self):
        return random.choice(self.adatoms)

    def count_adatoms(self):
        return len(self.adatoms)

    def get_total_rate(self):
        return self.rate * self.count_adatoms()

    def get_event_type(self):
        return self.event_type

# creation of lattice
def create_lattice():
    lat = np.array([0] * (N**2 - na) + [1] * na)
    np.random.shuffle(lat)
    lat = np.reshape(lat, (N, N))
    return lat

# function to count neighbouring adatoms
def count_neighbouring_adatoms(i, j, lattice):
    n = 0
    # lower
    if i > 0 and lattice[i, j] <= lattice[i-1, j]:
        n += 1
    elif i == 0 and lattice[i, j] <= lattice[N-1, j] + step_height:
        n += 1
    # upper
    if i < N-1 and lattice[i, j] <= lattice[i+1, j]:
        n += 1
    elif i ==  N-1 and lattice[i, j] <= lattice[0, j] - step_height:
        n += 1
    # left
    if j > 0 and lattice[i, j] <= lattice[i, j-1]:
        n += 1
    elif j == 0 and lattice[i, j] <= lattice[i, N-1]:
        n += 1
    # right
    if j < N-1 and lattice[i, j] <= lattice[i, j+1]:
        n += 1
    elif j ==  N-1 and lattice[i, j] <= lattice[i, 0]:
        n += 1
    return n

# finding diffusion rates
def get_diff_rate(n):
    return v*np.exp(-1/(k*T)*(E_S+n*E_N)) 

# finding desorption rates
def get_des_rate(n):
    return 0 #v*np.exp(-1/(k*T)*(E_S+n*E_N))  # v*np.exp(-1/(k*T)*(n*E_N))


def move_coordinate(x, k):
    x += k
    if x < 0:
        x += N
    if x > N-1:
        x -= N
    return x

# Function to remove adatom from a RateList and placing it into another
def assign_adatom(lattice, i, j, des_rates, diff_rates):
    for r in des_rates:
        r.remove_adatom((i,j))
    for r in diff_rates:
        r.remove_adatom((i,j))

    if lattice[i,j] == 0:
        return

    n = count_neighbouring_adatoms(i, j, lattice)

    diff_rates[n].add_adatom((i,j))
    des_rates[n].add_adatom((i,j))

# function to update adatoms to right RateLists after manipulating the lattice
def update_grid(lattice, i, j, des_rates, diff_rates):
    size = 5
    i = move_coordinate(i,-size // 2)
    j = move_coordinate(j, -size // 2)
    for l in range(5):
        for m in range(5):
            assign_adatom(
                lattice, move_coordinate(i, l),
                move_coordinate(j,m),
                des_rates, diff_rates)



def move_adatom(i, j, lattice):
        # remove adatom from initial place
        lattice[i, j] -= 1
        # draw direction of movement
        d = np.random.randint(4)
        # move left
        if d == 0:
            if i > 0:
                i -= 1
            else:
                i = N-1
        # move down
        elif d == 1:
            if j < N-1:
                j += 1
            else:
                j = 0
        # move right
        elif d == 2:
            if i < N-1:
                i += 1
            else:
                i = 0
        # move up
        elif d == 3:
            if j > 0:
                j -= 1
            else:
                j = N-1
        lattice[i,j] += 1
        return i,j

def do_animation(simulator, skip):
    fps = 25
    fig = plt.figure( figsize=(8,8) )
    ax = plt.axes()
    a = simulator.lattice
    im = plt.imshow(a, interpolation='none', aspect='auto', vmin=0, vmax=4, cmap=cm.YlOrRd)

    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='black')
    def animate_func(i):
        (frame,t) = simulator.get_frame()
        time_text.set_text(f'Coverage: {100 * np.sum(frame)/(N**2)} %')
        im.set_array(frame)
        if np.sum(frame) in adslist * (N**2)/100:
            plt.savefig(f'KMC_cov{100 * np.sum(frame)/N**2}%_T{T}.png', format='png')

    anim = animation.FuncAnimation(
                                fig, 
                                animate_func, 
                                frames = 10000,
                                interval = 1000 / fps, # in ms
                                )
    fig.suptitle(f'KMC-animation ({T} K)')
    plt.show()
    anim.save('animation.gif', writer='imagemagick', fps=60)

        
class Simulator(Thread):
    def __init__(self):
        super().__init__()
        self.q = Queue(maxsize=5)
        # set initial time
        self.t = 0
        self.intval = 0
        self.adsorptions = 0
        # run simulation:

        # First create a lattice of size NxN with na adatoms at random places

        self.lattice = create_lattice()


        # Then create the diffusion and desorption classes
        diff0 = RateList(get_diff_rate(0), "diffusion")
        diff1 = RateList(get_diff_rate(1), "diffusion")
        diff2 = RateList(get_diff_rate(2), "diffusion")
        diff3 = RateList(get_diff_rate(3), "diffusion")
        diff4 = RateList(get_diff_rate(4), "diffusion")
        self.diff_rates = [diff0, diff1, diff2, diff3, diff4]

        des0 = RateList(get_des_rate(0), "desorption")
        des1 = RateList(get_des_rate(1), "desorption")
        des2 = RateList(get_des_rate(2), "desorption")
        des3 = RateList(get_des_rate(3), "desorption")
        des4 = RateList(get_des_rate(4), "desorption")
        self.des_rates = [des0, des1, des2, des3, des4]

        self.ads = RateList(ads_rate, "adsorption")

        # add each adatom to right diffusion and desorption rate classes
        for i in range(N):
            for j in range(N):
                assign_adatom(self.lattice, i, j, self.des_rates, self.diff_rates)
                self.ads.add_adatom((i,j))
        

        # list of all event rates
        self.event_rates = [self.ads] + self.diff_rates + self.des_rates

    def simulationstep(self):
        # calculate the total rate of all events
        C = sum([i.get_total_rate() for i in self.event_rates])


        # get random number in range [0,C)
        R = random.uniform(0,C)

        # choose event

        for r in self.event_rates:
            #            print(R, r.get_total_rate())
            R -= r.get_total_rate()
            if R <= 0:
                event = r 
                break

        # choose random adatom and execute the process

        adatom = event.choose_random_adatom()
        self.t += 1/C
        self.intval += 1/C

        if event.get_event_type() == "adsorption":
            self.lattice[adatom[0], adatom[1]] += 1
            self.adsorptions += 1
            
        elif event.get_event_type() == "desorption":
            self.lattice[adatom[0], adatom[1]] -= 1
            print("des")

        elif event.get_event_type() == "diffusion":
            o,p = move_adatom(adatom[0], adatom[1], self.lattice)
            
            

        update_grid(self.lattice, adatom[0], adatom[1], self.des_rates, self.diff_rates)

    def skip(self):
        while self.adsorptions < 5:
            self.simulationstep()
        self.adsorptions = 0

    def run(self):
        while True:
            self.skip()
            self.q.put((self.lattice.copy(), self.t))

    def get_frame(self):
        return self.q.get()



def main():

    global simulator
    simulator = Simulator()
    simulator.start()
    do_animation(simulator, animationskip)





main()

import numpy as np
import sys, os, shutil
from pathlib import Path

# path setup
for p in ['../..', '../../core/exadis/python']:
    abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), p))
    print('Adding ' + abs_path + ' to path')
    if abs_path not in sys.path: sys.path.insert(0, abs_path)

# imports from opendis
import pyexadis
from pyexadis_base import ExaDisNet, DisNetManager
from pyexadis_base import CalForce
from pyexadis_utils import insert_prismatic_loop

def main():

    # Create the output directory
    write_dir = os.path.join(os.path.dirname(__file__), 'output_climb')
    
    if os.path.exists(write_dir): 
        print(f'Output directory {write_dir} already exists.')
        # Prompt the user for confirmation
        response = input('Do you want to overwrite it? (y/n): ')
        
        if response.strip().lower() in ['y', 'yes']:
            print('Overwriting directory...\n')
            shutil.rmtree(write_dir)
            os.makedirs(write_dir)
        else:
            print('Operation cancelled. Exiting script.')
            return None
    else:
        os.makedirs(write_dir)

    # Intitialize pyexadis instance
    pyexadis.initialize()

    # Create the state
    state = {
        'crstal'    :   'bcc',
        'burgmag'   :   1.0,
        'mu'        :   1.0,    # Shear Modulus
        'nu'        :   0.3,    # Poisson Ratio
        'a'         :   1.0,    # Lattice parameter
        'maxseg'    :   50.0,   # maximum distance between nodes
        'minseg'    :   10.0,   # minimum distance between nodes
        'rtol'      :   5.0,    # Re-meshing/Tolerance radius
        'rann'      :   5.0,    # Annihalation radius
        'nextdt'    :   1.0,    # Timestep
        'maxdt'     :   10.0,   # Maximum amount of time elapsed during simulation
    }

    # NOTE: the 'nextdt' and 'maxdt' are placeholders. We keep them 
    # here because they are required parameters to run the simulation 
    # through exadis, but we need to manually specify a dt and max_dt 
    # so that we have fine control over it.
    dt = 1.0
    max_dt = 200.0

    # Create the geometry
    # Here we are creating a dislocation loop in the middle of the box 
    # (Length is n*R) where R is radius of the loop and n is a integer
    R       = 200.0
    Lbox    = 10*R

    cell    = pyexadis.Cell(Lbox)
    center  = np.array([Lbox/2, Lbox/2, Lbox/2])
    b       = np.array([1., 1., 1.]) / np.sqrt(3.)

    nodes, segs = [], []

    nodes, segs = [], []
    nodes, segs = insert_prismatic_loop('bcc', cell, nodes, segs,
                                        burg=b, radius=R,
                                        center=center, maxseg=50.0)

    G = ExaDisNet(cell, nodes, segs)

    # This initialises a 'disnetmanager' object which allows us to do things with our 
    # dislocation network (such as apply a force and watch how it evolves) but it requires
    # the original network as a input
    net = DisNetManager(G)

    # Inspect crystallographic data
    data = net.export_data()

    # Box vectors
    h = np.array(data['cell']['h'])
    print(f"\nBox matrix (rows = box vectors):\n{h}")

    # Burgers vectors
    burgers = data['segs']['burgers']
    unique_b = np.unique(np.round(burgers, 4), axis=0)
    print(f"\nUnique Burgers vectors (normalised, Cartesian):")
    for b in unique_b:
        print(f"  {b}  |b| = {np.linalg.norm(b):.4f}")

    # Slip plane normals  
    planes = data['segs']['planes']
    unique_p = np.unique(np.round(planes, 4), axis=0)
    print(f"\nSlip plane normals:")
    for p in unique_p:
        print(f"  {p}")

    pos = data['nodes']['positions'] # Node positions

    # Define mobility law for dislocations
    # NOTE: Here we are concerned with climb so we 
    # are not too worried about the glide mobility law
    calforce = CalForce(force_mode='LINE_TENSION_MODEL', state=state)
    
    # Define a function to return the radius of the dislocation loop at each timestep
    def loop_radius(net):
        G = net.get_disnet(ExaDisNet)           # Get the disnet object
        pos = G.get_nodes_data()['positions']   # Get positions of nodes in the disnet object
        center = np.mean(pos, axis=0)           # Position of center of the simulation box
        rel_pos = pos - center                  # Distance of each node from the center of the sim box
        dists = np.linalg.norm(rel_pos, axis=1) # Get the distance of each node from the center
        radius = np.mean(dists)                 # Get the average distance of each node from the center
                                                # (This is also the radius)
        return radius
    
    initial_radius = loop_radius(net)
    print(f'Initial radius: {loop_radius(net):.4f} b')

    # Collect the initial configuration
    G0 = net.get_disnet(ExaDisNet)
    G0.write_data(os.path.join(write_dir, 'initial.config'))

    # Finalize pyexadis instance
    pyexadis.finalize()

    return None

if __name__ == '__main__':
    main()
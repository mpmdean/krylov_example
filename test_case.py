import numpy as np
import matplotlib.pyplot as plt
from wrappers import ed_wrapper, rixs_wrapper, NiPS3_ed, NiPS3_rixs
from datetime import datetime
import time
from mpi4py import MPI
comm = MPI.COMM_WORLD

rank = comm.rank

# four cores on mark's m2 macbook via docker. 
#nd = 5 ED 21 s RIXS 72
#nd = 4 ED 44 s RIXS 223
#nd = 3 ED 80 s RIXS 600 s
nd = 3
#nd = 8
if rank == 1:
    print(f"Start nd={nd}")
NiPS3_ed['nd'] = nd
NiPS3_rixs['v_noccu'] = NiPS3_ed['nd'] + NiPS3_ed['nbath']*NiPS3_ed['norb_d']
NiPS3_rixs['ominc'] = np.arange(850.5, 856, 2)
NiPS3_rixs['pol_type'] = [('linear', 0, 'linear', 0)]


def log(message, start_time=time.time()):
    """
    Print message with:
      - Current wall-clock time
      - Elapsed time since script start
    Only prints on MPI rank 1.
    """
    if rank == 1:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elapsed = time.time() - start_time
        print(f"### {message}: [{now}] (+{elapsed:8.3f}s)")
        return elapsed

t0 = log("ED fortran start")
eval_i_F, denmat, noccu_gs = ed_wrapper(comm, **NiPS3_ed)
t1 = log("ED fortran finish")

t2 = log("ED python start")
out = ed_wrapper(comm, fortran=False, **NiPS3_ed)
eval_i, evec_i, emat_i, emat_n, umat_i, umat_n = out
t3 = log("ED python finish")

t4 = log("RIXS fortran start")
rixs_F, poles_F = rixs_wrapper(comm, fortran=True, **NiPS3_rixs)
t5 = log("RIXS fortran finish")

t6 = log("RIXS python start")
rixs, poles = rixs_wrapper(comm, fortran=False, **NiPS3_rixs,
             eval_i=eval_i,
             evec_i=evec_i,
             emat_i=emat_i,
             umat_i=umat_i,
             emat_n=emat_n,
             umat_n=umat_n,
            )
t7 = log("RIXS python finish")

np.testing.assert_allclose(rixs, rixs_F, atol=1e-4)

if rank == 1:
    print(f"Doing nd={nd}")
    print(f"ED \t F={(t1-t0):8.3f}  s \t  P={(t3-t2):8.3} s")
    print(f"RIXS \t F={(t5-t4):8.3f} s \t P={(t7-t6):8.3} af")
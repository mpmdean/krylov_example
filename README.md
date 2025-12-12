Docker environment mpmdean/edrixs_petsc was created by adding petsc4py and slepc4py to 
https://github.com/EDRIXS/edrixs/blob/master/docker/Dockerfile#L54
and using a complex double precision numbers

ENV PETSC_CONFIGURE_OPTIONS="--with-scalar-type=complex --with-precision=double"

RUN pip install --upgrade pip setuptools

RUN pip install numpy scipy sympy matplotlib sphinx mpi4py ipython jupyter jupyterlab ipympl ipywidgets lmfit petsc petsc4py slepc slepc4py

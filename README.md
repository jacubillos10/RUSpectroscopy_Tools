# RUSpectroscopy_tools

A C extension module for generating Gamma and Epsilon matrices, in Resonant Ultrasound Spectroscopy.

## Authors

- Alejandro Cubillos - alejandro4cm@outlook.com
- Manuela Rivas
- Julián Rincón 

## Import: 
```python
from rusmodules import rus
```

## Usage Example
```python
gamma = rus.gamma_matrix(N, C, geometry, shape)
E = rus.E_matrix(N, shape) 
```

# Where: 
* N <int> represents the maximim grade of the polynomials of the basis functions 
* C <np.array> 6x6 matrix with the elastic constants
* geometry <np.array> (3,) shape array containing the dimensions of the sample
* shape <int> 0 for parallelepiped, 1 for cylinder and 2 to spheroid

Solve forward problem with scipy:
Eigenvalues return $m\omega^2$
```python
eigenvals, eigenvects = scipy.linalg.eigh(a = gamma, b = E)
omega = (eigenvals[6:]/m)**(1/2) #remember the fisrt 6 eigenvalues must be near zero
frequencies = omega/(2*np.pi)
``` 

## Installation: 
```bash
pip3 install ruspectroscopy-tools

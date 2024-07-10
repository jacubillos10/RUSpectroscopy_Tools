#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <stdio.h>
#include <stdlib.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <omp.h>

/*!******************************************************************************
 * @brief This function returns the double factorial of a given integer
 * This function return the double factorial of a given integer. Warning: returns a float
 * @param N <int> The integer which double factorial will be calculated
 * @return prod <double> The double factorial of N
 *********************************************************************************/
double fact2(int N)
{
	if (N < -1)
	{
		printf("Arithmetic Error: Cannot get double factorial of an input less than -1\n");
		exit(EXIT_FAILURE);
	} else if ( N == 0 || N == -1 || N == 1){
		return 1.0;
	} else {
		double prod = 1.0;
		for (int i = N; i > 1; i-=2)
		{
			prod = prod * i;
			//printf("idd: %i, prod: %li\n", i, prod);
		}
		return prod;
	}
}
/*!******************************************************************************
 * @brief This function changes from 4 index notation to Voigt notation. 
 * The function return an interer depending on the input of the two integers this vay:
 * [0 0] -> 0, [1 1]-> 1, [2 2] -> 2, [1 2] or [2 1] -> 3, [0 2] or [2 0] -> 4, 
 * [0 1] or [1 0] -> 5.
 * @param i <int> First Index
 * @param i <int> Second Index
 * @return index <int> New index transformed into Voigt notation 
 *******************************************************************************/
int it_c(int i, int j)
{
	int index;
	if (i == j) { index = i;}
	else {index = 6 - (i +j);}
	return index;
}

/*!******************************************************************************
 * @brief This function returns one term if the sum composing one element in gamma matrix
 * This function returns a term in the equation 20 of the document Plantilla_PropuestaTdG2015.tex (and the multiplied with the volume)
 * @param exp_index1 <int[3]> First set of exponents of the basis functions (lambda1, mu1, nu1)
 * @param exp_index2 <int[3]> Second set of exponents of the basis functions (lambda2, mu2, nu2)
 * @param i1 <int> Integer equivalent of the i index of equation 20
 * @param i2 <int> Integer equivalent of the k index of equation 20
 * @param j1 <int> Integer equivalent of the j index of equation 20
 * @param j2 <int> Integer equivalent of the l index of equation 20
 * @param C <double[6][6]> Matrix of elastic constants
 * @param geo_poar <double[3]> Array which contains the dimensions of the sample in cm 
 * @param options <int> Shape of the sample. 0 for parallelepiped, 1 for cylinder, 2 for ellipsoid
 * @return  <double> A float that represents a term in the sum of a element of gamma matrix 
 *******************************************************************************/
double generate_term_in_gamma_matrix_element(int exp_index1[3], int exp_index2[3], int i1, int i2, int j1, int j2, double C[6][6], double geo_par[3], int options)
{
	int coeff[3], Q, S;
	int array_j1[3] = {0,0,0};
	int array_j2[3] = {0,0,0};
	double R, alpha, beta, P;
	array_j1[j1] = 1;
	array_j2[j2] = 1;
	for (int i = 0; i < 3; i++) {coeff[i] = exp_index1[i] + exp_index2[i] + 1 - array_j1[i] - array_j2[i];}
	Q = (1 - (int)pow(-1,coeff[0]))*(1 - (int)pow(-1,coeff[1]))*(1 - (int)pow(-1,coeff[2]));
	S = exp_index1[j1]*exp_index2[j2];
	if (Q == 0 || S == 0) 
	{
		return 0;
	}
	if (options == 1)
	{
		R = (0.0+fact2(coeff[0] - 2)*fact2(coeff[1] - 2))/(fact2(coeff[0] + coeff[1])*coeff[2]);
		alpha = M_PI/4;
	} else if (options == 2) {
		R = (0.0+fact2(coeff[0] - 2)*fact2(coeff[1] - 2)*fact2(coeff[2] - 2))/fact2(coeff[0] + coeff[1] + coeff[2]);
		alpha = M_PI/6;
	} else {
		R = 1.0/(coeff[0]*coeff[1]*coeff[2]);
		alpha = 1.0;
	}
	if (j1 == j2) {beta = (geo_par[0]*geo_par[1]*geo_par[2])/pow(geo_par[j1],2);}
	else {beta = geo_par[3-j1-j2];}
	P = 4*alpha*C[it_c(i1,j1)][it_c(i2,j2)]*beta;
	//for (int i = 0; i < 6; i++){ for (int j = 0; j < 6; j++) {printf("%f ", C[i][j]);} printf("\n");}
	//for (int i = 0; i < 3; i++) {printf("%f, %i, %i\n", geo_par[i], exp_index1[i], exp_index2[i]); }
	return P*Q*S*R;
}
/*!******************************************************************************
 * @brief This function returns one element in gamma matrix
 * This function returns the full sum of  equation 20 of the document Plantilla_PropuestaTdG2015.tex (and then multiplied with the volume)
 * @param i1 <int> Integer equivalent of the i index of equation 20
 * @param i2 <int> Integer equivalent of the k index of equation 20
 * @param exp_index1 <int[3]> First set of exponents of the basis functions (lambda1, mu1, nu1)
 * @param exp_index2 <int[3]> Second set of exponents of the basis functions (lambda2, mu2, nu2)
 * @param C <double[6][6]> Matrix of elastic constants
 * @param geo_par <double[3]> Array which contains the dimensions of the sample in cm 
 * @param options <int> Shape of the sample. 0 for parallelepiped, 1 for cylinder, 2 for ellipsoid
 * @return <double> A float representing an element of gamma matrix 
 *******************************************************************************/
double generate_gamma_matrix_element(int i1, int i2, int exp_index1[3], int exp_index2[3], double C[6][6], double geo_par[3], int options)
{
	double suma = 0.0;
	for (int j = 0; j < 3; j++)
	{
		for (int l = 0; l < 3; l++)
		{
			suma += generate_term_in_gamma_matrix_element(exp_index1, exp_index2, i1, i2, j, l, C, geo_par, options);
		}
	}
	return suma;
}
/*!******************************************************************************
 * @brief This function returns one element in E matrix
 * This function returns the equation 19 of the document Plantilla_PropuestaTdG2015.tex 
 * @param i1 <int> Integer equivalent of the i index of equation 19
 * @param i2 <int> Integer equivalent of the k index of equation 19
 * @param exp_index1 <int[3]> First set of exponents of the basis functions (lambda1, mu1, nu1)
 * @param exp_index2 <int[3]> Second set of exponents of the basis functions (lambda2, mu2, nu2)
 * @param options <int> Shape of the sample. 0 for parallelepiped, 1 for cylinder, 2 for ellipsoid
 * @return <double> A float representing an element of E matrix 
 *******************************************************************************/
double generate_E_matrix_element(int i1, int i2, int exp_index1[3], int exp_index2[3], int options)
{
	if (i1 != i2)
	{
		return 0.0;
	} else {
		int coeff[3], Q;
		double R;
		for (int i = 0; i < 3; i++) {coeff[i] = exp_index1[i] + exp_index2[i] + 1;} 
		Q = (1 - (int)pow(-1,coeff[0]))*(1 - (int)pow(-1,coeff[1]))*(1 - (int)pow(-1,coeff[2]));
		if (Q == 0)
		{
			return 0.0;
		}
		if (options == 1)
		{
			R = (0.0+fact2(coeff[0] - 2)*fact2(coeff[1] - 2))/(fact2(coeff[0] + coeff[1])*coeff[2]);
		} else if (options == 2) {
			R = (0.0+fact2(coeff[0] - 2)*fact2(coeff[1] - 2)*fact2(coeff[2] - 2))/(0.0+fact2(coeff[0] + coeff[1] + coeff[2]));
		} else {
			R = 1.0/(coeff[0]*coeff[1]*coeff[2]);
		}	
		//for (int i = 0; i < 3; i++) {printf("coeff %i: %i, fact: %li \n", i, coeff[i], fact2(coeff[i]));}	
		//printf("Operacion 1: %li\n", (fact2(coeff[0] - 2)*fact2(coeff[1] - 2)));
		//printf("Operacion 2: %li\n", fact2(coeff[0] + coeff[1])*coeff[2]);
		//printf("35!!= %f\n", fact2(35));
		return Q*R;
	}
}
/*!******************************************************************************
 * @brief This function return all posible combinations of three integers which sum N
 * This function returns a memory addres which has stored other memory addreses where the 3D arrays of possible indexes summing N are stored
 * @param N <int> Mentioned integer N
 * @return <**int> Memory address where the memory addresses of the combinations are stored 
 *******************************************************************************/
int **generate_combinations(int N) {
    int R = ((N + 1) * (N + 2) * (N + 3))/6;

    int **combi = (int **)malloc(R * sizeof(int *));
    for (int i = 0; i < R; i++) 
	{
        combi[i] = (int *)malloc(3 * sizeof(int));
    }

    int l = 0;
    for (int n = 0; n <= N; n++) {
        for (int i = 0; i <= n; i++) {
            for (int j = 0; j <= n - i; j++) {
                int k = n - i - j;
                combi[l][0] = i;
                combi[l][1] = j;
                combi[l][2] = k;
                l++;
            }
        }
    }
    return combi;
}

/*!******************************************************************************
 * @brief This function cleans the memory allocations created by generate_combinations() function
 * @param combi <**int> Memory address where the combinations addresses are stored
 * @param total_combinations <int> Totan possible combinations of threee integers which sum N, or total number of addesses.
 *******************************************************************************/
void free_combinations(int **combi, int total_combinations) {
    for (int i = 0; i < total_combinations; i++) {
        free(combi[i]);
    }
    free(combi);
}

/*!******************************************************************************
 * @brief This function calculates each element in gamma matrix and exports it as a np.array() 
 *******************************************************************************/
PyObject  *gamma_matrix_py(PyObject *self, PyObject *args)
{
	PyArrayObject *C, *geo_par;
	int N, options;
	
	if (!PyArg_ParseTuple(args, "iO!O!i", &N, &PyArray_Type, &C, &PyArray_Type, &geo_par, &options))
	{
		return NULL;
	}

	if (!PyArray_Check(C) || !PyArray_Check(geo_par))
	{
		PyErr_SetString(PyExc_TypeError, "Expected numpy arrays for C and geo_par");
		return NULL;
	}
	double *geo_par_data = (double *)PyArray_DATA(geo_par);
	double (*C_data)[6] = (double (*)[6])PyArray_DATA(C);

	if (PyArray_NDIM(geo_par) != 1 || PyArray_DIM(geo_par, 0) != 3 || 
			PyArray_NDIM(C) != 2 || PyArray_DIM(C, 0) != 6 || PyArray_DIM(C, 1) != 6)
	{
		PyErr_SetString(PyExc_ValueError, "C must be an np.array of dimensions (6,6) and geo_par must be an array of dimensions (3,)");
		return NULL;
	}
	/*
	 * Aquí se empieza a escribir la función de verdad. Bueno casi. Aquí definimos que tan grande queremos que sea la matriz de respuesta 
	 */
	int R = ((N + 1)*(N + 2)*(N + 3))/6;
	npy_intp dims[2] = {3*R, 3*R};
	PyObject *gamma_array = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
	if (gamma_array == NULL)
	{
		PyErr_SetString(PyExc_MemoryError, "Gamma array memory could be not allocated");
		return NULL;
	}
	double *gamma_data = (double *)PyArray_DATA((PyArrayObject *)gamma_array);
	/*
	 * Ahora sí aquí empiezas a codificar la lógica de la función 
	 */
	 int **combi;
	 combi = generate_combinations(N); //Recuerde que el primer índice de combi es la combinación específica y el segundo va de cero a 2
	 if (combi == NULL) 
	 {
		 PyErr_SetString(PyExc_MemoryError, "Combinations array memory could not be allocated");
		 Py_DECREF(gamma_array); // Aquí se libera gamma_array en caso de que la asignación falle
		 return NULL;
     }			
	 #pragma omp parallel for collapse(4)
	 for (int i = 0; i < 3; i++)
	 {
		 for (int k = 0; k < 3; k++)
		 {
			 for (int lm = 0; lm < R; lm++)
			 {
				 for (int lf = 0; lf < R; lf++)
				 {
					 gamma_data[3*R*(i*R +lm) + (k*R + lf)] = generate_gamma_matrix_element(i, k, combi[lm], combi[lf], C_data, geo_par_data, options);
				 }
			 }
		 } 
	 }
	 free_combinations(combi, R);
	 //printf("Hello from C \n");
	 return gamma_array;

}

/*!******************************************************************************
 * @brief This function calculates each element in E matrix and exports it as a np.array() 
 *******************************************************************************/
PyObject *E_matrix_py(PyObject *self, PyObject *args)
{
	int N, options;
	if (!PyArg_ParseTuple(args, "ii", &N, &options))
	{
		return NULL;
	}
	int R = ((N + 1)*(N + 2)*(N + 3))/6;
	int **combi;
	npy_intp dims[2] = {3*R, 3*R};
	PyObject *E_array = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
	if (E_array == NULL)
	{
		PyErr_SetString(PyExc_MemoryError, "Memory for E matrix could not be allocated");
		return NULL;
	}
	double *E_data = (double *)PyArray_DATA((PyArrayObject *)E_array);
	combi = generate_combinations(N);
	
	if (combi == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Combinations array memory could not be allocated");
        Py_DECREF(E_array); 
        return NULL;
    }
	#pragma omp parallel for collapse(4)	
	for (int i = 0; i < 3; i++)
	{
		for (int k = 0; k < 3; k++)
		{
			for (int lm = 0; lm < R; lm++)
			{
				for (int lf = 0; lf < R; lf++)
				{
					E_data[3*R*(i*R +lm) + (k*R + lf)] = generate_E_matrix_element(i, k, combi[lm], combi[lf], options);
				}
			}
		} 
	}
	free_combinations(combi, R);
	return E_array;

}

PyObject *generate_element_in_gamma_matrix_py(PyObject *self, PyObject *args)
{
	PyArrayObject *exp_index1, *exp_index2, *C, *geo_par;
    int i1, i2, options;
	double result;


	if (!PyArg_ParseTuple(args, "iiO!O!O!O!i",
				&i1, &i2,
				&PyArray_Type, &exp_index1,
				&PyArray_Type, &exp_index2,
				&PyArray_Type, &C,
				&PyArray_Type, &geo_par,
				&options))
	{
		return NULL;
	}

    if (!PyArray_Check(exp_index1) || !PyArray_Check(exp_index2) ||
        !PyArray_Check(C) || !PyArray_Check(geo_par))
    {
        PyErr_SetString(PyExc_TypeError, "Expected numpy arrays for exp_index1, exp_index2, C, and geo_par.");
        return NULL;
    }

    int *exp_index1_data = (int *)PyArray_DATA(exp_index1);
    int *exp_index2_data = (int *)PyArray_DATA(exp_index2);
    double (*C_data)[6] = (double (*)[6])PyArray_DATA(C);
    double *geo_par_data = (double *)PyArray_DATA(geo_par);

    if (PyArray_NDIM(exp_index1) != 1 || PyArray_DIM(exp_index1, 0) != 3 ||
        PyArray_NDIM(exp_index2) != 1 || PyArray_DIM(exp_index2, 0) != 3 ||
        PyArray_NDIM(C) != 2 || PyArray_DIM(C, 0) != 6 || PyArray_DIM(C, 1) != 6 ||
        PyArray_NDIM(geo_par) != 1 || PyArray_DIM(geo_par, 0) != 3)
    {
        PyErr_SetString(PyExc_ValueError, "Invalid dimensions for input arrays.");
        return NULL;
    }
	//for (int i = 0; i < 3; i++) {printf("|%i|%i|\n", exp_index1_data[i], exp_index2_data[i]);}
    result = generate_gamma_matrix_element(i1, i2, exp_index1_data, exp_index2_data,
                                                   C_data,
                                                   geo_par_data, options);

    return PyFloat_FromDouble(result);
}

PyObject *generate_element_in_E_matrix_py(PyObject *self, PyObject *args)
{
	PyArrayObject *exp_index1, *exp_index2;
	int i1, i2, options;
	double result;
	if (!PyArg_ParseTuple(args, "iiO!O!i",
				&i1, &i2,
				&PyArray_Type, &exp_index1,
				&PyArray_Type, &exp_index2,
				&options))
	{
		return NULL;
	}
	if (!PyArray_Check(exp_index1) || !PyArray_Check(exp_index2))
	{
		PyErr_SetString(PyExc_TypeError, "Expected numpy arrays for exp_index1 and exp_index2");
		return NULL;
	}

	int *exp_index1_data = (int *)PyArray_DATA(exp_index1);
	int *exp_index2_data = (int *)PyArray_DATA(exp_index2);
	if (PyArray_NDIM(exp_index1) != 1 || PyArray_DIM(exp_index1, 0) != 3 ||
			PyArray_NDIM(exp_index2) != 1 || PyArray_DIM(exp_index2, 0) != 3)
	{
		PyErr_SetString(PyExc_ValueError, "Invalid dimensions for exp_index1 or exp_index2. Must be (3,)");
		return NULL;
	}
	result = generate_E_matrix_element(i1, i2, exp_index1_data, exp_index2_data, options);
	return PyFloat_FromDouble(result);
}	


static PyMethodDef methods[] = {
	{"generate_gamma_matrix_element", generate_element_in_gamma_matrix_py, METH_VARARGS, "This functions generates a term in teh sum of an element of gamma matrix"},
	{"generate_E_matrix_element", generate_element_in_E_matrix_py, METH_VARARGS, "This function generates a matrix element of E"},
	{"gamma_matrix", gamma_matrix_py, METH_VARARGS, "This function computes the gamma matrix.\n@Input N <int>: Maximum order of the exponents in the base functions.\n@Input C <np.array>: Matrix of elastic constants (6x6).\n@Input geo_par <np.array>: Array with the dimensions of the sample in cm.\n@Input options <int>: 0 for parallelepiped, 1 for cylinder, 2 for ellipsoid.\n@Return <np.array>: The gamma matrix.\n"},
	{"E_matrix", E_matrix_py, METH_VARARGS, "This function computes the E matrix.\n@Input N <int>: Maximum order of the exponents in the base functions.\n@Input options <int>: 0 for parallelepiped, 1 for cylinder, 2 for ellipsoid.\n@Return <np.array>: The E matrix.\n"},
	{NULL, NULL, 0, NULL}
};

static struct PyModuleDef rus_module = {
	PyModuleDef_HEAD_INIT,
	"rus",
	"Prototipo de modelo de rus.",
	-1,
	methods
};

PyMODINIT_FUNC PyInit_rus()
{ 
	PyObject *module = PyModule_Create(&rus_module);
	import_array();
	return module;
}

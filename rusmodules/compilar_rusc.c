#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <stdio.h>
#include <stdlib.h>
#include </usr/include/python3.12/Python.h>
#include </home/cubos/.local/lib/python3.12/site-packages/numpy/core/include/numpy/arrayobject.h>
#include <math.h>

int fact2(int N)
{
	if (N < -1)
	{
		printf("Arithmetic Error: Cannot get double factorial of an input less than -1\n");
		exit(EXIT_FAILURE);
	} else if ( N == 0 || N == -1 || N == 1){
		return 1;
	} else {
		return N * fact2(N-2);
	}
}

int it_c(int i, int j)
{
	int index;
	if (i == j) { index = i;}
	else {index = 6 - (i +j);}
	return index;
}

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


PyObject *fact_2(PyObject *self, PyObject *args)
{
	// OK :) Esta function ya sirve. 
	int x, resp;
	// Aquí guarda los argumentos "dados" en variables de C 
	PyArg_ParseTuple(args, "i", &x);
	resp = fact2(x);
	return PyLong_FromLong(resp);
	/*return  PyLong_FromLong(x + y);*/
}

PyObject *generate_term_in_gamma_matrix_element_py(PyObject *self, PyObject *args)
{
	PyArrayObject *exp_index1, *exp_index2, *C, *geo_par;
    int i1, i2, j1, j2, options;
	double result;

	if (!PyArg_ParseTuple(args, "O!O!iiiiO!O!i",
				&PyArray_Type, &exp_index1,
				&PyArray_Type, &exp_index2,
				&i1, &i2, &j1, &j2,
				&PyArray_Type, &C,
				&PyArray_Type, &geo_par,
				&options))
	{
		return NULL;
	}
	// Validate input arrays
    if (!PyArray_Check(exp_index1) || !PyArray_Check(exp_index2) ||
        !PyArray_Check(C) || !PyArray_Check(geo_par))
    {
        PyErr_SetString(PyExc_TypeError, "Expected numpy arrays for exp_index1, exp_index2, C, and geo_par.");
        return NULL;
    }

    // Convert numpy arrays to C arrays
    int *exp_index1_data = (int *)PyArray_DATA(exp_index1);
    int *exp_index2_data = (int *)PyArray_DATA(exp_index2);
    double (*C_data)[6] = (double (*)[6])PyArray_DATA(C);
    double *geo_par_data = (double *)PyArray_DATA(geo_par);

    // Ensure dimensions match expected sizes
    if (PyArray_NDIM(exp_index1) != 1 || PyArray_DIM(exp_index1, 0) != 3 ||
        PyArray_NDIM(exp_index2) != 1 || PyArray_DIM(exp_index2, 0) != 3 ||
        PyArray_NDIM(C) != 2 || PyArray_DIM(C, 0) != 6 || PyArray_DIM(C, 1) != 6 ||
        PyArray_NDIM(geo_par) != 1 || PyArray_DIM(geo_par, 0) != 3)
    {
        PyErr_SetString(PyExc_ValueError, "Invalid dimensions for input arrays.");
        return NULL;
    }
	//for (int i = 0; i < 3; i++) {printf("|%i|%i|\n", exp_index1_data[i], exp_index2_data[i]);}
    // Call the C function
    result = generate_term_in_gamma_matrix_element(exp_index1_data, exp_index2_data,
                                                   i1, i2, j1, j2, C_data,
                                                   geo_par_data, options);

    // Return the result as a Python float
    return PyFloat_FromDouble(result);
}

/*
··································································································
······················· AHORA SI CREEMOS UNA FUNCIÓN QUE DEVUELVA UN ARRAY ·······················
··································································································
*/

PyObject *funcion1(PyObject *self, PyObject *args)
{
	/* Esta si deberitas va a devolver un array */
	PyArrayObject *arr;
	PyArg_ParseTuple(args, "O", &arr); // "O" de object, de python. 
	if (PyErr_Occurred())
	{
		return NULL;
	} 

	if (!PyArray_Check(arr))// || PyArray_TYPE(arr) != NPY_DOUBLE || !PyArray_IS_C_CONTIGUOUS(arr))
	{
		PyErr_SetString(PyExc_TypeError, "Debe colocar un np.array c-contiguo con dtype de double como argumento");
		return NULL;
	}

	int64_t size = PyArray_SIZE(arr); //equivalente a len(arr)
	double *datos;
	// Las siguientes líneas guardan los elementos del array de python en "datos"
	int nd = 1; //número de dimensiones (del array supongo)
	npy_intp dims[] = { [0] = size };
	PyArray_AsCArray((PyObject **)&arr, &datos, dims, nd, PyArray_DescrFromType(NPY_DOUBLE));
	//double *datos = PyArray_DATA(arr);
	if (PyErr_Occurred())
	{
		//El dueño del video dice que hay otro concepto importante por aprender que es
		// "REFERENCE COUNTING" y que no está cubierto en el video
		return NULL;
	}

	PyObject *resultado = PyArray_SimpleNew(nd, dims, NPY_DOUBLE);
	if (PyErr_Occurred())
	{
		return NULL;
	}
	double *datos_resultado = PyArray_DATA((PyArrayObject *)resultado);
	for (int i=0;i<size;i++)
	{
		datos_resultado[i] = 2 * datos[i];
	}
	PyArray_Free((PyObject *)arr, datos);
	return resultado;	
	
}


static PyMethodDef methods[] = {
	{"fact2", fact_2, METH_VARARGS, "This functions computes the double factorial of a given integer"},
	{"generate_term_in_gamma_matrix_element", generate_term_in_gamma_matrix_element_py, METH_VARARGS, "This functions generates a term in teh sum of an element of gamma matrix"},
	{"funcion_chimba", funcion1, METH_VARARGS, "La funcion que devuelve un array"},
	{NULL, NULL, 0, NULL}
};

static struct PyModuleDef rus_test_module = {
	PyModuleDef_HEAD_INIT,
	"rus_test",
	"Prototipo de modelo de rus.",
	-1,
	methods
};

PyMODINIT_FUNC PyInit_rus_test()
{ 
	PyObject *module = PyModule_Create(&rus_test_module);
	import_array();
	return module;
}

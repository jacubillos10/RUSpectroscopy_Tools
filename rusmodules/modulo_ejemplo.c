#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <stdio.h>
#include <stdlib.h>
#include </usr/include/python3.12/Python.h>
#include </home/cubos/.local/lib/python3.12/site-packages/numpy/core/include/numpy/arrayobject.h>

PyObject *add(PyObject *self, PyObject *args)
{
	double x;
	double y;
	// Aquí guarda los argumentos "dados" en variables de C 
	PyArg_ParseTuple(args, "dd", &x, &y);
	/*PyArg_ParseTuple(args, "ii", &x, &y);*/
	// Recuerde que esta clase de funciones debe retornar siempre un pyobject
	return PyFloat_FromDouble(x + y);
	/*return  PyLong_FromLong(x + y);*/
}

PyObject *la_suma(PyObject *self, PyObject *args)
{
	/* Muy bonit ala función que suma los elementos de un array. SIn embargo, lo que necesitamos pa la tesis es una función que me devuelva un array. Ojalá dos dimensional */
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
		return NULL;
	}
	double total = 0;
	for (int i=0;i<size;i++)
	{
		total += datos[i] + 0.0;
	}
	// IMPORTANTE: EN LA DOCUMENTACIÓN DE NUMPY DICE QUE ES IMPORTANTE COLOCAR ESTO PARA EVITAR
	// FUGAS DE MEMORIA
	PyArray_Free((PyObject *)arr, datos);
	return PyFloat_FromDouble(total);	
	
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
	{"add", add, METH_VARARGS, "Esta función suma dos enteros"},
	{"la_mera_suma", la_suma, METH_VARARGS, "Alguna suma de dos arrays"},
	{"funcion_chimba", funcion1, METH_VARARGS, "La funcion que devuelve un array"},
	{NULL, NULL, 0, NULL}
};

static struct PyModuleDef modulo_ejemplo = {
	PyModuleDef_HEAD_INIT,
	"modulo_ejemplo",
	"Este es un módulo de ejemplo para extender python con código en C",
	-1,
	methods
};

PyMODINIT_FUNC PyInit_modulo_ejemplo()
{ 
	PyObject *module = PyModule_Create(&modulo_ejemplo);
	import_array();
	return module;
}

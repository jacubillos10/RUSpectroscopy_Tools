#include <stdio.h>
#include <stdlib.h>
#include </usr/include/python3.12/Python.h>

PyObject *add(PyObject *self, PyObject *args)
{
	double x;
	double y;
	PyArg_ParseTuple(args, "dd", &x, &y);
	/*PyArg_ParseTuple(args, "ii", &x, &y);*/
	return PyFloat_FromDouble(x + y);
	/*return  PyLong_FromLong(x + y);*/
}

static PyMethodDef methods[] = {
	{"add", add, METH_VARARGS, "Esta función suma dos enteros"},
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
	return PyModule_Create(&modulo_ejemplo);
}

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import math
import numpy as np
import matplotlib.pyplot as plt




"""APARTADO A"""

"""1)"""

"""
Voy a definir la funcion f de forma que me permita aplicarla
a un array completo, es decir, calcula f a cada elemento del array.
"""
def f(x,sigma):
	if type(x) not in (int, float, long, list):
		raise TypeError, "f input must be a number or list of numbers."
	else:
		if(type(x)==list ):
			rt=[0 for i in xrange(len(x))]
			position=0
			for i in x:
				rt[position]=math.exp(-0.5*(float(i)*float(i))/(float(sigma)*float(sigma)))
				position=position+1
			return rt
		else:
			return math.exp(-0.5*(float(x)*float(x))/(float(sigma)*float(sigma)))


"""
La funcion getMask recibe como parametro sigma, es decir el numero de
pixeles (numero de elementos) que tendra la mascara. Entenderemos como
relevantes aquellos valores que se encuentran como maximo a 3*sigma de
distancia de la media (0). GetMask devuelve un array con sigma elementos
donde cada elemento contiene el valor de f evaluado en la discretizacion
de la funcion.

Un ejemplo. Supongamos f con sigma=5. GetMask devolveria un vector de 5
elementos, que serian: |f(-15,5)|f(-7.5,5)|f(0,5)|f(15,5)|f(7.5,5)|.
Hay que multiplicar por un factor de normalizacion para que el valor de la
suma de todos los elementos de la mascara sumen 1.
"""
def getMask(sigma):
	if(sigma>1):
		limit=3*sigma
		if(sigma%2==0):
			raise ValueError, "Mask length must be an odd value."
		else:
			mask1D=[0 for i in xrange(sigma)]
			positionzero=int(sigma/2)
			xAxisValue=0
			mask1D[positionzero]=f(0,sigma)
			sumUnit=(limit)/float((sigma/2))
			for i in xrange(positionzero+1,sigma):
				xAxisValue=xAxisValue+sumUnit
				mask1D[i]=f(xAxisValue,sigma)
				mask1D[(sigma-1)-i]=f(xAxisValue,sigma)

			normFactor=(1/sum(mask1D))
			
			for i in xrange(len(mask1D)):
				mask1D[i]=mask1D[i]*normFactor
			return mask1D
	else:
		return [f(0,sigma)]
	

"""2)"""

"""
La version 1 (V1) corresponde al primer paso del apartado 2). 
Coge un vector y una mascara y calcula el filtrado del vector
con respecto de dicha mascara.

El tamaño de la mascara debe ser impar, pues el pixel central
es el que marca que pixel es para el que estamos calculando
los valores
"""
def convolution2Vectors(mask,vect):
	maskPositionZero=len(mask)/2

	result=[0 for i in xrange(len(vect))]
	
	startPosition=maskPositionZero
	finishPosition=len(result)-maskPositionZero
	
	for i in xrange(startPosition,finishPosition):
		result[i]=0.0
		for j in xrange(len(mask)):
			#print(mask[j],"*",vect[j+(i-startPosition)])
			result[i]=float(result[i])+(float(mask[j])*float(vect[j+(i-startPosition)]))
			#print(type(result[i]),"-",result[i])
			#print(type(mask[j]),"--",mask[j])
			#print(type(vect[j+(i-startPosition)]),"---",vect[j+(i-startPosition)])
		#result[i]=float(result[i])/len(mask)
		result[i]=float(result[i])

	return result[startPosition:finishPosition]

"""
La version 2 (V2) corresponde al segundo paso del apartado 2).
De lo que se trata es de copiar los bordes ya sea constante a
cero o con reflejo
"""

def createAuxVector(mask,vect,borderType):
	if(len(mask)<len(vect)): #Forzamos a que la mascara sea menor que el vector
		maskPositionZero=len(mask)/2
		
		result=[0 for i in xrange(len(vect)+(len(mask)-1))]
		
		startPosition=maskPositionZero
		finishPosition=len(result)-maskPositionZero
		
		for i in xrange(startPosition,finishPosition):
			result[i]=vect[i-startPosition]

		if(borderType==0): #Borde a ceros
			for i in xrange(0,startPosition):
				result[i]=0
				result[len(result)-(1+i)]=0

		if(borderType==1): #Borde reflejo
			for i in xrange(0,startPosition):
				result[i]=result[finishPosition-(1+i)]
				result[len(result)-(1+i)]=result[startPosition+i]

		return result
	else:
		raise TypeError, "Mask's length must be smaller than signal length."


def loadImage(path):
	im = cv2.imread(path,3)
	b,g,r = cv2.split(im)	#CUIDADO!! OpenCV usa BGR en lugar de RGB.
	return cv2.merge([r,g,b])

def paintImage(image,windowtitle="",imagetitle="",axis=False):
	fig = plt.figure()
	fig.canvas.set_window_title(windowtitle)
	plt.imshow(image),plt.title(imagetitle)
	if(not axis):
		plt.xticks([]),plt.yticks([])
	plt.show()

def paintMatrixImages(imagematrix,imagetitles,windowtitle="",axis=False):
	nrow=len(imagematrix)
	ncol=len(imagematrix[0])

	prefix=int(str(nrow)+str(ncol))
	
	for i in xrange(len(imagematrix)):
		for j in xrange(len(imagematrix[i])):
			#print(int(str(prefix)+str(1+(i*ncol+j))))
			plt.subplot(int(str(prefix)+str(1+(i*ncol+j))))
			plt.imshow(imagematrix[i][j])
			plt.title(imagetitles[i][j])
			if(not axis):
				plt.xticks([]),plt.yticks([])

	plt.show()

plt.show()

if __name__=='__main__':
	#miVector=[58,42,56,255,12,45,58,12]
	#miVector=[15,158,254,36,25,85,147,8,235,26,87,21,45,58]
	#mascara=getMask(11)
	#senialBorde=createAuxVector(mascara,miVector,0)
	#print(senialBorde)
	#print(convolution2Vectors(mascara,senialBorde))
	
	imagen=loadImage("imagenes/cat.bmp")
	r,g,b=cv2.split(imagen)
	mascara=getMask(57)
	
	for i in xrange(0,len(r)):
 		senialBorde=createAuxVector(mascara,r[i],0)
 		r[i]=convolution2Vectors(mascara,senialBorde)
	
	for i in xrange(0,len(g)):
 		senialBorde=createAuxVector(mascara,g[i],0)
 		g[i]=convolution2Vectors(mascara,senialBorde)
 	
	for i in xrange(0,len(b)):
 		senialBorde=createAuxVector(mascara,b[i],0)
 		b[i]=convolution2Vectors(mascara,senialBorde)
	
	imagensmooth=cv2.merge([r,g,b])
	
	paintMatrixImages([[imagen,imagensmooth]],[["ORIGINAL","SMOOTH HORIZONTAL"]],"PROBANDOOOO")

	print("Hola")
	
	#imagen = cv2.imread("imagenes/marilyn.bmp",3)	
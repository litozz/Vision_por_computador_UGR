#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import random

"""------------------------------FUNCIONES AUXILIARES------------------------------"""

"""
La funcion loadImage carga una imagen en color o blanco y negro.
el parametro color debe valer: 0->blanco/negro, distinto de 0 -> 
"""
def loadImage(path,color):
	if(color=="COLOR"):
		im = cv2.imread(path,cv2.IMREAD_COLOR)
		b,g,r = cv2.split(im)	#CUIDADO!! OpenCV usa BGR en lugar de RGB.
		return cv2.merge([r,g,b])
	elif(color=="GRAYSCALE"):
		im = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
		return cv2.merge([im,im,im])
	else:
		raise ValueError, "LoadImage color values must be COLOR or GRAYSCALE"

"""
La funcion paintImage pinta la imagen por pantalla
"""
def paintImage(image,windowtitle="",imagetitle="",axis=False):
	fig = plt.figure()
	fig.canvas.set_window_title(windowtitle)
	plt.imshow(image),plt.title(imagetitle)
	if(not axis):
		plt.xticks([]),plt.yticks([])
	plt.show()


"""
La funcion paintMatrixImages pinta un conjunto de imagenes en un lienzo
"""
def paintMatrixImages(imagematrix,imagetitles,windowtitle="",axis=False):
	nrow=len(imagematrix)
	ncol=len(imagematrix[0])

	prefix=int(str(nrow)+str(ncol))

	fig = plt.figure()
	fig.canvas.set_window_title(windowtitle)
	
	for i in xrange(nrow):
		for j in xrange(len(imagematrix[i])):
			plt.subplot(int(str(prefix)+str(1+(i*ncol+j))))
			plt.imshow(imagematrix[i][j])
			plt.title(imagetitles[i][j])

			if(not axis):
				plt.xticks([]),plt.yticks([])

	plt.show()






"""
La funcion fillImage tiene como objetivo mostrar como las imagenes
se rellenan y mostrarlas, no se usa en ninguna otra funcion, pues
su cometido es meramente ilustrativo.
"""
def fillImage(image,sigma,bordertype):
 	mask=getMask(sigma)
 	lenMask=len(mask)
	r,g,b=cv2.split(image)
	nrow=len(r)
	ncol=len(r[0])

	if((r!=g).any() or (r!=b).any()): #COLOR
		senialBordeR=np.zeros( (nrow,ncol+lenMask-1) )
		senialBordeG=np.zeros( (nrow,ncol+lenMask-1) )
		senialBordeB=np.zeros( (nrow,ncol+lenMask-1) )

		for i in xrange(0,nrow):
	 		senialBordeR[i]=createAuxVector(mask,r[i],bordertype)
	 		senialBordeG[i]=createAuxVector(mask,g[i],bordertype)	
			senialBordeB[i]=createAuxVector(mask,b[i],bordertype)

		senialBordeR=np.transpose(senialBordeR) 
		senialBordeG=np.transpose(senialBordeG)
		senialBordeB=np.transpose(senialBordeB)

		nrow=len(senialBordeR)
		ncol=len(senialBordeR[0])

		senialBordeRV=np.zeros(  (nrow,ncol+lenMask-1) )
		senialBordeGV=np.zeros(  (nrow,ncol+lenMask-1) )
		senialBordeBV=np.zeros(  (nrow,ncol+lenMask-1) )

		for i in xrange(0,nrow):
	 		senialBordeRV[i]=createAuxVector(mask,senialBordeR[i],bordertype)
	 		senialBordeGV[i]=createAuxVector(mask,senialBordeG[i],bordertype) 		
	 		senialBordeBV[i]=createAuxVector(mask,senialBordeB[i],bordertype)

		senialBordeRV=np.uint8(np.transpose(senialBordeRV)) 
		senialBordeGV=np.uint8(np.transpose(senialBordeGV))
		senialBordeBV=np.uint8(np.transpose(senialBordeBV))	
	 	

	 	imagereflected=cv2.merge([senialBordeRV,senialBordeGV,senialBordeBV])
	 	return imagereflected

	else: #ESCALA DE GRISES
		senialBordeR=np.zeros( (nrow,ncol+lenMask-1) )
		for i in xrange(0,nrow):
	 		senialBordeR[i]=createAuxVector(mask,r[i],bordertype)
		senialBordeR=np.transpose(senialBordeR) 
		nrow=len(senialBordeR)
		ncol=len(senialBordeR[0])
		senialBordeRV=np.zeros(  (nrow,ncol+lenMask-1) )
		for i in xrange(0,nrow):
	 		senialBordeRV[i]=createAuxVector(mask,senialBordeR[i],bordertype)
		senialBordeRV=np.uint8(np.transpose(senialBordeRV)) 
	 	imagereflected=cv2.merge([senialBordeRV,senialBordeRV,senialBordeRV])
	 	return imagereflected
		



"""
Cuando hacemos el vector auxiliar para los bordes,
la imagen queda en negativo, por lo que es necesario negar el negativo
para volver al color de origen.
"""
def negative(shape):
	return 255-shape


"""
La funcion normalize mapea un valor en un intervalo conocido al mismo valor
medido en un nuevo intervalo.
"""
def normalize(shape,min_v,max_v):
	if(min_v<max_v):
		maxvalue=256
		minvalue=0
		nshape=np.zeros( (len(shape),len(shape[0])) )
		for i in xrange(len(shape)):
			for j in xrange(len(shape[0])):
				if(shape[i][j]>maxvalue):
					maxvalue=shape[i][j]
				if(shape[i][j]<minvalue):
					minvalue=shape[i][j]

		coc=(max_v-min_v)/(maxvalue-minvalue)
		for i in xrange(len(shape)):
			for j in xrange(len(shape[0])):
				nshape[i][j]=(shape[i][j] - minvalue)*coc + min_v
		return nshape
	else:
		raise ValueError, "min value in range must be smaller than max value in range."

"""------------------------------FIN FUNCIONES AUXILIARES------------------------------"""






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
	sigma=abs(sigma)
	limit=3*sigma #DEFINIMOS EL LIMITE DE LA FUNCION F EN 3 SIGMAS
	#La longitud de la mascara va a ser 2*LIMITE+1, asi forzamos que sea impar
	mask1D=[0.0 for i in xrange((2*int(limit))+1)]

	lenMask=len(mask1D)
	positionzero=int(lenMask/2)
	mask1D[positionzero]=f(0,sigma)
	suma=mask1D[positionzero]
	
	for i in xrange(positionzero+1,lenMask):
		resultF=f(i-positionzero,sigma)
		mask1D[i]=resultF
		mask1D[(lenMask-1)-i]=resultF
		suma+=2*resultF

	normFactor=(1/suma)
	
	finalMask=[x * normFactor for x in mask1D]

	return finalMask



"""2)"""

"""
La funcion convolution2Vectors
coge un vector y una mascara y calcula el filtrado del vector
con respecto de dicha mascara.

El tamaño de la mascara debe ser impar, pues el pixel central
es el que marca que pixel es para el que estamos calculando
los valores
"""
"""def convolution2Vectors(mask,vect): 
	result=np.array([0.0 for i in xrange(len(vect))])
	startPosition=len(mask)/2
	finishPosition=len(result)-startPosition

	for i in xrange(startPosition,finishPosition):
		result[i]=sum(mask*vect[i-startPosition:i+startPosition+1])
	return result[startPosition:finishPosition]"""

def convolution2Vectors(mask,vect):	#ESTA VERSION TARDA LA MITAD QUE LA ANTERIOR VERSION
	startPosition=len(mask)/2
	finishPosition=len(vect)-startPosition
	result=[sum([float(a*b) for a,b in zip(mask,vect[i-startPosition:i+startPosition+1])]) for i in xrange(startPosition,finishPosition)]
	return result


"""
La funcion createAuxVector rellena los extremos del vector senial
con los bordes constante a cero, con reflejo o copia del ultimo pixel
o reflejado.
"""

def createAuxVector(mask,vect,borderType):
	result=np.array([0 for i in xrange(len(vect)+(len(mask)-1))])
	startPosition=len(mask)/2
	finishPosition=len(result)-startPosition
	result[startPosition:finishPosition]=vect

	#if(borderType==0): #Borde a ceros

	if(borderType==1): #Borde reflejo
		result[0:startPosition]=result[2*startPosition-1:startPosition-1:-1]
		result[finishPosition:len(result)]=result[finishPosition-1:finishPosition-startPosition-1:-1]
		

	elif(borderType==2): #Borde copia
		result[0:startPosition]=result[startPosition]
		result[finishPosition:len(result)]=result[finishPosition-1]

	return result






"""
La funcion my_imGaussConvol realiza la convolucion de una imagen.
Los parametros que se usan son la imagen que se quiere convolucionar,
el sigma y el tipo de borde: 1 -> borde reflejo,
2-> borde copia, cualquier otro valor -> borde negro
"""

def my_imGaussConvol(image,sigma,bordertype,only_horizontal=False):
	mask=getMask(sigma)
	r,g,b=cv2.split(image)
	if((r!=g).any() or (r!=b).any()): #COLOR
		#Trabajamos en modo CV_32FC3 (FLOAT)
		r=np.float32(r)
		g=np.float32(g)
		b=np.float32(b) 

		r=np.transpose([convolution2Vectors(mask,createAuxVector(mask,r[i],bordertype)) for i in xrange(len(r))])
		g=np.transpose([convolution2Vectors(mask,createAuxVector(mask,g[i],bordertype)) for i in xrange(len(g))])
		b=np.transpose([convolution2Vectors(mask,createAuxVector(mask,b[i],bordertype)) for i in xrange(len(b))])

		r=np.transpose([convolution2Vectors(mask,createAuxVector(mask,r[i],bordertype)) for i in xrange(len(r))])
		g=np.transpose([convolution2Vectors(mask,createAuxVector(mask,g[i],bordertype)) for i in xrange(len(g))])
		b=np.transpose([convolution2Vectors(mask,createAuxVector(mask,b[i],bordertype)) for i in xrange(len(b))])
		#Fin de comentarios
		#Regresamos al modo CV_8UC3 (ENTERO)
		r=np.uint8(r)
		g=np.uint8(g)
		b=np.uint8(b)
		imgi=cv2.merge([r,g,b]) 
	 	return imgi
	else: #ESCALA GRISES
		#Trabajamos en modo CV_32FC3 (FLOAT)
		r=np.float32(r) 
		r=np.transpose([convolution2Vectors(mask,createAuxVector(mask,r[i],bordertype)) for i in xrange(len(r))])
		r=np.transpose([convolution2Vectors(mask,createAuxVector(mask,r[i],bordertype)) for i in xrange(len(r))])
		#Fin de comentarios
		#Regresamos al modo CV_8UC3 (ENTERO)
		r=np.uint8(r)
		imgi=cv2.merge([r,r,r]) 
	 	return imgi





def my_imGaussConvol_singleShape(image,sigma,bordertype,only_horizontal=False):
	mask=getMask(sigma)
	#Trabajamos en modo CV_32FC3 (FLOAT)
	newimage=np.copy(image)
	newimage=np.transpose([convolution2Vectors(mask,createAuxVector(mask,newimage[i],bordertype)) for i in xrange(len(newimage))])
	newimage=np.transpose([convolution2Vectors(mask,createAuxVector(mask,newimage[i],bordertype)) for i in xrange(len(newimage))])
	#Fin de comentarios
	#Regresamos al modo CV_8UC3 (ENTERO)
	#r=np.uint8(r)
	#imgi=cv2.merge([r,r,r]) 
	return newimage



"""
La funcion getHighFrequences toma dos veces la misma imagen,
la original y la convolucionada. Resta la convolucionada a
la imagen original y obtenemos los detalles (altas frecuencias).
Se puede aplicar un filtro laplaciano, por lo que se pasa un coeficiente
laplaciano.
"""
def getHighFrequences(image,imageconv,hFfactor):
	r,g,b=cv2.split(image)
	rc,gc,bc=cv2.split(imageconv)
	if((r!=g).any() or (r!=b).any()): #COLOR
		#Trabajamos en modo CV_32FC3 (CALCULOS EN COMA FLOTANTE)
		r=np.float32(r)
		g=np.float32(g)
		b=np.float32(b) 
		
		rc=np.float32(rc)
		gc=np.float32(gc)
		bc=np.float32(bc)
		
		r=hFfactor*r-rc
		g=hFfactor*g-gc
		b=hFfactor*b-bc
		
		r=normalize(r,0,255)
		g=normalize(g,0,255)
		b=normalize(b,0,255)

		#Regresamos al modo CV_8UC3 (ENTERO)
		r=np.uint8(r)
		g=np.uint8(g)
		b=np.uint8(b)
		
		finalimage=cv2.merge([r,g,b])
		
		return finalimage
	else: #ESCALA GRISES
		#Trabajamos en modo CV_32FC3 (CALCULOS EN COMA FLOTANTE)
		r=np.float32(r)
		rc=np.float32(rc)
		r=hFfactor*r-rc
		r=normalize(r,0,255)
		#Regresamos al modo CV_8UC3 (ENTERO)
		r=np.uint8(r)
		finalimage=cv2.merge([r,r,r])
		return finalimage
		



"""
La funcion getHybridImage construye una imagen hibrida
a partir de dos imagenes. El primer parametro debe ser
una imagen correspondiente a las altas frecuencias 
(extraidas previamente) y el segundo, una imagen alisada
convenientemente.
"""
def getHybridImage(imageHF,imageLF):
	imghf=np.float32(imageHF)
	imglf=np.float32(imageLF)
	hybridImage=(imghf+imglf)/2
	return np.uint8(hybridImage)




"""
La funcion scaleDownImage se encarga de escalar a mas pequeña
una imagen, alisando la imagen y quitando filas y columnas
alternativamente
"""
def scaleDownImage(image,sigma,borderType):
	img=my_imGaussConvol(image,sigma,borderType)
	r,g,b=cv2.split(img)
	nrow=len(r)/2
	ncol=len(r[0])/2
	if((r!=g).any() or (r!=b).any()): #COLOR
		for i in xrange(0,nrow):
			r=np.delete(r,i,0) #Quitando arrays en el eje x (quitar filas)
			g=np.delete(g,i,0)
			b=np.delete(b,i,0)

		for i in xrange(0,ncol):
			r=np.delete(r,i,1) #Quitando arrays en el eje y (quitar columnas)
			g=np.delete(g,i,1)
			b=np.delete(b,i,1)
		return cv2.merge([r,g,b])
	else: #ESCALA GRISES
		for i in xrange(0,nrow):
			r=np.delete(r,i,0) #Quitando arrays en el eje x (quitar filas)
		for i in xrange(0,ncol):
			r=np.delete(r,i,1) #Quitando arrays en el eje y (quitar columnas)
		return cv2.merge([r,r,r])

"""
Esta funcion calcula las imagenes de una piramide Gaussiana de level niveles
y devuelve una lista con todas las imagenes calculadas
"""
def getPyramid(image,sigma,borderType,level):
	pyramid=[np.uint8(image)]
	min_size=min(len(image),len(image[0]))
	if(level<=int(math.log(min_size,2))+1):
		scimage=image
		for i in xrange(1,level):
			scimage=scaleDownImage(scimage,sigma,borderType)
			pyramid.append(np.uint8(scimage))
			#print(len(scimage),len(scimage[0]))
	else:
		raise ValueError, "Image cannot be scaled down more than "+str(int(math.log(min_size,2))+1)+" times."

	return pyramid







"""
Esta funcion pasa el filtro Sobel a una imagen para calcular sus derivadas en X, Y o ambas
"""
def imageGradientSobel(image,bordertype,diffX,diffY):
	maskV=np.array([1,2,1])
	maskH=np.array([-1,0,1])
	r,g,b=cv2.split(image)
	if((r!=g).any() or (r!=b).any()): #COLOR
		#Trabajamos en modo CV_32FC3 (FLOAT)
		r=np.float32(r)
		g=np.float32(g)
		b=np.float32(b) 
		
		if(diffY):
			r=np.transpose([convolution2Vectors(maskV,createAuxVector(maskV,r[i],bordertype)) for i in xrange(len(r))])
			g=np.transpose([convolution2Vectors(maskV,createAuxVector(maskV,g[i],bordertype)) for i in xrange(len(g))])
			b=np.transpose([convolution2Vectors(maskV,createAuxVector(maskV,b[i],bordertype)) for i in xrange(len(b))])

			r=np.transpose([convolution2Vectors(maskH,createAuxVector(maskH,r[i],bordertype)) for i in xrange(len(r))])
			g=np.transpose([convolution2Vectors(maskH,createAuxVector(maskH,g[i],bordertype)) for i in xrange(len(g))])
			b=np.transpose([convolution2Vectors(maskH,createAuxVector(maskH,b[i],bordertype)) for i in xrange(len(b))])

		if(diffX):
			r=np.transpose([convolution2Vectors(maskH,createAuxVector(maskH,r[i],bordertype)) for i in xrange(len(r))])
			g=np.transpose([convolution2Vectors(maskH,createAuxVector(maskH,g[i],bordertype)) for i in xrange(len(g))])
			b=np.transpose([convolution2Vectors(maskH,createAuxVector(maskH,b[i],bordertype)) for i in xrange(len(b))])

			r=np.transpose([convolution2Vectors(maskV,createAuxVector(maskV,r[i],bordertype)) for i in xrange(len(r))])
			g=np.transpose([convolution2Vectors(maskV,createAuxVector(maskV,g[i],bordertype)) for i in xrange(len(g))])
			b=np.transpose([convolution2Vectors(maskV,createAuxVector(maskV,b[i],bordertype)) for i in xrange(len(b))])

		r=normalize(r,0,255)
		g=normalize(g,0,255)
		b=normalize(b,0,255)
		
		#Regresamos al modo CV_8UC3 (ENTERO)
		r=np.uint8(r)
		g=np.uint8(g)
		b=np.uint8(b)
		imgi=cv2.merge([r,g,b]) 
	 	return imgi
	else: #ESCALA GRISES
	 	#Trabajamos en modo CV_32FC3 (FLOAT)
		r=np.float32(r)
		if(diffY):
			r=np.transpose([convolution2Vectors(maskV,createAuxVector(maskV,r[i],bordertype)) for i in xrange(len(r))])
			r=np.transpose([convolution2Vectors(maskH,createAuxVector(maskH,r[i],bordertype)) for i in xrange(len(r))])
		if(diffX):
			r=np.transpose([convolution2Vectors(maskH,createAuxVector(maskH,r[i],bordertype)) for i in xrange(len(r))])
			r=np.transpose([convolution2Vectors(maskV,createAuxVector(maskV,r[i],bordertype)) for i in xrange(len(r))])
		r=normalize(r,0,255)
		#Regresamos al modo CV_8UC3 (ENTERO)
		r=np.uint8(r)
		imgi=cv2.merge([r,r,r]) 
	 	return imgi









"""------------FUNCIONES PARA MOSTRAR LOS DIFERENTES APARTADOS------------"""

"""
Muestra todos los bordes posibles para calcular el alisado.
"""
def showAllBorders(ruta,color,sigma):
	imagen=loadImage(ruta,color)
	im0=fillImage(imagen,sigma,0)
	im1=fillImage(imagen,sigma,1)
	im2=fillImage(imagen,sigma,2)

	print("Esta viendo los diferentes rellenos de imagen para poder filtrar, cierre la ventana para continuar.")

	paintMatrixImages(
		[[imagen,im0],[im1,im2]],
		[["ORIGINAL","UNIFORM 0"],["REFLECT","UNIFORM COPY"]],
		"Practica 1 - Vision por computador - Jose Carlos Martinez Velazquez"
	)



"""
Muestra todo el proceso de construccion de un alisado.
"""
def showSmoothedImage(ruta,color,sigma,border):
	imagen=loadImage(ruta,color)
	
	imr=fillImage(imagen,sigma,border)

	imconvH=my_imGaussConvol(imagen,sigma,border,True)

	r,g,b=cv2.split(imconvH)
	r=np.transpose(r)
	g=np.transpose(g)
	b=np.transpose(b)

	aux=cv2.merge([r,g,b])

	imconv=my_imGaussConvol(aux,sigma,border,True)

	r,g,b=cv2.split(imconv)
	r=np.transpose(r)
	g=np.transpose(g)
	b=np.transpose(b)	

	imconv=cv2.merge([r,g,b])

	print("Esta viendo el proceso de construccion del suavizado, cierre la ventana para continuar.")

	paintMatrixImages(
		[[imagen,imr,imconvH,imconv]],
		[["ORIGINAL","BORDED","HORIZ. SMOOTH","COMPLETE SMOOTH"]],
		"Practica 1 - Vision por computador - Jose Carlos Martinez Velazquez"
	)



"""
Muestra solo la imagen de altas frecuencias, la de bajas frecuencias y la imagen hibrida.
"""
def showConstructionHybridImage(rutaAltas,colorAltas,sigmaAltas,
								rutaBajas,colorBajas,sigmaBajas,
								factorLaplaciano,border):

	imagenAltas=loadImage(rutaAltas,colorAltas)
	imagenBajas=loadImage(rutaBajas,colorBajas)

	imconv=my_imGaussConvol(imagenAltas,sigmaAltas,border)
	imAltas=getHighFrequences(imagenAltas,imconv,factorLaplaciano)

	imBajas=my_imGaussConvol(imagenBajas,sigmaBajas,border)

	hybridimage=getHybridImage(imAltas,imBajas)

	print("Esta viendo el proceso de construccion de la imagen hibrida, cierre la ventana para continuar.")

	paintMatrixImages(
		[[imAltas,imBajas,hybridimage]],
		[["Hi-FREQUENCES","Lo-FREQUENCES","HYBRID IMAGE"]],
		"Practica 1 - Vision por computador"
	)

	return hybridimage





"""
Muestra el proceso completo de calculo de altas frecuencias, bajas frecuencias e
imagen hibrida, a continuacion muestra la imagen hibrida sola.
"""
def showConstructionHybridImage2(rutaAltas,colorAltas,sigmaAltas,
								rutaBajas,colorBajas,sigmaBajas,
								factorLaplaciano,border):

	imagenAltas=loadImage(rutaAltas,colorAltas)
	imagenBajas=loadImage(rutaBajas,colorBajas)

	imconv=my_imGaussConvol(imagenAltas,sigmaAltas,border)
	imAltas=getHighFrequences(imagenAltas,imconv,factorLaplaciano)

	imBajas=my_imGaussConvol(imagenBajas,sigmaBajas,border)

	hybridimage=getHybridImage(imAltas,imBajas)

	print("Esta viendo el proceso de construccion de la imagen hibrida, cierre la ventana para continuar.")

	paintMatrixImages(
		[[imagenAltas,imconv,imAltas],[imagenBajas,imBajas],[hybridimage]],
		[["ORIGINAL_HF","SMOOTHED","Hi-FREQUENCES"],["ORIGINAL_LF","Lo-FREQUENCES"],["HYBRID IMAGE"]],
		"Practica 1 - Vision por computador"
	)

	print("Esta viendo la imagen hibrida, cierre la ventana para continuar.")

	paintMatrixImages(
		[[hybridimage]],
		[["HYBRID IMAGE"]],
		"Practica 1 - Vision por computador - Jose Carlos Martinez Velazquez"
	)

	return hybridimage



"""
Muestra una piramide Gaussiana previamente calculada
"""
def showPyramid(pyramid,windowtitle):
	nrow=len(pyramid[0])
	ncol=len(pyramid[0][0])
	
	min_size=min(nrow,ncol)
	
	fig = plt.figure()
	fig.canvas.set_window_title(windowtitle)
		
	ax1=plt.subplot2grid((nrow,ncol+ncol/2), (0,0), rowspan=nrow,colspan=ncol)
	plt.xticks([]),plt.yticks([])
	plt.imshow(pyramid[0])

	rowini=0
	row_span=nrow/2
	col_span=ncol/2

	ax2 = plt.subplot2grid((nrow,ncol+ncol/2), (rowini,ncol), rowspan=row_span,colspan=col_span)
	plt.xticks([]),plt.yticks([])
	plt.imshow(pyramid[1])

	for i in xrange(2,len(pyramid)):
		rowini+=row_span
		row_span=nrow/2**i
		col_span=ncol/2**i

		plt.subplot2grid((nrow,ncol+ncol/2), (rowini, ncol), rowspan=row_span, colspan=col_span)
		plt.xticks([]),plt.yticks([])
		plt.imshow(pyramid[i])
	
	print("Esta viendo la piramide gaussiana, cierre la ventana para continuar.")
	plt.show()
"""-----------------------------------------------------------------------"""









"""
-------------------------------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------INICIO PRACTICA 2--------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------------------------
"""



"""
Detector de esquinas de Harris. Dada una imagen, se calculan sus eigenvalores a partir de su imagen en escala de grises.
Para ello necesitamos definir el tamaño del vecindario o entorno y la apertura del operador Sobel k (tamaño mascara derivadas).
El valor harris sera calculado para cada pixel i por (lambda_i_1)*(lambda_i_2)-alpha*((lambda_i_1)+(lambda_i_2))^2.
"""


def harrisCornerDetector(image,blocksize,k_size,alpha=0.04,borderType=0):
	#r,g,b=cv2.split(image)
	r = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
	if(borderType==1):
		eigenvvr=cv2.cornerEigenValsAndVecs(r, blockSize=blocksize, ksize=k_size,borderType=cv2.BORDER_REFLECT)
	elif(borderType==2):
		eigenvvr=cv2.cornerEigenValsAndVecs(r, blockSize=blocksize, ksize=k_size,borderType=cv2.BORDER_REPLICATE)
	else:
		eigenvvr=cv2.cornerEigenValsAndVecs(r, blockSize=blocksize, ksize=k_size)
	l1,l2,x1,y1,x2,y2=cv2.split(eigenvvr)
	rnew=(l1*l2)-alpha*((l1+l2)*(l1+l2))
	#rnew=(l1*l2)/(l1+l2) #Operador de Harris alternativo
	#rnew=np.uint8(normalize(rnew,0,255))

	image_corner=cv2.merge([rnew,rnew,rnew])
	return image_corner


"""
La funcion isLocalMax permite saber si el valor del centro de una matriz es maximo local. El numero de filas y columnas
debe ser siempre impar, de lo contrario, siempre devuelve False, pues no existe un elemento central.
"""
def isLocalMax(neighborhood):
	nrow=len(neighborhood)
	ncol=len(neighborhood[0])
	if (nrow%2!=0 and ncol%2!=0):
		max_v=neighborhood[nrow/2][ncol/2]
		return (np.amax(neighborhood)==max_v)
	else:
		return false

"""
La funcion modifyZeros pone a 0 una region cuadrada de una matriz. Indicando el pixel
central, calcula los pixeles que debe poner a 0 sumando y restando la mitad en el eje X
y el eje Y.
"""
def modifyZeros(matrix,row,col,windowsize):
	matrix[row-windowsize/2:(row+1)+windowsize/2,col-windowsize/2:(col+1)+windowsize/2]=0
	#return matrix

#def getWindow(matrix,row,col,windowsize):
#	return matrix[row-windowsize/2:(row+1)+windowsize/2,col-windowsize/2:(col+1)+windowsize/2]


#La mascara de supresion de no maximos no se puede descomponer
# def nonMaximumSuppression(image,windowSize):
# 	startPosition=int(windowSize/2)
# 	finishpositionrow=len(image)-startPosition
# 	finishpositioncol=len(image[0])-startPosition
# 	r,g,b=cv2.split(image)
	
# 	newr=np.zeros( (len(r),len(r[0])))
	
# 	for i in xrange(startPosition,finishpositionrow):
# 		for j in xrange(startPosition,finishpositioncol):
# 			window=r[i-startPosition:(i+1)+startPosition,j-startPosition:(j+1)+startPosition]
# 			if(isLocalMax(window)):
# 				newr=modifyZeros(newr,i,j,windowSize)
# 				newr[i][j]=255
	
# 	return cv2.merge([newr,newr,newr])


"""
Fase de supresion de no maximos. Dado el tamaño de la ventana se calcula una posicion de comienzo
asi como una posicion de final para fila y columna. A partir de aqui trabajamos con una imagen en
escala de grises representante de la imagen real y una matriz del mismo tamaño que la imagen toda
a negro. Se recorre en orden de izquierda a derecha y de arriba a abajo de forma que si el pixel
del centro de la ventana es maximo, se queda todo negro menos el pixel central, si no, el pixel 
central queda negro.
"""
def nonMaximumSuppression(image,windowSize,thresold):
	startPosition=int(windowSize/2)
	finishpositionrow=image.shape[0]-startPosition
	finishpositioncol=image.shape[1]-startPosition
	r = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

	newr=np.zeros( (r.shape[0],r.shape[1]))
	for i in xrange(startPosition,finishpositionrow):
		for j in xrange(startPosition,finishpositioncol):
			window=r[i-startPosition:(i+1)+startPosition,j-startPosition:(j+1)+startPosition]
			if(isLocalMax(window) and np.amax(window)>thresold):
				modifyZeros(newr,i,j,windowSize)
				newr[i][j]=255
			
	#print("Puntos detectados "+str(np.count_nonzero(newr)))
	newr=np.uint8(newr)
	return cv2.merge([newr,newr,newr])



"""
Dada una matriz de valores harris y una matriz con supresion de no maximos,
sortPoints devuelve una lista de puntos detectados como maximos ordenados en
funcion de su valor harris. Para ello uso la funcion sortIdx que provee 
OpenCV.
"""
def sortPoints(imageharris,imagenms,nPoints,scale):
	nfilas=imageharris.shape[0]
	ncol=imageharris.shape[1]
	#r,g,b=cv2.split(imageharris)
	#rn,gn,bn=cv2.split(imagenms)
	r = cv2.cvtColor(imageharris,cv2.COLOR_RGB2GRAY)
	rn = cv2.cvtColor(imagenms,cv2.COLOR_RGB2GRAY)

	lista=r.reshape((1,r.shape[0]*r.shape[1]))[0]
	
	ordered=cv2.sortIdx(lista,cv2.SORT_EVERY_COLUMN+cv2.SORT_DESCENDING)

	points=[]
	for posicion in xrange(ordered.shape[0]):
		fila=ordered[posicion][0]/ncol
		columna=ordered[posicion][0]-(fila*ncol)
		if(rn[fila][columna]==255):
			points.append((fila,columna,scale,r[fila][columna]))
		
	return points[0:nPoints+1]




"""
La funcion getStrongestPoints devuelve un conjunto ordenado 
de puntos en formato (x,y,escala) en funcion de su valor harris
"""
def getStrongestPoints(pir,blockSize,kSize,suppresionsize,thresold,maxPoints,k=0.04,bordertype=0,weights=[]):
	harrispir=[]
	nmspir=[]
	npointsPir=[]
	nScales=len(pir)
	harrispir=[harrisCornerDetector(pir[i],blockSize,kSize,k,bordertype) for i in xrange(nScales)]
	nmspir=[nonMaximumSuppression(harrispir[i],suppresionsize,thresold) for i in xrange(nScales)]
	if(len(weights)==0):
		for i in xrange(nScales): 
			npointsPir.extend(sortPoints(harrispir[i],nmspir[i],maxPoints,i))
	elif (len(weights)==len(pir) and sum(weights)<1):
		for i in xrange(nScales):
			npointsPir.extend(sortPoints(harrispir[i],nmspir[i],int(maxPoints*weights[i]),i))
	else:
		raise UnboundLocalError, "Legth of weights must be equal to lenght of pyramid and must amount 1."
	npointsPir=sorted(npointsPir, key=lambda x: -x[3])
	return [(x,y,escala) for (x,y,escala,valor) in npointsPir[0:maxPoints]]






"""
La funcion paintPointsInImage pinta en una imagen una lista de puntos, con diametro
inversamente proporcional a su tamaño.
"""
def paintPointsInImage(image,points,r=255,g=0,b=0,thickness=2):
	newimg=np.copy(image)
	for point in points:
		cv2.circle(newimg,(int(point[1])*2**int(point[2]),int(point[0])*2**int(point[2])), 5+2**(2*(int(point[2]))), (r,g,b), thickness)

	#print("Se estan pintando "+str(len(points))+" puntos.")
	return newimg



"""
La funcion refinePoints ajusta las coordenadas de los puntos fuertes
encontrados. Deadpoint es una tupla, normalmente (-1,1). Hay que
refinar cada punto en la escala a la que pertenece.
"""
def refinePoints(pir,criteria,previouspoints,maskSize,zerozone):
	pir_gray=[]
	coords_scale=[]
	result_scale=[]
	nImages=len(pir)
	
	for scale in xrange(nImages):
		pir_gray.append(cv2.cvtColor(pir[scale],cv2.COLOR_RGB2GRAY))
		coords_scale.append([])
		for point in previouspoints:
			if(point[2]==scale):
				coords_scale[scale].append(np.array(np.float32( [point[0],point[1]] )))
	
	result_scale=[(cv2.cornerSubPix( pir_gray[i], np.array(coords_scale[i]), (int(maskSize/2),int(maskSize/2)), zerozone, criteria )) for i in xrange(len(pir))]
	
	newpoints=[]
	for i in xrange(len(result_scale)):
		for j in xrange(len(result_scale[i])):
			newpoints.append( (result_scale[i][j][0],result_scale[i][j][1],i) )

	return newpoints





"""
La funcion getOrientation calcula la orientacion del vector gradiente 
en cada pixel en radianes. Para ello alisa fuertemente las derivadas
con un sigma 4.5. La derivada en X, al ser divisor, tenemos que no puede
ser 0, entonces le asignamos 1 como nos explican en el video siguiente:
https://www.youtube.com/watch?v=j7r3C-otk-U (sobre el minuto 10 - 11)
"""
def getOrientation(image,sigma=4.5,kSize=3,bordertype=cv2.BORDER_REFLECT):
	#r,g,b=cv2.split(newimg)
	r = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
	derx=cv2.Sobel(r,cv2.CV_32F,1,0,ksize=kSize)
	dery=cv2.Sobel(r,cv2.CV_32F,0,1,ksize=kSize)
	derx=my_imGaussConvol_singleShape(derx,sigma,bordertype)
	dery=my_imGaussConvol_singleShape(dery,sigma,bordertype)
	#magnitude=np.sqrt((derx*derx)+(dery*dery))
	derx[derx==0.]=1.
	orientation=np.arctan(dery/derx)
	return orientation




"""
La funcion addOrientationToPoints añade la orientación a cada punto 
de forma que cada punto tiene formato (row,col,escala,orientacion).
"""
def addOrientationToPoints(image,points):
	orientation=getOrientation(image)
	newpoints=[(row,col,scale,orientation[row][col]) for (row,col,scale) in points]
	return newpoints


	
"""
La funcion paintPointsAndOrientationInImage pinta los puntos y la orientacion de los mismos.
"""
def paintPointsAndOrientationInImage(image,points_with_orientation,rp=255,gp=0,bp=0,ro=0,go=255,bo=0,thickness=2):
	newimg=np.copy(image)
	for pnt in points_with_orientation:
		radius=5+2**(2*(int(pnt[2])))
		pointorigin=(int(pnt[1])*2**int(pnt[2]),int(pnt[0])*2**int(pnt[2]))
		pointdstx=int(pointorigin[0] + (math.cos(pnt[3])*radius))
		pointdsty=int(pointorigin[1] - (math.sin(pnt[3])*radius))
		#destinyx = int(int(pnt[1])*2**int(pnt[2])) + int(pnt[3])
		#destinyy = int(int(pnt[0])*2**int(pnt[2])) + 5+2**(2*(int(pnt[2])))
		pointdst=(pointdstx,pointdsty)
		cv2.circle(newimg,pointorigin, radius, (rp,gp,bp), thickness)
		cv2.arrowedLine(newimg,
			pointorigin,
			pointdst,
			(ro,go,bo),
			thickness)

	return newimg










"""
La funcion getStrongestPointsComplete es una compilacion de getStrongestPoints (obtener los
puntos fuertes en base a su valor harris), la ordenacion, la fase de refinamiento y el calculo
de la orientacion. Asi, a partir de una piramide, y un numero maximo de puntos asi como 
otros parametros como tamaños de mascara permite obtener una lista de puntos fuertes ordenados
en base a su valor Harris de manera que cada punto es una cuadrupla (fila, columna, escala, orientacion)
"""
def getStrongestPointsComplete(pir,blockSize,kSize,suppresionsize,thresold,maxPoints,
						criteria,maskSize,deadzone,sigma=4.5,
						k=0.04,bordertype=0,weights=[]):
	#FASE DE OBTENCION DE PUNTOS FUERTES Y ORDENACION
	points=getStrongestPoints(pir,blockSize,kSize,suppresionsize,thresold,maxPoints,k,bordertype,weights)
	
	#FASE DE REFINAMIENTO DE PUNTOS FUERTES
	points=refinePoints(piry1,criteria,points,maskSize,deadzone)
	
	#FASE DE CALCULO DE LA ORIENTACION
	points=addOrientationToPoints(pir[0],points)

	return points





"""
La funcion getKeyPointsAndDescriptorsKAZE permite obtener los keypoints y descriptores usando el detector KAZE
"""
def getKeyPointsAndDescriptorsKAZE(image):
	kaze = cv2.KAZE_create()
	kp = kaze.detect(image, None)
	kp, des = kaze.compute(image, kp)
	return kp,des

"""
La funcion getKeyPointsAndDescriptorsAKAZE permite obtener los keypoints y descriptores usando el detector KAZE
"""
def getKeyPointsAndDescriptorsAKAZE(image):
	akaze = cv2.AKAZE_create()
	kp = akaze.detect(image, None)
	kp, des = akaze.compute(image, kp)
	return kp,des

	
"""
La funcion getMatchesBF permite obtener matches a partir de los descriptores usando BFMatcher
"""
def getMatchesBF(des1,des2,crosscheck=True):
	bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crosscheck)
	matches = bf.match(des1,des2)
	matches = sorted(matches, key = lambda x:x.distance)
	return matches


"""
La funcion getMatchesDescriptor permite obtener matches a partir de los descriptores usando DescriptorMatcher
Los parametros tipo deben ser "BruteForce" o "BruteForce-Hamming"
"""
def getMatchesDescriptor(des1,des2):
	dm = cv2.DescriptorMatcher_create("BruteForce")
	matches = dm.match(des1,des2)
	matches = sorted(matches, key = lambda x:x.distance)
	return matches


"""
La funcion drawMatches permite dibujar las correspondencias entre dos imagenes. Devuelve una imagen que
une las dos pasadas por parametro y dibuja lineas rectas entre los puntos correspondidos
"""
def drawMatches(img1,kp1,img2,kp2,matches):
	return cv2.drawMatches(img1,kp1,img2,kp2,matches,None,flags=2)


"""
La funcion H devuelve una homografia de imagenIzq a imagenRight
"""
def getHomography(imgLeft,imgRight):
	kp1,des1=getKeyPointsAndDescriptorsAKAZE(imgLeft)
	kp2,des2=getKeyPointsAndDescriptorsAKAZE(imgRight)
	matches=getMatchesBF(des1,des2,crosscheck=True)
	p1=np.float32([(kp1[m.queryIdx].pt) for m in matches])
	p2=np.float32([(kp2[m.trainIdx].pt) for m in matches])
	H, mask = cv2.findHomography(p1, p2, cv2.RANSAC, 1.0)
	return H








"""
Dada una imagen con pixeles que sobran devuelve la imagen sin esos pixeles
"""
def quitLeftOver(image):
	imageGray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
	x,y,w,h = cv2.boundingRect(imageGray)
	newImg = image[y:y+h,x:x+w]
	return newImg


"""
Dadas dos imagenes, construye una imagen panoramica de ambas, para lo que se calcula
una homografia de translacion para poner la imagen de la izquierda y la homografia
entre ambas imagenes, que se multiplica la de translacion para finalmente colocar la
imagen de la derecha.
"""
def constructPanoramaTwoImages(imgLeft,imgRight,quitleftover=True):
	maxX=0
	maxY=0
	if imgLeft.shape[0]>imgRight.shape[0]:
		maxY=imgLeft.shape[0]
	else:
		maxY=imgLeft.shape[0]
	if imgLeft.shape[1]>imgRight.shape[1]:
		maxX=imgLeft.shape[1]
	else:
		maxX=imgLeft.shape[1]
	
	size_lienzo=[maxX*2,maxY*2]
	H0=np.array([[1.,0.,(size_lienzo[0]/2)-(maxX/2)],[0.,1.,(size_lienzo[1]/2)-(maxY/2)],[0.,0.,1.]])
	H=np.eye(3)
	result=cv2.warpPerspective(imgLeft, H0.dot(H), (size_lienzo[0],size_lienzo[1]),cv2.BORDER_TRANSPARENT)
	H=H.dot(getHomography(imgRight,imgLeft))
	cv2.warpPerspective(imgRight, H0.dot(H), (size_lienzo[0],size_lienzo[1]),dst=result,borderMode=cv2.BORDER_TRANSPARENT)
	if(quitleftover):
		result=quitLeftOver(result)
	return result
	




"""
Dadas N imagenes en una lista, construye una imagen panoramica con todas, para lo que se calcula
la posicion central y se coloca en el lienzo. A partir de aqui se van componiendo homografias por
la derecha y por la izquierda, de manera que el panorama va creciendo hacia ambos lados. Esto es
para reducir las deformaciones producidas por las perspectivas.
"""
def constructPanoramaMultipleImages(listOfImages,quitleftover=True):
	maxX=0
	maxY=0
	numImages=len(listOfImages)
	imgCentro=numImages/2
	for image in listOfImages:
		if image.shape[0]>maxY:
			maxY=image.shape[0]
		if image.shape[1]>maxX:
			maxX=image.shape[1]
	
	size_lienzo=[maxX*numImages,maxY*numImages]
	H0=np.array([[1.,0.,(size_lienzo[0]/2)-(maxX/2)],[0.,1.,(size_lienzo[1]/2)-(maxY/2)],[0.,0.,1.]])
	H=np.eye(3)
	result=cv2.warpPerspective(listOfImages[imgCentro], H0.dot(H), (size_lienzo[0],size_lienzo[1]),cv2.BORDER_TRANSPARENT)
	
	#paintImage(quitLeftOver(result))
	
	#Crecimiento hacia la derecha
	for i in xrange(imgCentro,numImages-1):
		H=H.dot(getHomography(listOfImages[i+1],listOfImages[i]))
		cv2.warpPerspective(listOfImages[i+1], H0.dot(H), (size_lienzo[0],size_lienzo[1]),dst=result,borderMode=cv2.BORDER_TRANSPARENT)
		#paintImage(quitLeftOver(result))
	
	H=np.eye(3)
	#Crecimiento hacia la izquierda
	for i in xrange(imgCentro,0,-1):
		H=H.dot(getHomography(listOfImages[i-1],listOfImages[i]))
		cv2.warpPerspective(listOfImages[i-1], H0.dot(H), (size_lienzo[0],size_lienzo[1]),dst=result,borderMode=cv2.BORDER_TRANSPARENT)
		#paintImage(quitLeftOver(result))
	
	if(quitleftover):
		result=quitLeftOver(result)
		
	
	return result
	




# """
# Construye un panorama de izquierda a derecha. Es la primera idea 
# pero no es una buena idea, el panorama está construido mal.
# """
# def constructPanoramaMultipleImagesBAD(listOfImages,quitleftover=True):
# 	result=listOfImages[0]
# 	for imagen in listOfImages:
# 		result=constructPanoramaTwoImages(result,imagen)
# 	return result




"""
-------------------------------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------INICIO PRACTICA 3--------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------------------------
"""


"""
La funcion generateCamera devuelve una matriz 3x4
con determinante distinto de cero, esto es, una
cámara finita.
"""
def generateCamera():
	camera=np.random.rand(3,4)
	while(np.linalg.det(camera[0:3,0:3])==0):
		camera=np.random.rand(3,4)
	return camera


"""
La funcion generatePointGrid genera una rejilla
de puntos en dos planos distintos ortogonales.
"""
def generatePointGrid():
	dec_vals=np.arange(0.1,1.1,0.1)
	grid=[]
	for x_1 in dec_vals:
		for x_2 in dec_vals:
			grid.append((0,x_1,x_2))
			grid.append((x_2,x_1,0))
	return grid



"""
La funcion projectPointsPixelCoords, recibe la
matriz de camara y puntos en formato mundo.
Convierte los puntos a 1x4 y realiza la multiplicacion
Px. La tercera coordenada siempre es 1, luego las
coordenadas x e y quedan divididas por una constante
(la coordenada z proyectada).
"""
def projectPoints(camera,world_points):
	#Añado una coordenada más a los puntos para convertirlos en 1x4
	proj_points=np.array([[(x,y,z,1)] for (x,y,z) in world_points]) 
	#Realizo el producto (P*x^T)^T para cada punto
	proj_points=[np.transpose(np.dot(camera,np.transpose(point))) for point in proj_points]
	#Vuelvo a convertir a lista de tuplas en coordenadas píxel.
	#Para ello divido las coordenadas x e y por la tercera coordenada
	#y a continuacion me quedo con las dos primeras coordenadas.
	proj_points=[(point[0][0]/point[0][2],point[0][1]/point[0][2],1) for point in proj_points]
	return proj_points


"""
Dado un punto en coordenadas homogéneas, devuelve los puntos en pixel.

def convertToPixelCoords(points):
	return [(point[0]/point[2],point[1]/point[2]) for point in points if len(point)==3]
"""


"""
La función calculate_A_matrix calcula 
la matriz A del algoritmo DLT a partir
de dos conjuntos de puntos en correspondencias.
Por cada pareja de puntos se calcula una matriz
Ai 2x9 que se compone de:
Pmundoi			0^T 		 		-xi*Pmundoi
0^T 			Pmundoi 			-yi*Pmundoi
La matriz A definitiva estará compuesta por
todas las filas de las matrices Ai, es decir,
si tenemos n parejas de puntos en correspondencias,
entonces la matriz A definitiva tendrá tamaño 2nx9
"""
def calculate_A_matrix(world_points,projected_points):
	A=[]
	npoints=len(world_points)
	if(npoints==len(projected_points) and npoints>=6):
		for i in xrange(npoints):
			A.append([
						world_points[i][0], 						#Xi
						world_points[i][1], 						#Yi
						world_points[i][2], 						#Zi
						1.0, 											#1
						0.0, 											#0
						0.0, 											#0
						0.0, 											#0
						0.0, 											#0
						-projected_points[i][0]*world_points[i][0],	#-xi*Xi
						-projected_points[i][0]*world_points[i][1],	#-xi*Yi
						-projected_points[i][0]*world_points[i][2],	#-xi*Zi
						-projected_points[i][0]						#-xi
					])
			A.append([
						0.0, 											#0
						0.0, 											#0
						0.0, 											#0
						0.0, 											#0
						world_points[i][0], 						#Xi
						world_points[i][1], 						#Yi
						world_points[i][2], 						#Zi
						1.0, 											#1
						-projected_points[i][1]*world_points[i][0],	#-yi*Xi
						-projected_points[i][1]*world_points[i][1],	#-yi*Yi
						-projected_points[i][1]*world_points[i][2],	#-yi*Zi
						-projected_points[i][1]						#-yi
					])
	else:
		raise ValueError, "Must be at least 6 points in correspondence and both sets must have the same lenght."
	return np.array(A)



"""
Dados dos conjuntos de puntos en correspondencia,
donde conj1[i] esta en correspondencia con conj2[i],
la funcion estimate_camera estima una cámara que
hace corresponder estos puntos.
"""
def estimate_camera(world_points, projected_points):
	A=calculate_A_matrix(world_points, projected_points)
	Dcv,Ucv,Vtcv=cv2.SVDecomp(A)
	column_min_D=np.where(Dcv == np.min(Dcv))[0][0]
	P=Vtcv[column_min_D].reshape(3,4)
	return P








"""
Normaliza una matriz en un rango determinado.
Es necesario para igualar en rango al calcular
el error.
"""

def normalizeMatrix(M,min_d,max_d):
	if(min_d<max_d):
		max_v=np.max(M)
		min_v=np.min(M)
		return (((max_d - min_d)/(max_v-min_v))*(M-min_v))+min_d
	else:
		raise ValueError, "Desired min must be smaller than desired max."

"""
Calcula la norma de frobenius de una matriz.
"""
def frobenius_norm(matrix):
	#mat_abs=np.abs(matrix)
	#normalized_matrix=normalizeMatrix(mat_abs,0,1)
	return math.sqrt(np.sum(matrix*matrix))





def giveCanvas(proj_points,marginw,marginh,scaling):
	max_value=np.max(proj_points)		
	wcanvas=int((max_value*scaling)+(2*marginw))
	hcanvas=int((max_value*scaling)+(2*marginh))
	shapecanvas=np.zeros((hcanvas,wcanvas),dtype=np.uint8)
	image=cv2.merge([shapecanvas,shapecanvas,shapecanvas])
	return image




"""
La funcion drawProjectedGrid devuelve una imagen 
con los puntos proyectados del el espacio.
"""
def drawProjectedGrid(proj_points,marginw,marginh,scaling):

	image=giveCanvas(proj_points,marginw,marginh,scaling)

	for i in xrange(1,len(proj_points),2):
		#cv2.circle(image,( int( (factor_escalado_w*proj_points[i][0])+marginw ) , int( (hcanvas)-(factor_escalado_h*proj_points[i][1]+marginh) ) ), 1, (255,0,0), 2)
		cv2.circle(image,( int( (proj_points[i][0]*scaling)+(marginw) ) , int( image.shape[1]-(proj_points[i][1]*scaling+marginh) )), 1, (255,0,0), 2)

	for i in xrange(0,len(proj_points)-1,2):
		#cv2.circle(image,( int( (factor_escalado_w*proj_points[i][0])+marginw ) , int( (hcanvas)-(factor_escalado_h*proj_points[i][1]+marginh) ) ), 1, (0,255,0), 2)
		cv2.circle(image,( int( (proj_points[i][0]*scaling)+(marginw) ) , int( image.shape[1]-(proj_points[i][1]*scaling+marginh) )), 1, (0,255,0), 2)
	
	return image










"""
La funcion drawPoints dibuja puntos en una imagen 
pasada por parametro o crea dicha imagen.
"""
def drawPoints(proj_points,marginw,marginh,scaling,color,image=None):
	if(image==None):
		image=giveCanvas(proj_points,marginw,marginh,scaling)
	for i in xrange(len(proj_points)):
		cv2.circle(image,( int( (proj_points[i][0]*scaling)+(marginw) ) , int( image.shape[1]-(proj_points[i][1]*scaling+marginh) )), 1, color, 2)

	return image




"""
La funcion drawChessboardPlane devuelve una imagen 
con las esquinas interiores de un tablero dibujado
"""
def drawChessboardPlane(pointsChessBoard,marginw,marginh,scaling):
	img=drawPoints(pointsChessBoard,marginw,marginh,scaling,(random.random()*1000%255,random.random()*1000%255,random.random()*1000%255))
	return img


"""
La funcion drawAllChessboardPlanes devuelve una imagen 
con las esquinas interiores de todos los tableros dibujados
"""
def drawAllChessboardPlanes(pointsChessBoard,numImages,marginw,marginh,scaling):
	img=giveCanvas(pointsChessBoard,marginw,marginh,scaling)
	for i in xrange(0,numImages):
		img=drawPoints(pointsChessBoard[i*len(pointsChessBoard)/numImages:(i+1)*len(pointsChessBoard)/numImages],marginw,marginh,scaling,(random.random()*1000%255,random.random()*1000%255,random.random()*1000%255),image=img)
	return img





"""
La funcion cropUndistordedImage elimina las zonas
vacias de una imagen con la distorsion corregida
"""
def cropUndistordedImage(originalImage):
	# Separamos las capas y trabajamos con una sola
	r,g,b=cv2.split(originalImage)
	h, w = r.shape[0:2]
	xleft, xright, ysup, yinf = (0,0,0,0)
	# Color la parte vacia
	empty_color = 0
	# Lista de colores en las lineas verticales
	# y horizontales en la mitad
	vertical = [r[i][int(w/2)] for i in range(h)]
	horizontal = [r[int(h/2)][i] for i in range(w)]
	# Las mismas listas de antes del revés (voltea la imagen)
	vertical_rev = vertical[::-1]
	horizontal_rev = horizontal[::-1]
	# Buscamos un cambio de color en las lineas trazadas
	for i in range(2,h):
	    if vertical[i] > empty_color and ysup == 0:
	        ysup = i
	    if vertical_rev[i] > empty_color and yinf == 0:
	        yinf = i
	    if ysup != 0 and yinf != 0:
	        break
	for i in range(2,w):
	    if horizontal[i] > empty_color and xleft == 0:
	        xleft = i
	    if horizontal_rev[i] > empty_color and xright == 0:
	        xright = i
	    if xleft != 0 and xright != 0:
	        break
	#Recortamos conforme a las filas y columnas obtenidas cada capa.
	r = r[ysup:h-yinf, xleft:w-xright]
	g = g[ysup:h-yinf, xleft:w-xright]
	b = b[ysup:h-yinf, xleft:w-xright]
	#Mezclamos las capas de color ya recortadas
	cropped_img=cv2.merge([r,g,b])
	return cropped_img















"""
La funcion selectImagesAndCalibrate recibe una
lista de rutas de imagenes, un tamaño de tablero
a buscar y un tamaño de ventana de refinamiento.
Con estos datos carga y selecciona las imagenes
válidas para calibrar. A continuación calibra la
cámara y devuelve las imágenes corregidas.
"""
def selectImagesAndCalibrate(arrayNames,size_corners,size_refine):
	#Me creo los flags con los valores de la documentacion de OpenCV
	#Esto es necesario porque en Python no se pueden pasar flags
	#a la funcion findChessboardCorners()
	CV_CALIB_CB_ADAPTIVE_THRESH		=	1
	CV_CALIB_CB_FILTER_QUADS		=	4
 	CV_CALIB_CB_NORMALIZE_IMAGE		=	2
 	CV_CALIB_ZERO_TANGENT_DIST		= 	8
	CV_CALIB_FIX_K1 				= 	32
	CV_CALIB_FIX_K2					=	64
	CV_CALIB_FIX_K3					=	128
 	
 	#Criterio de parada para todas las busquedas
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.0001)

	# Lista que contendra las imagenes originales
	originalImages=[]
	# Lista que contendra los puntos de las esquinas encontradas
	imgpoints=[]
	# Lista que contendra los nombres de las imagenes validas
	validimgnames=[]
	# Lista que contendra las imagenes con los puntos pintados
	validimages=[]
	
	for name in arrayNames:
		#Cargar la imagen y convertir a escala de grises
		image=loadImage(name,"COLOR")
		image_gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
		# Encontrar las esquinas
		isvalid, corners = cv2.findChessboardCorners(image_gray, size_corners,flags=CV_CALIB_CB_ADAPTIVE_THRESH or CV_CALIB_CB_FILTER_QUADS or CV_CALIB_CB_NORMALIZE_IMAGE )
		
		#Si la imagen es válida
		if isvalid == True:
			originalImages.append(np.copy(image))
			#Añadimos el nombre de la imagen
			validimgnames.append(name)
			#Refinamiento de los puntos encontrados
			refined_corners=cv2.cornerSubPix(image_gray,corners,size_refine,(-1,-1),criteria)
			#Añado los puntos refinados a la lista que se devolvera
			imgpoints.append(refined_corners)
			#Dibujo los puntos refinados en la imagen correspondiente
			img=cv2.drawChessboardCorners(image,size_corners,refined_corners,isvalid)
			#Añado la imagen con los puntos pintados
			validimages.append(img)
	
	#Si hay al menos tres imagenes vállidas, procedemos a la calibración
	if(len(validimages)>3):
		#Creo los puntos del mundo en mis unidades de medida (de uno en uno)
		objpoints=np.float32([np.array([[j,i,0.0] for i in xrange(size_corners[1]) for j in xrange(size_corners[0])],dtype=np.float32) for image in validimages])

		#Calibro la cámara en base a los puntos medidos en mis unidades y los puntos de la imagen:
		#1. Suponiendo que no hay distorsion
		retNoDist, KNoDist, distortionCoeffNoDist, rotationVectorsNoDist, translationVectorsNoDist = cv2.calibrateCamera(objpoints, imgpoints, image_gray.shape,
																														None,None, 
																														flags= CV_CALIB_FIX_K1 
																														+ CV_CALIB_FIX_K2 
																														+ CV_CALIB_FIX_K3 
																														+ CV_CALIB_ZERO_TANGENT_DIST)
		#2. Suponiendo que hay solo distorsion radial
		retOnlyRadial, KOnlyRadial, distortionCoeffOnlyRadial, rotationVectorsOnlyRadial, translationVectorsOnlyRadial = cv2.calibrateCamera(objpoints, imgpoints, image_gray.shape,
																														None,None, 
																														flags= CV_CALIB_ZERO_TANGENT_DIST)
		#3. Suponiendo que hay solo distorsion tangencial
		retOnlyTan, KOnlyTan, distortionCoeffOnlyTan, rotationVectorsOnlyTan, translationVectorsOnlyTan = cv2.calibrateCamera(objpoints, imgpoints, image_gray.shape,
																														None,None, 
																														flags= CV_CALIB_FIX_K1 
																														+ CV_CALIB_FIX_K2 
																														+ CV_CALIB_FIX_K3)
		#2. Suponiendo que hay ambas distorsiones radial
		retDist, KDist, distortionCoeffDist, rotationVectorsDist, translationVectorsDist = cv2.calibrateCamera(objpoints, imgpoints, image_gray.shape,None,None)
		

		# A partir de aqui solo queda corregir o no la distorsion y añadir las imagenes a
		# las listas correspondientes.
		correctedImagesNoDist=[]
		correctedImagesDist=[]
		croppedImagesDist=[]

		for c_img in originalImages:
			h,  w = c_img.shape[:2]

			newcameramtx, roi=cv2.getOptimalNewCameraMatrix(KNoDist,distortionCoeffNoDist,(w,h),1,(w,h))
			corr_image = cv2.undistort(c_img, KNoDist , distortionCoeffNoDist, None, newcameramtx)
			#corr_image = cv2.undistort(c_img, K, np.float32([0,0,0,0,0]), None)
			correctedImagesNoDist.append(corr_image)

		
			newcameramtx, roi=cv2.getOptimalNewCameraMatrix(KDist,distortionCoeffDist,(w,h),1,(w,h))
			corr_image_dist = cv2.undistort(c_img, KDist, distortionCoeffDist, None, newcameramtx)
			#corr_image_dist = cv2.undistort(c_img, K, distortionCoeff, None)
			correctedImagesDist.append(corr_image_dist)
	

			croppedImagesDist.append(cropUndistordedImage(corr_image_dist))

		return imgpoints,validimgnames,validimages,correctedImagesNoDist,correctedImagesDist,croppedImagesDist,retNoDist,retOnlyRadial,retOnlyTan,retDist

		
		
	else:
		raise IndexError, "Must be at least 3 images where pattern is detected."






"""
La funcion findCorrespondencesORB encuentra puntos
en correspondencia usando ORB
"""
def findCorrespondencesORB(img1,img2,distance_thr):
	orb = cv2.ORB_create()

	kp1, des1 = orb.detectAndCompute(img1,None)
	kp2, des2 = orb.detectAndCompute(img2,None)

	bfdetector = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
	matches = bfdetector.knnMatch(des1,des2,k=2)

	good_matches = []
	pts1 = []
	pts2 = []
	for i,(m,n) in enumerate(matches):
		if m.distance < n.distance*distance_thr:
			good_matches.append(m)
			pts2.append(kp2[m.trainIdx].pt)
			pts1.append(kp1[m.queryIdx].pt)


	good_matches=np.array(good_matches)
	pts1=np.array(pts1)
	pts2=np.array(pts2)

	imgresult=np.zeros((1,1))
	imgresult = cv2.drawMatches(img1,kp1,img2,kp2,good_matches,imgresult,flags=2)
	return imgresult, pts1, pts2





"""
La funcion findCorrespondencesBRISK encuentra puntos
en correspondencia usando BRISK
"""
def findCorrespondencesBRISK(img1,img2,distance_thr):
	brisk = cv2.BRISK_create()

	kp1, des1 = brisk.detectAndCompute(img1,None)
	kp2, des2 = brisk.detectAndCompute(img2,None)

	bfdetector = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
	matches = bfdetector.knnMatch(des1,des2,k=2)

	good_matches = []
	pts1 = []
	pts2 = []
	for i,(m,n) in enumerate(matches):
		if m.distance < n.distance*distance_thr:
			good_matches.append(m)
			pts2.append(kp2[m.trainIdx].pt)
			pts1.append(kp1[m.queryIdx].pt)


	good_matches=np.array(good_matches)
	pts1=np.array(pts1)
	pts2=np.array(pts2)

	imgresult=np.zeros((1,1))
	imgresult = cv2.drawMatches(img1,kp1,img2,kp2,good_matches,imgresult,flags=2)
	return imgresult, pts1, pts2





"""
La funcion findCorrespondencesAKAZE encuentra puntos
en correspondencia usando AKAZE
"""
def findCorrespondencesAKAZE(img1,img2,distance_thr):
	akaze = cv2.AKAZE_create()

	kp1, des1 = akaze.detectAndCompute(img1,None)
	kp2, des2 = akaze.detectAndCompute(img2,None)

	bfdetector = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
	matches = bfdetector.knnMatch(des1,des2,k=2)

	good_matches = []
	pts1 = []
	pts2 = []
	for i,(m,n) in enumerate(matches):
		if m.distance < n.distance*distance_thr:
			good_matches.append(m)
			pts2.append(kp2[m.trainIdx].pt)
			pts1.append(kp1[m.queryIdx].pt)


	good_matches=np.array(good_matches)
	pts1=np.array(pts1)
	pts2=np.array(pts2)

	imgresult=np.zeros((1,1))
	imgresult = cv2.drawMatches(img1,kp1,img2,kp2,good_matches,imgresult,flags=2)
	return imgresult, pts1, pts2





"""
Devuelve solo las correspondencias buenas en base
a la mascara devuelta por findCorrespondences
"""
def getFineCorrespondences(mask,points1,points2):
	return points1[mask.ravel()==1], points2[mask.ravel()==1]







"""
Calcula el error de los puntos en correspondencia
con respecto a las lineas epipolares
"""
def calculateMeanError(points,lines):
	if(len(points)==len(lines)):
		error_sum=sum([abs(lines[i][0]*points[i][0]+lines[i][1]*points[i][1]+lines[i][2])/math.sqrt(lines[i][0]*lines[i][0]+lines[i][1]*lines[i][1]) for i in xrange(len(lines))])
		return (error_sum/len(lines))
	else:
		raise IndexError, "Both lists must have the same lenght."






"""
Permite calcular cuanto se desvía de 0
el producto x'Fx.
"""
def checkCorrectFundamentalMatrix(F,mask,points1,points2):
	if(len(points1)==len(points2)):
		if(len(mask)==len(points1)):
			points1 = points1[mask.ravel()==1]
			points2 = points2[mask.ravel()==1]

		points1=np.float32([[[x,y,1.0]] for [x,y] in points1])
		points2=np.float32([[[x,y,1.0]] for [x,y] in points2])

		mean_error=0
		for i in xrange(len(points1)):
			mean_error+=(points2[i].dot(F)).dot(np.transpose(points1[i]))

		mean_error/=len(points1)
		return mean_error
	else:
		raise IndexError, "Both lists must have the same lenght."










"""
Calcula el movimiento (R,t) de una camara
a partir de sus parametros intrinsecos,
la matriz fundamental y puntos en correspondencia

def calculateCameraMotion(K,F,points1,points2):
	E = np.transpose(K).dot(F).dot(K)
	D,U,Vt=cv2.SVDecomp(E)
	t=np.transpose(U)[-1]
	t1=t
	t2=-t
	#Calculo de las matrices de rotacion
	w=[[0,-1,0],[1,0,0],[0,0,1]]
	R1=U.dot(np.transpose(w)).dot(Vt)				#[UW^TV^T]
	R2=U.dot(np.transpose(w)).dot(np.transpose(Vt))	#[UW^TV]

	#Ver qué camara es la buena
	I=np.eye(3)
	P=K.dot(np.float32([np.append(I[i],0) for i in xrange(len(I))]))
	P1=K.dot(np.float32([np.append(np.copy(R1[i]),t1[i]) for i in xrange(len(R1))]))
	P2=K.dot(np.float32([np.append(np.copy(R1[i]),t2[i]) for i in xrange(len(R1))]))
	P3=K.dot(np.float32([np.append(np.copy(R2[i]),t1[i]) for i in xrange(len(R2))]))
	P4=K.dot(np.float32([np.append(np.copy(R2[i]),t2[i]) for i in xrange(len(R2))]))

	combinations=[[R1,t1],[R1,t2],[R2,t1],[R2,t2]]

	rec_points1=np.float32([cv2.triangulatePoints(P, P1, points1[i], points2[i]) for i in xrange(len(points1))])
	rec_points1=np.float32([[x/k,y/k,z/k] for [x,y,z,k] in rec_points1])
	cont_neg_1=0
	for point in rec_points1:
		if(point[2]<0):
			cont_neg_1+=1

	rec_points2=np.float32([cv2.triangulatePoints(P, P2, points1[i], points2[i]) for i in xrange(len(points1))])
	rec_points2=np.float32([[x/k,y/k,z/k] for [x,y,z,k] in rec_points2])
	cont_neg_2=0
	for point in rec_points2:
		if(point[2]<0):
			cont_neg_2+=1

	rec_points3=np.float32([cv2.triangulatePoints(P, P3, points1[i], points2[i]) for i in xrange(len(points1))])
	rec_points3=np.float32([[x/k,y/k,z/k] for [x,y,z,k] in rec_points3])
	cont_neg_3=0
	for point in rec_points3:
		if(point[2]<0):
			cont_neg_3+=1

	rec_points4=np.float32([cv2.triangulatePoints(P, P4, points1[i], points2[i]) for i in xrange(len(points1))])
	rec_points4=np.float32([[x/k,y/k,z/k] for [x,y,z,k] in rec_points4])
	cont_neg_4=0
	for point in rec_points4:
		if(point[2]<0):
			cont_neg_4+=1

	negative_depths=np.array([cont_neg_1,cont_neg_2,cont_neg_3,cont_neg_4])
	#devuelve E,R,t
	return E,combinations[np.argmin(negative_depths)][0],combinations[np.argmin(negative_depths)][1]

"""


"""
def calculateCameraMotion(K,F,points1,points2):
	npoints=len(points1)

	if(npoints==len(points2)):
		E = np.transpose(K).dot(F).dot(K)
		D,U,Vt=cv2.SVDecomp(E)
		t=np.transpose(U)[-1]
		t1=t
		t2=-t
		#Calculo de las matrices de rotacion
		w=[[0,-1,0],[1,0,0],[0,0,1]]
		R1=U.dot(np.transpose(w)).dot(Vt)				#[UW^TV^T]
		R2=U.dot(np.transpose(w)).dot(np.transpose(Vt))	#[UW^TV]

		#Ver qué camara es la buena
		# I=np.eye(3)
		# P=K.dot(np.float32([np.append(I[i],0) for i in xrange(len(I))]))
		# P1=K.dot(np.float32([np.append(np.copy(R1[i]),t1[i]) for i in xrange(len(R1))]))
		# P2=K.dot(np.float32([np.append(np.copy(R1[i]),t2[i]) for i in xrange(len(R1))]))
		# P3=K.dot(np.float32([np.append(np.copy(R2[i]),t1[i]) for i in xrange(len(R2))]))
		# P4=K.dot(np.float32([np.append(np.copy(R2[i]),t2[i]) for i in xrange(len(R2))]))

		combinations=[[R1,t1],[R1,t2],[R2,t1],[R2,t2]]
		pts1=[[x,y,1.0] for [x,y] in points1]
		pts2=[[x,y,1.0] for [x,y] in points2]

		#Sistema para la combinacion [R1|t1]
		system=np.zeros( (npoints,npoints+1) )
		for i in xrange(npoints):
			system[i][i]=sum(np.float32([np.cross(pts2[i],R1.dot(pts1[i]))]).ravel())
			system[i][-1]=sum(np.cross(pts2[i],t1))
		
		D,U,Vt=cv2.SVDecomp(system)
		min_index=np.argmin(D)
		solC1=Vt[min_index]
		cont_neg_1=len(solC1[solC1<0])


		#Sistema para la combinacion [R1|t2]
		system=np.zeros( (npoints,npoints+1) )
		for i in xrange(npoints):
			system[i][i]=sum(np.float32([np.cross(pts2[i],R1.dot(pts1[i]))]).ravel())
			system[i][-1]=sum(np.cross(pts2[i],t2))
		
		D,U,Vt=cv2.SVDecomp(system)
		min_index=np.argmin(D)
		solC2=Vt[min_index]
		cont_neg_2=len(solC2[solC2<0])


		#Sistema para la combinacion [R2|t1]
		system=np.zeros( (npoints,npoints+1) )
		for i in xrange(npoints):
			system[i][i]=sum(np.float32([np.cross(pts2[i],R2.dot(pts1[i]))]).ravel())
			system[i][-1]=sum(np.cross(pts2[i],t1))
		
		D,U,Vt=cv2.SVDecomp(system)
		min_index=np.argmin(D)
		solC3=Vt[min_index]
		cont_neg_3=len(solC3[solC3<0])

		#Sistema para la combinacion [R2|t2]
		system=np.zeros( (npoints,npoints+1) )
		for i in xrange(npoints):
			system[i][i]=sum(np.float32([np.cross(pts2[i],R2.dot(pts1[i]))]).ravel())
			system[i][-1]=sum(np.cross(pts2[i],t2))
		
		D,U,Vt=cv2.SVDecomp(system)
		min_index=np.argmin(D)
		solC4=Vt[min_index]
		cont_neg_4=len(solC4[solC4<0])

		negative_depths=np.array([cont_neg_1,cont_neg_2,cont_neg_3,cont_neg_4])
		print(negative_depths)
		#devuelve E,R,t
		return E,combinations[np.argmin(negative_depths)][0],combinations[np.argmin(negative_depths)][1]
"""



"""
Calcula el movimiento (R,t) de una camara
a partir de sus parametros intrinsecos,
la matriz fundamental y puntos en correspondencia
"""
def calculateCameraMotion(K,F,points1,points2):
	npoints=len(points1)

	if(npoints==len(points2)):
		E = np.transpose(K).dot(F).dot(K)
		D,U,Vt=cv2.SVDecomp(E)

		#Cálculo de los vectores de translacion
		t=np.transpose(U)[-1]
		t1=t
		t2=-t

		#Calculo de las matrices de rotacion
		w=[[0,-1,0],[1,0,0],[0,0,1]]
		R1=U.dot(np.transpose(w)).dot(Vt)				#[UW^TV^T]
		R2=U.dot(np.transpose(w)).dot(np.transpose(Vt))	#[UW^TV]

		#Ver qué camara es la buena
		combinations=[[R1,t1],[R1,t2],[R2,t1],[R2,t2]]
		pts1=[[x,y,1.0] for [x,y] in points1]
		pts2=[[x,y,1.0] for [x,y] in points2]

		#Sistema para la combinacion [R1|t1]
		system=np.zeros( (npoints,npoints+1) )
		for i in xrange(npoints):
			system[i][i]=sum(np.float32([np.cross(pts2[i],R1.dot(pts1[i]))]).ravel())
			system[i][-1]=sum(np.cross(pts2[i],t1))
		
		D,U,Vt=cv2.SVDecomp(system)
		min_index=np.argmin(D)
		solC1=Vt[min_index]
		cont_neg_1=len(solC1[solC1<0])


		#Sistema para la combinacion [R1|t2]
		system=np.zeros( (npoints,npoints+1) )
		for i in xrange(npoints):
			system[i][i]=sum(np.float32([np.cross(pts2[i],R1.dot(pts1[i]))]).ravel())
			system[i][-1]=sum(np.cross(pts2[i],t2))
		
		D,U,Vt=cv2.SVDecomp(system)
		min_index=np.argmin(D)
		solC2=Vt[min_index]
		cont_neg_2=len(solC2[solC2<0])


		#Sistema para la combinacion [R2|t1]
		system=np.zeros( (npoints,npoints+1) )
		for i in xrange(npoints):
			system[i][i]=sum(np.float32([np.cross(pts2[i],R2.dot(pts1[i]))]).ravel())
			system[i][-1]=sum(np.cross(pts2[i],t1))
		
		D,U,Vt=cv2.SVDecomp(system)
		min_index=np.argmin(D)
		solC3=Vt[min_index]
		cont_neg_3=len(solC3[solC3<0])

		#Sistema para la combinacion [R2|t2]
		system=np.zeros( (npoints,npoints+1) )
		for i in xrange(npoints):
			system[i][i]=sum(np.float32([np.cross(pts2[i],R2.dot(pts1[i]))]).ravel())
			system[i][-1]=sum(np.cross(pts2[i],t2))
		
		D,U,Vt=cv2.SVDecomp(system)
		min_index=np.argmin(D)
		solC4=Vt[min_index]
		cont_neg_4=len(solC4[solC4<0])

		negative_depths=np.array([cont_neg_1,cont_neg_2,cont_neg_3,cont_neg_4])
		#devuelve E,R,t
		index_valid_sol=np.argmin(negative_depths)
		return E,combinations[index_valid_sol][0],combinations[index_valid_sol][1]
	else:
		raise IndexError, "Both lists of points must have the same lenght."









































def rectifyImages(imgLeft,KLeft,distLeft,imgRight,KRight,distRight,R_lr,T_lr):
	R1, R2, P1, P2, Q, roi1, roi2=cv2.stereoRectify(KLeft,distLeft,KRight,distRight,imgLeft.shape[:2],R_lr,T_lr)

	max_size_left=np.max(np.array(imgLeft.shape[:2]))
	max_size_right=np.max(np.array(imgRight.shape[:2]))

	left_maps = cv2.initUndistortRectifyMap(KLeft, distLeft, R1, P1, imgLeft.shape[:2], cv2.CV_16SC2)
	right_maps = cv2.initUndistortRectifyMap(KRight, distRight, R2, P2, imgRight.shape[:2], cv2.CV_16SC2)

	left_img_remap = cv2.remap(imgLeft, left_maps[0], left_maps[1], cv2.INTER_NEAREST)
	right_img_remap = cv2.remap(imgRight, right_maps[0], right_maps[1], cv2.INTER_NEAREST)

	return left_img_remap,right_img_remap












############################################################################################################################################################################
############################################################################################################################################################################
#########################################################################           MAIN           #########################################################################
############################################################################################################################################################################
############################################################################################################################################################################



#	print("APARTADO A.1)")
#	print("A continuacion vamos a ver la mascara normalizada con sigma=2")
#	print(getMask(2))
#	raw_input("Pulse intro para continuar...")
#	print("-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.")





if __name__=='__main__':
	#Fijar la semilla para los experimentos aleatorios
	np.random.seed(42)

#APARTADOS 1a,1b,1c	
	print("APARTADO 1.a)")
	print("A continuacion vemos la camara generada aleatoriamente: ")
	
	P=generateCamera()
	grid=generatePointGrid()
	print(P)
	print("-----------------------------")
	print("Pulse intro para continuar...")
	print("-----------------------------")
	raw_input("")

	print("APARTADO 1.b, 1.c)")
	print("A continuacion vemos los puntos en alzado y proyectados con la cámara aleatoria: ")

	grid2d=[(x,y) for (x,y,z) in grid]
	im_front_points=drawProjectedGrid(grid2d,100,100,500)
	
	projected_points=projectPoints(P,grid)
	im_proj_points=drawProjectedGrid(projected_points,100,100,500)

	print("--------------------------------")
	print("Cierre la ventana para continuar")
 	print("--------------------------------")

	paintMatrixImages([[im_front_points,im_proj_points]],[["Puntos en vista alzado","Puntos proyectados"]],"Cámara")
	

#APARTADO 1d
	print("APARTADO 1.d)")
	print("A continuacion comparamos la cámara original con la estimada mediante el algoritmo DLT: ")

	P_estimated=estimate_camera(grid,projected_points)

	print("Cámara original:")
	print(P)
	print("Cámara estimada:")
	print(P_estimated)
	print("Constante que multiplica a la estimada para obtener la original: {}".format(P[0][0]/P_estimated[0][0]))
	print("-----------------------------")
	print("Pulse intro para continuar...")
	print("-----------------------------")
	raw_input("")

#APARTADO 1e
	print("APARTADO 1.e)")
	print("Error entre ambas cámaras, calculado con la norma de Frobenius: ")
	print("Error entre las cámaras: {}".format(frobenius_norm( normalizeMatrix(np.abs(P),0,1) - normalizeMatrix(np.abs(P_estimated),0,1)   )))
	print("-----------------------------")
	print("Pulse intro para continuar...")
	print("-----------------------------")
	raw_input("")
#APARTADO 1f
	print("APARTADO 1.f)")
	print("A continuación comparamos las proyecciones que realizan la cámara original contra la estimada")	
	projected_points=projectPoints(P_estimated,grid)
	im_proj_points_est=drawProjectedGrid(projected_points,100,100,500)
	print("-----------------------------------")
	print("Cierre la ventana para continuar...")
	print("-----------------------------------")
	paintMatrixImages([[im_proj_points,im_proj_points_est]],[["Camara real","Camara estimada"]],"Cámara")












#APARTADO 2a
	print("APARTADO 2.a) (I)")
	print("A continuación veremos las imágenes válidas para calibrar: ")
	imageNames=[
				"imagenes/Image1.tif",
				"imagenes/Image2.tif",
				"imagenes/Image3.tif",
				"imagenes/Image4.tif",
				"imagenes/Image5.tif",
				"imagenes/Image6.tif",
				"imagenes/Image7.tif",
				"imagenes/Image8.tif",
				"imagenes/Image9.tif",
				"imagenes/Image10.tif",
				"imagenes/Image11.tif",
				"imagenes/Image12.tif",
				"imagenes/Image13.tif",
				"imagenes/Image14.tif",
				"imagenes/Image15.tif",
				"imagenes/Image16.tif",
				"imagenes/Image17.tif",
				"imagenes/Image18.tif",
				"imagenes/Image19.tif",
				"imagenes/Image20.tif",
				"imagenes/Image21.tif",
				"imagenes/Image22.tif",
				"imagenes/Image23.tif",
				"imagenes/Image24.tif",
				"imagenes/Image25.tif",
				]

	imgpoints,validimgnames,images,correctedImagesNoDist,correctedImagesDist,croppedImagesDist,errorNoDist,errorOnlyRad,errorOnlyTan,errorDist=selectImagesAndCalibrate(imageNames,(12,11),(11,11)) #Corrigiendo distorsion
	print("-----------------------------------")
	print("Cierre la ventana para continuar...")
	print("-----------------------------------")
	paintMatrixImages([images[0:2],images[2:4]],[validimgnames[0:2],validimgnames[2:4]],"Imagenes validas")
	paintMatrixImages([images[4:6],images[6:7]],[validimgnames[4:6],validimgnames[6:7]],"Imagenes validas")
	
	print("APARTADO 2.a) (II)")
	print("A continuación vemos todos los puntos detectados desde la misma cámara: ")

	origen=np.float32([(x,y,0) for i in xrange(len(imgpoints)) for [[x,y]] in imgpoints[i]])
	paintMatrixImages([[drawAllChessboardPlanes(origen,len(imgpoints),10,10,1)]],[["Todos los planos desde la misma camara"]],"Todos los tableros juntos")

	print("-----------------------------------")
	print("Cierre la ventana para continuar...")
	print("-----------------------------------")


	print("APARTADO 2.a) (III)")
	print("A continuación veremos los errores suponiendo las diferentes distorsiones: ")

	print("Error en la calibracion suponiendo que no hay distorsion: {}".format(errorNoDist))
	print("Error en la calibracion suponiendo solo distorsion tangencial: {}".format(errorOnlyTan))
	print("Error en la calibracion suponiendo solo distorsion radial: {}".format(errorOnlyRad))
	print("Error en la calibracion suponiendo que hay ambas distorsiones: {}".format(errorDist))

	print("-----------------------------")
	print("Pulse intro para continuar...")
	print("-----------------------------")
	raw_input("")

	

#Apartado 2b
	print("APARTADO 2.b)")
	print("A continuación vemos algunos ejemplos de las imágenes de calibración con la distorsión corregida: ")
	paintMatrixImages(	[
							[correctedImagesNoDist[0] , correctedImagesDist[0], croppedImagesDist[0]],
							[correctedImagesNoDist[2] , correctedImagesDist[2], croppedImagesDist[2]],
							[correctedImagesNoDist[5] , correctedImagesDist[5], croppedImagesDist[5]]
						],
						[
							["No corrige distorsion","Distorsion corregida","Corregida final"],
							["No corrige distorsion","Distorsion corregida","Corregida final"],
							["No corrige distorsion","Distorsion corregida","Corregida final"]
						],"Correccion de la distorsion.")

	print("-----------------------------------")
	print("Cierre la ventana para continuar...")
	print("-----------------------------------")












#Apartado 3a
	print("APARTADO 3.a)")
	print("A continuación vemos el número de puntos en correspondencia detectados por cada detector. Nos quedamos con el que más puntos detecte: ")

	img1=loadImage("imagenes/Vmort1.pgm","COLOR")
	img2=loadImage("imagenes/Vmort2.pgm","COLOR")

	imgresultORB, points1ORB, points2ORB=findCorrespondencesORB(img1,img2,0.8)
	imgresultBRISK, points1BRISK, points2BRISK=findCorrespondencesBRISK(img1,img2,0.8)
	imgresultAKAZE, points1AKAZE, points2AKAZE=findCorrespondencesAKAZE(img1,img2,0.8)

	print("Puntos encontrados por ORB: {}".format(len(points1ORB)))
	print("Puntos encontrados por BRISK: {}".format(len(points1BRISK)))
	print("Puntos encontrados por AKAZE: {}".format(len(points1AKAZE)))	

	print("-----------------------------")
	print("Pulse intro para continuar...")
	print("-----------------------------")
	raw_input("")

#Apartado 3b
	print("APARTADO 3.b)")
	print("A continuación vemos la matriz fundamental calculada con los puntos en correspondencia por algoritmo 8-POINT+RANSAC: ")

	F, mask = cv2.findFundamentalMat(points1BRISK,points2BRISK,method=cv2.FM_RANSAC+cv2.FM_8POINT,param1=0.1)
	print("Matriz fundamental: ")
	print(F)
	
	print("-----------------------------")
	print("Pulse intro para continuar...")
	print("-----------------------------")
	raw_input("")
#Apartado 3c
	print("APARTADO 3.c)")
	print("A continuación vemos algunas de las líneas epipolares calculadas: ")

	points1,points2=getFineCorrespondences(mask,points1BRISK,points2BRISK)

	lines1 = cv2.computeCorrespondEpilines(points2, 2,F).reshape(-1,3)
	lines2 = cv2.computeCorrespondEpilines(points1, 1,F).reshape(-1,3)

	line_points1=[ [(0, int(-eq[2]/eq[1]) ), (img1.shape[1], int(-(eq[2]+eq[0]*img1.shape[1])/eq[1] ))]  for eq in lines1]
	line_points2=[ [(0, int(-eq[2]/eq[1]) ), (img2.shape[1], int(-(eq[2]+eq[0]*img1.shape[1])/eq[1] ))]  for eq in lines2]

	#Con [::10] pinta solo una de cada 10
	for points,point in zip(line_points1[::10],points1[::10]):
		color=tuple(np.random.randint(0,255,3).tolist())
		cv2.line(img1, points[0], points[1], color,1)
		cv2.circle(img1,(int(point[0]),int(point[1])),5,color,-1)

	for points,point in zip(line_points2[::10],points2[::10]):
		color=tuple(np.random.randint(0,255,3).tolist())
		cv2.line(img2, points[0], points[1], color,1)
		cv2.circle(img2,(int(point[0]),int(point[1])),5,color,-1)

	print("-----------------------------------")
	print("Cierre la ventana para continuar...")
	print("-----------------------------------")
	paintMatrixImages([[img1,img2]],[["imagen1","imagen2"]],"Rectas epipolares")

#Apartado 3d
	print("APARTADO 3.d)")
	print("A continuación vemos el error medio en cada imagen en cuanto a las rectas epipolares y los puntos: ")

	print("Media de error en la imagen 1: {}".format(calculateMeanError(points1,lines1)))
	print("Media de error en la imagen 2: {}".format(calculateMeanError(points2,lines2)))
	print("Media de error en ambas imagenes: {}".format((calculateMeanError(points1,lines1)+calculateMeanError(points2,lines2))/2))

	print("-----------------------------")
	print("Pulse intro para continuar...")
	print("-----------------------------")
	raw_input("")











#Apartado 4a

	rdimage000=loadImage("imagenes/rdimage.000.ppm","COLOR")
	rdimageK				=	np.float32(	[
											[1839.6300000000001091,0,1024.2000000000000455],
											[0,1848.0699999999999363,686.5180000000000291],
											[0,0,1]
											])

	rdimage000distortion	=	np.float32([0,0,0])
	
	rdimage000rotation		=	np.float32(	[
											[0.99989455260937387671, 0.0049395902289571594346, 0.013655918514328903648 ],
											[-0.0048621000000000002758, 0.99997192395009415478, -0.0057018676884779640954 ],
											[-0.01368370000000000003, 0.0056348700000000001564, 0.99989049630166648708 ]
											])
	rdimage000translation	=	np.float32([-0.24386599999999999944, 0.2348540000000000072, 0.44275100000000000566])


	rdimage001=loadImage("imagenes/rdimage.001.ppm","COLOR")
	

	rdimage001distortion	=	np.float32([0,0,0])
	
	rdimage001rotation		=	np.float32(	[
											[0.99975480066385657985, 0.019390648831902848603, 0.010693048557372427862],
											[-0.019846499999999999558, 0.99882047750746361103, 0.044314446284616657024],
											[-0.0098211500000000007127, -0.044515800000000001202, 0.99896040390149476451]
											])
	rdimage001translation	=	np.float32([-0.35059299999999998798, -0.70982999999999996099, 0.75288600000000005519])

	rdimage004=loadImage("imagenes/rdimage.004.ppm","COLOR")
	
	rdimage004distortion	=	np.float32([0,0,0])
	rdimage004rotation		=	np.float32([
											[0.99996050522707646824, 0.008791445647177747319, -0.0013032534069471815585 ],
											[-0.0087515400000000003605, 0.99957003802442112583, 0.027984810728066047275 ],
											[0.0015487199999999999578 ,-0.027972299999999998554, 0.99960749892098743619],
											])
	rdimage004translation	=	np.float32([-0.86724299999999998612, -0.17648199999999999998, 3.9854500000000001592])

#Apartado 4b
	print("APARTADO 4.a, 4.b)")
	print("Los datos de cámara, distorsión, rotación y translación de cada imagen han sido cargados.")
	print("A continuación usamos AKAZE para encontrar correspondencias entre cada par de imágenes: ")
	
	imgresult_01, points1_01, points2_01=findCorrespondencesAKAZE(rdimage000,rdimage001,0.8)
	imgresult_04, points1_04, points2_04=findCorrespondencesAKAZE(rdimage000,rdimage004,0.8)
	imgresult_14, points1_14, points2_14=findCorrespondencesAKAZE(rdimage001,rdimage004,0.8)

	print("Se han encontrado {} puntos entre 0 y 1.".format(len(points1_01)))
	print("Se han encontrado {} puntos entre 0 y 4.".format(len(points1_04)))
	print("Se han encontrado {} puntos entre 1 y 4.".format(len(points1_14)))

	print("-----------------------------")
	print("Pulse intro para continuar...")
	print("-----------------------------")
	raw_input("")
#Apartado 4c
	print("APARTADO 4.c (I)")
	print("A continuación vemos las matrices fundamentales calculadas entre cada par de imagenes: ")

	F_01, mask_01 = cv2.findFundamentalMat(points1_01,points2_01,method=cv2.FM_RANSAC+cv2.FM_8POINT,param1=0.1)
	F_04, mask_04 = cv2.findFundamentalMat(points1_04,points2_04,method=cv2.FM_RANSAC+cv2.FM_8POINT,param1=0.1)
	F_14, mask_14 = cv2.findFundamentalMat(points1_14,points2_14,method=cv2.FM_RANSAC+cv2.FM_8POINT,param1=0.1)

	points1_01,points2_01 = getFineCorrespondences(mask_01,points1_01,points2_01)
	points1_04,points2_04 = getFineCorrespondences(mask_04,points1_04,points2_04)
	points1_14,points2_14 = getFineCorrespondences(mask_14,points1_14,points2_14)

	print("Matriz fundamental entre 0 y 1")
	print(F_01)
	print("Matriz fundamental entre 0 y 4")
	print(F_04)
	print("Matriz fundamental entre 1 y 4")
	print(F_14)

	print("-----------------------------")
	print("Pulse intro para continuar...")
	print("-----------------------------")
	raw_input("")

	#print("Error en la bondad de la matriz fundamental entre 0 y 1: {}".format(checkCorrectFundamentalMatrix(F_01,mask_01,points1_01,points2_01)[0,0]))
	#print("Error en la bondad de la matriz fundamental entre 0 y 4: {}".format(checkCorrectFundamentalMatrix(F_04,mask_04,points1_04,points2_04)[0,0]))
	#print("Error en la bondad de la matriz fundamental entre 1 y 4: {}".format(checkCorrectFundamentalMatrix(F_14,mask_14,points1_14,points2_14)[0,0]))

	print("APARTADO 4.c (II)")
	print("A continuación vemos los movimientos calculados entre cada par de imagenes: ")

	E_01,R_01,T_01=calculateCameraMotion(rdimageK,F_01,points1_01,points2_01)
	E_04,R_04,T_04=calculateCameraMotion(rdimageK,F_04,points1_04,points2_04)
	E_14,R_14,T_14=calculateCameraMotion(rdimageK,F_14,points1_14,points2_14)

	np.set_printoptions(suppress=True)#Para que no salga en notación científica

	print("-------------Matriz esencial y movimiento entre las imagenes 0 y 1-------------")
	print("\tMatriz esencial: ")
	print(E_01)
	print("\tMatriz de rotacion: ")
	print(R_01)
	print("\tVector de translacion: ")
	print(T_01)
	
	print("-------------Matriz esencial y movimiento entre las imagenes 0 y 4-------------")
	print("\tMatriz esencial: ")
	print(E_04)
	print("\tMatriz de rotacion: ")
	print(R_04)
	print("\tVector de translacion: ")
	print(T_04)
	
	print("-------------Matriz esencial y movimiento entre las imagenes 1 y 4-------------")
	print("\tMatriz esencial: ")
	print(E_14)
	print("\tMatriz de rotacion: ")
	print(R_14)
	print("\tVector de translacion: ")
	print(T_14)
	
	print("-----------------------------")
	print("Pulse intro para continuar...")
	print("-----------------------------")
	raw_input("")











#Apartado 5a

	print("APARTADO 5.a, 5.b, 5.c")
	print("A continuación vemos los puntos calculados por cada par de imagenes: ")
	print("-----------------------------------")
	print("Cierre la ventana para continuar...")
	print("-----------------------------------")
	
	#DETECCION ENTRE 0 Y 1
	imgresult_01, points1_01, points2_01=findCorrespondencesAKAZE(rdimage000,rdimage001,0.8)
	P=rdimageK.dot(np.float32([np.append(np.copy( rdimage000rotation[i]),rdimage000translation[i]) for i in xrange(len(rdimage000rotation))]))
	P1=rdimageK.dot(np.float32([np.append(np.copy( rdimage001rotation[i]),rdimage001translation[i]) for i in xrange(len(rdimage001rotation))]))

	world_points=np.float32([cv2.triangulatePoints(P, P1, points1_01[i], points2_01[i]) for i in xrange(len(points1_01))])
	world_points=np.float32([ [x/k,y/k,z/k] for [x,y,z,k] in world_points])
	pix_points=np.float32([ [int(1000*(x/z)),int(1000*(y/z))] for [x,y,z] in world_points])

	canvas=giveCanvas(pix_points,10,10,5)
	rdimage000copy=np.copy(rdimage000)
	rdimage001copy=np.copy(rdimage001)

	for point,pz,p_real_1,p_real_2 in zip(pix_points,world_points,points1_01,points2_01):
		cv2.circle(canvas,(point[0],point[1]), 1, (int(1000*pz[2])%255,int(1000*pz[2])%255,int(1000*pz[2])%255), 2)
		cv2.circle(rdimage000copy,(int(p_real_1[0]),int(p_real_1[1])), 2, (int(1000*pz[2])%255,int(1000*pz[2])%255,int(1000*pz[2])%255), 10)
		cv2.circle(rdimage001copy,(int(p_real_2[0]),int(p_real_2[1])), 2, (int(1000*pz[2])%255,int(1000*pz[2])%255,int(1000*pz[2])%255), 10)
	
	paintMatrixImages([[rdimage000copy,rdimage001copy,quitLeftOver(canvas)]],[["0","1","puntos"]],"Puntos detectados para reconstruccion")


	resshape=quitLeftOver(canvas).shape[0:2]

	#DETECCION ENTRE 0 Y 4
	imgresult_04, points1_04, points2_04=findCorrespondencesAKAZE(rdimage000,rdimage004,0.8)
	P=rdimageK.dot(np.float32([np.append(np.copy( rdimage000rotation[i]),rdimage000translation[i]) for i in xrange(len(rdimage000rotation))]))
	P1=rdimageK.dot(np.float32([np.append(np.copy( rdimage004rotation[i]),rdimage004translation[i]) for i in xrange(len(rdimage004rotation))]))

	world_points=np.float32([cv2.triangulatePoints(P, P1, points1_04[i], points2_04[i]) for i in xrange(len(points1_04))])
	world_points=np.float32([ [x/k,y/k,z/k] for [x,y,z,k] in world_points])
	pix_points=np.float32([ [int(1000*(x/z)),int(1000*(y/z))] for [x,y,z] in world_points])

	canvas=giveCanvas(pix_points,10,10,5)
	rdimage000copy=np.copy(rdimage000)
	rdimage004copy=np.copy(rdimage004)

	for point,pz,p_real_1,p_real_2 in zip(pix_points,world_points,points1_01,points2_01):
		cv2.circle(canvas,(point[0],point[1]), 1, (int(1000*pz[2])%255,int(1000*pz[2])%255,int(1000*pz[2])%255), 2)
		cv2.circle(rdimage000copy,(int(p_real_1[0]),int(p_real_1[1])), 2, (int(1000*pz[2])%255,int(1000*pz[2])%255,int(1000*pz[2])%255), 10)
		cv2.circle(rdimage004copy,(int(p_real_2[0]),int(p_real_2[1])), 2, (int(1000*pz[2])%255,int(1000*pz[2])%255,int(1000*pz[2])%255), 10)
	
	paintMatrixImages([[rdimage000copy,rdimage004copy,quitLeftOver(canvas)[0:resshape[0],0:resshape[1]]]],[["0","4","puntos"]],"Puntos detectados para reconstruccion")

	#DETECCION ENTRE 1 Y 4
	imgresult_14, points1_14, points2_14=findCorrespondencesAKAZE(rdimage001,rdimage004,0.8)
	P=rdimageK.dot(np.float32([np.append(np.copy( rdimage001rotation[i]),rdimage001translation[i]) for i in xrange(len(rdimage001rotation))]))
	P1=rdimageK.dot(np.float32([np.append(np.copy( rdimage004rotation[i]),rdimage004translation[i]) for i in xrange(len(rdimage004rotation))]))

	world_points=np.float32([cv2.triangulatePoints(P, P1, points1_14[i], points2_14[i]) for i in xrange(len(points1_14))])
	world_points=np.float32([ [x/k,y/k,z/k] for [x,y,z,k] in world_points])
	pix_points=np.float32([ [int(1000*(x/z)),int(1000*(y/z))] for [x,y,z] in world_points])

	canvas=giveCanvas(pix_points,10,10,5)
	rdimage001copy=np.copy(rdimage001)
	rdimage004copy=np.copy(rdimage004)

	for point,pz,p_real_1,p_real_2 in zip(pix_points,world_points,points1_14,points2_14):
		cv2.circle(canvas,(point[0],point[1]), 1, (int(1000*pz[2])%255,int(1000*pz[2])%255,int(1000*pz[2])%255), 2)
		cv2.circle(rdimage001copy,(int(p_real_1[0]),int(p_real_1[1])), 2, (int(1000*pz[2])%255,int(1000*pz[2])%255,int(1000*pz[2])%255), 10)
		cv2.circle(rdimage004copy,(int(p_real_2[0]),int(p_real_2[1])), 2, (int(1000*pz[2])%255,int(1000*pz[2])%255,int(1000*pz[2])%255), 10)
	
	paintMatrixImages([[rdimage001copy,rdimage004copy,quitLeftOver(canvas)[0:resshape[0],0:resshape[1]]]],[["1","4","puntos"]],"Puntos detectados para reconstruccion")
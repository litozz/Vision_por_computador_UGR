#!/usr/bin/env python3
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














############################################################################################################################################################################
############################################################################################################################################################################
#########################################################################           MAIN           #########################################################################
############################################################################################################################################################################
############################################################################################################################################################################


if __name__=='__main__':
	img1=loadImage("imagenes/Yosemite1.jpg","COLOR")
	piry1=getPyramid(img1,1,1,3)
	blockSize=5
	kSize=7
	escala=0
	sizeSupresion=9
	radioCirculo=8


#HITO 1: COMPARACION DE HARRIS DE OPENCV CON EL MIO
	
 	print("Preparando el siguiente apartado...")
 	print("Apartado 1.a (1) Comparación del resultado del detector de esquinas Harris programado con el de OpenCV.")
 	
	r=cv2.cvtColor(piry1[escala],cv2.COLOR_RGB2GRAY)
	cornerr=cv2.cornerHarris(r,blockSize,kSize,0.04)
	haropencv=cv2.merge([cornerr,cornerr,cornerr])

	miharris=harrisCornerDetector(piry1[escala],blockSize,kSize,0.04)
	print("----------------------------------------------------------------------------------------------------")
	print("Cierre la ventana para continuar")
 	print("----------------------------------------------------------------------------------------------------")
	
	paintMatrixImages([[img1,haropencv,miharris]],
					  [["Original","Harris Corner Detector OpenCV", "Mi Harris Corner Detector"]],
					  "Comparacion Harris OpenCV vs. mi Harris.- José Carlos Martínez Velázquez")
	
 	
 	

#HITO 2: SUPRESION DE NO MAXIMOS
	print("Preparando el siguiente apartado...")
	print("Apartado 1.a (2) Supresión de No máximos.")
 	
	#miharris=harrisCornerDetector(piry1[escala],blockSize,kSize,0.04) #SOBRA
	nms=nonMaximumSuppression(miharris,21,0.05)
	r=cv2.cvtColor(nms,cv2.COLOR_RGB2GRAY)
 	for i in xrange(len(r)):
 		for j in xrange(len(r[i])):
 			if(r[i][j]==255):
 				cv2.circle(piry1[escala],(j,i), 3, (255,0,0), 2)
 	print("----------------------------------------------------------------------------------------------------")
 	print("Cierre la ventana para continuar")
 	print("----------------------------------------------------------------------------------------------------")

	paintMatrixImages([[nms,piry1[escala]]],
 					  [["NMS", "Maximos en imagen original"]],
 					  "Supresion de no maximos.- José Carlos Martínez Velázquez")
	
 	

#HITO 3: BUSQUEDA DE PUNTOS EN LAS DIFERENTES ESCALAS
	print("Preparando el siguiente apartado...")
	print("Apartado 1.a (3) Puntos detectados en cada escala.")
 	
	img1=loadImage("imagenes/Yosemite1.jpg","COLOR")
	piry1=getPyramid(img1,1,1,3)
	for i_escala in xrange(len(piry1)):
		miharris=harrisCornerDetector(piry1[i_escala],blockSize,kSize,0.04)
		nms=nonMaximumSuppression(miharris,5,0.05)
		r,g,b=cv2.split(nms)
		for i in xrange(len(r)):
			for j in xrange(len(r[i])):
				if(r[i][j]==255):
					cv2.circle(piry1[i_escala],(j,i), 1, (255,0,0), 2)
	print("----------------------------------------------------------------------------------------------------")
	print("Cierre la ventana para continuar")
	showPyramid(piry1,"")
	print("----------------------------------------------------------------------------------------------------")
 	



#HITO 3.1: OBTENCION DE LOS MEJORES PUNTOS, COMPARACION DE LOS DOS CRITERIOS
	print("Preparando el siguiente apartado...")
	print("Apartado 1.a (4) Comparación de criterios Ponderado vs. Absoluto para 100 puntos.")
 	
	img1=loadImage("imagenes/Yosemite1.jpg","COLOR")
	piry1=getPyramid(img1,1,1,3)
 
 	
 	imgponderado=np.copy(img1)
 	imgabsoluto=np.copy(img1)

 	pointsponderado=getStrongestPoints(piry1,blockSize,kSize,sizeSupresion,0.05,100,weights=[0.7,0.2,0.1])
 	imgponderado=paintPointsInImage(imgponderado,pointsponderado)
	
 	pointsabsoluto=getStrongestPoints(piry1,blockSize,kSize,sizeSupresion,0.05,100,weights=[])
 	imgabsoluto=paintPointsInImage(imgabsoluto,pointsabsoluto)
 	print("----------------------------------------------------------------------------------------------------")
	print("Cierre la ventana para continuar")
  	print("----------------------------------------------------------------------------------------------------")
 	
 	paintMatrixImages([[imgabsoluto,imgponderado]],
  					  [["Orden absoluto", "Orden ponderado"]],
  					  "Orden de puntos.- José Carlos Martínez Velázquez")
 	
  	


#HITO 3.2: OBTENCION DE LOS 1500 PUNTOS CON CRITERIO PONDERADO
	print("Preparando el siguiente apartado...")
	print("Apartado 1.a (FINAL) Selección de 1500 puntos por el criterio de ponderación.")
 	
	imgponderado=np.copy(img1)

	pointsponderado=getStrongestPoints(piry1,blockSize,kSize,sizeSupresion,0.05,1500,weights=[0.7,0.2,0.1])
	imgponderado=paintPointsInImage(imgponderado,pointsponderado)
	print("----------------------------------------------------------------------------------------------------")
	print("Cierre la ventana para continuar")
 	print("----------------------------------------------------------------------------------------------------")

	paintMatrixImages([[imgponderado]],
 					  [["1500 puntos orden ponderado"]],
 					  "Orden de puntos.- José Carlos Martínez Velázquez")
	
 	




#APARTADO 1.B: REFINAMIENTO DE PUNTOS
	print("Preparando el siguiente apartado...")
	print("Apartado 1.b) Refinamiento de puntos.")
	imgNormal=np.copy(img1)
	imgRefined=np.copy(img1)
	points=getStrongestPoints(piry1,blockSize,kSize,sizeSupresion,0.05,300,weights=[0.7,0.2,0.1])
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
	refinedPoints=refinePoints(piry1,criteria,points,7,(-1,1))
	imgNormal=paintPointsInImage(imgNormal,points)
	imgRefined=paintPointsInImage(imgRefined,refinedPoints)
	print("----------------------------------------------------------------------------------------------------")
	print("Cierre la ventana para continuar")
	print("----------------------------------------------------------------------------------------------------")
	paintMatrixImages([[imgNormal,imgRefined]],
	 					  [["Puntos sin refinar","Puntos refinados"]],
	 					  "Refinamiento de puntos.- José Carlos Martínez Velázquez")








#HITO 4: ORDENAR CON SORTIDX(), REFINAMIENTO, CALCULO DE LA ORIENTACION Y PINTAR RESULTADOS
	yos1=loadImage("imagenes/Yosemite1.jpg","COLOR")
	piry1=getPyramid(yos1,1,1,3)
	yos2=loadImage("imagenes/Yosemite2.jpg","COLOR")
	piry2=getPyramid(yos2,1,1,3)
	blockSize=7
	kSize=7


	print("Preparando el siguiente apartado...")
	print("Apartado 1.c Selección de 100 puntos y representación gráfica de la orientación.")	
	
	criteria 	= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
	
	pointsy1=getStrongestPointsComplete(piry1,blockSize,kSize,7,0.05,100,criteria,7,(-1,1),weights=[0.7,0.2,0.1])
	pointsy2=getStrongestPointsComplete(piry2,blockSize,kSize,7,0.05,100,criteria,7,(-1,1),weights=[0.7,0.2,0.1])
	
	imageorienty1=paintPointsAndOrientationInImage(yos1,pointsy1)
	imageorienty2=paintPointsAndOrientationInImage(yos2,pointsy2)
	print("----------------------------------------------------------------------------------------------------")
	print("Cierre la ventana para continuar")
	print("----------------------------------------------------------------------------------------------------")
	paintMatrixImages([[imageorienty1,imageorienty2]],[["Yosemite1","Yosemite2"]],"Orientacion del vector gradiente.- José Carlos Martínez Velázquez")



#HITO 5 MATCHING DE PUNTOS KAZE/AKAZE
	
	#KAZE
	
	
	kp1,des1=getKeyPointsAndDescriptorsKAZE(yos1)
	kp2,des2=getKeyPointsAndDescriptorsKAZE(yos2)

	print("Preparando el siguiente apartado...")
	print("Apartado 2.a Detector KAZE con DescriptorMatcher opción BruteForce.")
	matches=getMatchesDescriptor(des1,des2)
	imageMatches2a=drawMatches(yos1,kp1,yos2,kp2,matches[0:500])
	print("Se han encontrado {} puntos con KAZE y DescriptorMatcher.".format(len(matches)))	
	print("----------------------------------------------------------------------------------------------------")
	print("Cierre la ventana para continuar")
	print("----------------------------------------------------------------------------------------------------")
	paintMatrixImages([[imageMatches2a]],[["KAZE + DescriptorMatcher BruteForce"]],"Detector KAZE.- José Carlos Martínez Velázquez")


	print("Preparando el siguiente apartado...")
	print("Apartado 2.b Detector KAZE con BFMatcher opción BruteForce sin CrossCheck.")
	matches=getMatchesBF(des1,des2,crosscheck=False)
	imageMatches2b=drawMatches(yos1,kp1,yos2,kp2,matches[0:500])
	print("Se han encontrado {} puntos con KAZE y BFMatcher sin crossCheck.".format(len(matches)))
	print("----------------------------------------------------------------------------------------------------")
	print("Cierre la ventana para continuar")
	print("----------------------------------------------------------------------------------------------------")	
	paintMatrixImages([[imageMatches2b]],[["KAZE + BFMatcher sin Crosscheck"]],"Detector KAZE.- José Carlos Martínez Velázquez")



	print("Preparando el siguiente apartado...")
	print("Apartado 2.c Detector KAZE con BFMatcher opción BruteForce con CrossCheck.")
	matches=getMatchesBF(des1,des2,crosscheck=True)
	imageMatches2c=drawMatches(yos1,kp1,yos2,kp2,matches[0:500])
	print("Se han encontrado {} puntos con KAZE y BFMatcher con crossCheck.".format(len(matches)))
	print("----------------------------------------------------------------------------------------------------")
	print("Cierre la ventana para continuar")
	print("----------------------------------------------------------------------------------------------------")	
	paintMatrixImages([[imageMatches2c]],[["KAZE + BFMatcher con Crosscheck"]],"Detector KAZE.- José Carlos Martínez Velázquez")


	#AKAZE
	kp1,des1=getKeyPointsAndDescriptorsAKAZE(yos1)
	kp2,des2=getKeyPointsAndDescriptorsAKAZE(yos2)
	
	print("Preparando el siguiente apartado...")
	print("Apartado 2.d Detector AKAZE con DescriptorMatcher y BruteForce.")
	matches=getMatchesDescriptor(des1,des2)
	imageMatches2d=drawMatches(yos1,kp1,yos2,kp2,matches[0:500])
	print("Se han encontrado {} puntos con AKAZE opción DescriptorMatcher.".format(len(matches)))	
	print("----------------------------------------------------------------------------------------------------")
	print("Cierre la ventana para continuar")
	print("----------------------------------------------------------------------------------------------------")
	paintMatrixImages([[imageMatches2d]],[["AKAZE + DescriptorMatcher Bruteforce"]],"Detector AKAZE.- José Carlos Martínez Velázquez")


	print("Preparando el siguiente apartado...")
	print("Apartado 2.e Detector AKAZE con BFMatcher opción BruteForce sin CrossCheck.")
	matches=getMatchesBF(des1,des2,crosscheck=False)
	imageMatches2e=drawMatches(yos1,kp1,yos2,kp2,matches[0:500])
	print("Se han encontrado {} puntos con AKAZE y BFMatcher sin crossCheck.".format(len(matches)))
	print("----------------------------------------------------------------------------------------------------")
	print("Cierre la ventana para continuar")
	print("----------------------------------------------------------------------------------------------------")	
	paintMatrixImages([[imageMatches2e]],[["AKAZE + BFMatcher sin Crosscheck"]],"Detector AKAZE.- José Carlos Martínez Velázquez")

	print("Preparando el siguiente apartado...")
	print("Apartado 2.f Detector AKAZE con BFMatcher opción BruteForce con CrossCheck.")
	matches=getMatchesBF(des1,des2,crosscheck=True)
	imageMatches2f=drawMatches(yos1,kp1,yos2,kp2,matches[0:500])
	print("Se han encontrado {} puntos con AKAZE y BFMatcher con crossCheck.".format(len(matches)))
	print("----------------------------------------------------------------------------------------------------")
	print("Cierre la ventana para continuar")
	print("----------------------------------------------------------------------------------------------------")	
	paintMatrixImages([[imageMatches2f]],[["AKAZE + BFMatcher sin Crosscheck"]],"Detector AKAZE.- José Carlos Martínez Velázquez")
	

#HITO 6: Construccion de panoramas de dos imagenes
	
 	print("Preparando el siguiente apartado...")
 	print("Apartado 3.a Construccion de un panorama de dos imagenes.")
 	paintMatrixImages([[constructPanoramaTwoImages(yos1,yos2)]],[["Panorama Yosemite"]],"Panorama con 2 imagenes.- José Carlos Martínez Velázquez")
 	print("----------------------------------------------------------------------------------------------------")
 	print("Cierre la ventana para continuar")
 	print("----------------------------------------------------------------------------------------------------")	
	
# # #HITO 7: Construccion de panoramas de varias imagenes
	print("Preparando el siguiente apartado...")
	print("Apartado 3.b Construccion de un panorama de N imagenes.")
	imagenes=[]
	imagenes.append(loadImage("imagenes/mosaico002.jpg","COLOR"))
	imagenes.append(loadImage("imagenes/mosaico003.jpg","COLOR"))
	imagenes.append(loadImage("imagenes/mosaico004.jpg","COLOR"))
	imagenes.append(loadImage("imagenes/mosaico005.jpg","COLOR"))
	imagenes.append(loadImage("imagenes/mosaico006.jpg","COLOR"))
	imagenes.append(loadImage("imagenes/mosaico007.jpg","COLOR"))
	imagenes.append(loadImage("imagenes/mosaico008.jpg","COLOR"))
	imagenes.append(loadImage("imagenes/mosaico009.jpg","COLOR"))
	imagenes.append(loadImage("imagenes/mosaico010.jpg","COLOR"))
	imagenes.append(loadImage("imagenes/mosaico011.jpg","COLOR"))
	print("----------------------------------------------------------------------------------------------------")
	print("Cierre la ventana para continuar")
	print("----------------------------------------------------------------------------------------------------")
	
	paintMatrixImages([[constructPanoramaMultipleImages(imagenes)]],[["Panorama ETSIIT"]],"Panorama con N imagenes.- José Carlos Martínez Velázquez")

	print("Preparando el siguiente apartado...")
	print("Apartado 3.c Construccion de otro panorama de N imagenes.")
	imagenes=[]
	imagenes.append(loadImage("imagenes/yosemite1.jpg","COLOR"))
	imagenes.append(loadImage("imagenes/yosemite2.jpg","COLOR"))
	imagenes.append(loadImage("imagenes/yosemite3.jpg","COLOR"))
	imagenes.append(loadImage("imagenes/yosemite4.jpg","COLOR"))

	print("----------------------------------------------------------------------------------------------------")
	print("Cierre la ventana para continuar")
	print("----------------------------------------------------------------------------------------------------")
	
	paintMatrixImages([[constructPanoramaMultipleImages(imagenes)]],[["Panorama Yosemite"]],"Panorama con N imagenes.- José Carlos Martínez Velázquez")

	
	





	

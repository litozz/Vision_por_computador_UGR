#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

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



"""
Cuando hacemos el vector auxiliar para los bordes,
la imagen queda en negativo, por lo que es necesario negar el negativo
para volver al color de origen.
"""
def negative(shape):
	nvector=np.zeros( (len(shape),len(shape[0])) )
	for i in xrange(len(nvector)):
		for j in xrange(len(nvector[i])):
			nvector[i][j]=255-shape[i][j]
	return nvector


"""
La funcion normalize mapea un valor en un intervalo conocido al mismo valor
medido en un nuevo intervalo.
"""

def normalize(shape,min_v,max_v):
	if(min_v<max_v):
		maxvalue=256
		minvalue=0
		nshape=np.array([[0]*len(shape[0])]*len(shape))
		for i in xrange(len(shape)):
			for j in xrange(len(shape[0])):
				if(shape[i][j]>maxvalue):
					maxvalue=shape[i][j]
				if(shape[i][j]<minvalue):
					minvalue=shape[i][j]
		for i in xrange(len(shape)):
			for j in xrange(len(shape[0])):
				nshape[i][j]=(shape[i][j] - minvalue)*(max_v-min_v)/(maxvalue-minvalue) + min_v
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

# def my_imGaussConvol(image,sigma,bordertype,only_horizontal=False):
# 	mask=getMask(sigma)
# 	r,g,b=cv2.split(image)
# 	#Trabajamos en modo CV_32FC3 (FLOAT)
# 	r=np.float32(r)
# 	g=np.float32(g)
# 	b=np.float32(b) 

# 	for i in xrange(len(r)):
#  		r[i]=convolution2Vectors(mask,createAuxVector(mask,r[i],bordertype))
#  		g[i]=convolution2Vectors(mask,createAuxVector(mask,g[i],bordertype))
#  		b[i]=convolution2Vectors(mask,createAuxVector(mask,b[i],bordertype))

# #Comentar a partir de aqui para ver la diferencia entre solo horizontal y vertical tambien
# 	if(not only_horizontal):
# 	 	r=np.transpose(r)
# 	 	g=np.transpose(g)
# 	 	b=np.transpose(b)

# 	   	for i in xrange(len(r)):
#   			r[i]=convolution2Vectors(mask,createAuxVector(mask,r[i],bordertype))
#   			g[i]=convolution2Vectors(mask,createAuxVector(mask,g[i],bordertype))
#   			b[i]=convolution2Vectors(mask,createAuxVector(mask,b[i],bordertype))
	 
#    		r=np.transpose(r)
#    		g=np.transpose(g)
#    		b=np.transpose(b)
# #Fin de comentarios
# 	#Regresamos al modo CV_8UC3 (ENTERO)
# 	r=np.uint8(r)
# 	g=np.uint8(g)
# 	b=np.uint8(b)
# 	imgi=cv2.merge([r,g,b]) 
#  	return imgi

def my_imGaussConvol(image,sigma,bordertype,only_horizontal=False):
	mask=getMask(sigma)
	r,g,b=cv2.split(image)
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


"""
La funcion getHighFrequences toma dos veces la misma imagen,
la original y la convolucionada. Resta la convolucionada a
la imagen original y obtenemos los detalles (altas frecuencias).
Se puede aplicar un filtro laplaciano, por lo que se pasa un coeficiente
laplaciano.
"""
def getHighFrequences(image,imageconv,hFfactor):
	r,g,b=cv2.split(image)

	#Trabajamos en modo CV_32FC3 (CALCULOS EN COMA FLOTANTE)
	r=np.float32(r)
	g=np.float32(g)
	b=np.float32(b) 
	
	rc,gc,bc=cv2.split(imageconv)
	
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

	
	for i in xrange(0,nrow):
		r=np.delete(r,i,0) #Quitando arrays en el eje x (quitar filas)
		g=np.delete(g,i,0)
		b=np.delete(b,i,0)

	for i in xrange(0,ncol):
		r=np.delete(r,i,1) #Quitando arrays en el eje y (quitar columnas)
		g=np.delete(g,i,1)
		b=np.delete(b,i,1)

	return cv2.merge([r,g,b])











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
Calcula y pinta la piramide gaussiana de una imagen a level niveles
"""
def showPyramid(imagen,windowtitle,sigma,border,level):
	nrow=len(imagen)
	ncol=len(imagen[0])
	
	min_size=min(nrow,ncol)
	
	if(level<int(math.log(min_size,2))):
		fig = plt.figure()
		fig.canvas.set_window_title(windowtitle)
		
		ax1=plt.subplot2grid((nrow,ncol+ncol/2), (0,0), rowspan=nrow,colspan=ncol)
		plt.xticks([]),plt.yticks([])
		plt.imshow(imagen)
		
		i=0
		rowini=i
		row_span=nrow/2**(i+1)
		col_span=ncol/2**(i+1)

		img1=scaleDownImage(imagen,sigma,border)	
		ax2 = plt.subplot2grid((nrow,ncol+ncol/2), (rowini,ncol), rowspan=row_span,colspan=col_span)
		plt.xticks([]),plt.yticks([])
		plt.imshow(img1)

		for i in xrange(1,level-1):
			rowini+=row_span
			row_span=nrow/2**(i+1)
			col_span=ncol/2**(i+1)

			imgbucle=scaleDownImage(img1,sigma,border)
			plt.subplot2grid((nrow,ncol+ncol/2), (rowini, ncol), rowspan=row_span, colspan=col_span)
			plt.xticks([]),plt.yticks([])
			plt.imshow(imgbucle)
		
			img1=imgbucle
		print("Esta viendo la piramide gaussiana, cierre la ventana para continuar.")
		plt.show()
	else:
		raise ValueError, "Image cannot be scaled down more than "+int(math.log(min_size,2))+" times."

"""-----------------------------------------------------------------------"""











if __name__=='__main__':
#APARTADO A.1) MUESTRA DEL CALCULO DE LA MASCARA
	print("APARTADO A.1)")
	print("A continuacion vamos a ver la mascara normalizada con sigma=2")
	print(getMask(2))
	raw_input("Pulse intro para continuar...")
	print("-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.")

#APARTADO A.2 y A.3: CONVOLUCION DE VECTORES + FUNCIONAMIENTO EN IMAGENES
	print("APARTADO A.2\n")
	
	print("\tAPARTADO A.2.1)")
	print("\tA continuacion se mostrara el resultado de convolucion 1D solo donde fue posible el calculo.")
	mask=getMask(1)
	senial=[1,2,3,4,5,6,7,8,9]
	print("\tsignal:\t"+str(senial))
	print("\tmask:\t"+str(mask))
	print("")
	print("\tresult:\t"+str(convolution2Vectors(mask,senial)))
	raw_input("Pulse intro para continuar...")

	print("-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.")

	print("\tAPARTADO A.2.2)")
	print("\tA continuacion se mostrara el vector auxiliar segun el tipo de borde.")
	print("\tsignal:\t"+str(senial))
	print("\tmask:\t"+str(mask))
	print("")
	print("\tBorde negro:\t"+str(createAuxVector(mask,senial,0)))
	print("\tReflejo:\t"+str(createAuxVector(mask,senial,1)))
	print("\tPrimer y ultimo pixel:\t"+str(createAuxVector(mask,senial,2)))
	raw_input("Pulse intro para continuar...")
 	
	print("-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.")

	print("\tAPARTADO A.2.3)")
	print("\tA continuacion se mostrara el calculo de la convolucion para diferentes rellenos.")
	print("\tsignal:\t"+str(senial))
	print("\tmask:\t"+str(mask))
	print("")
	print("\tSignal borde negro:\t"+str(createAuxVector(mask,senial,0)))
	print("\tSignal reflejo:\t"+str(createAuxVector(mask,senial,1)))
	print("\tSignal primer y ultimo pixel:\t"+str(createAuxVector(mask,senial,2)))
	print("")
	print("\tConvolucion 1D borde negro:\t"+str(convolution2Vectors(mask,createAuxVector(mask,senial,0))))
	print("\tConvolucion 1D reflejo:\t"+str(convolution2Vectors(mask,createAuxVector(mask,senial,1))))
	print("\tConvolucion 1D primer y ultimo pixel:\t"+str(convolution2Vectors(mask,createAuxVector(mask,senial,2))))
	raw_input("Pulse intro para continuar...")

	print("-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.")
 	
	print("APARTADO A.3\n")

 	print("A continuacion se mostrara el proceso de construccion del suavizado, espere un momento...")
 	showSmoothedImage("imagenes/motorcycle.bmp","COLOR",7,1)
 	print("-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.")

	
#APARTADO B+C) 1 IMAGEN HIBRIDA EINSTEIN-MARILYN + PIRAMIDE
 	print("APARTADO B+C) 1")
 	print("Construyendo la imagen hibrida Einstein-Marilyn, espere un momento...")
 	hybrid1=showConstructionHybridImage("imagenes/einstein.bmp","GRAYSCALE",1.8,
 								"imagenes/marilyn.bmp","GRAYSCALE",6,
 								1,1)
 	print("Construyendo la piramide gaussiana Einstein-Marilyn, espere un momento...")
 	showPyramid(hybrid1,"Piramide Gaussiana - Vision por computador - Jose Carlos Martinez Velazquez",1,1,5)
 	print("-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.")
	

#APARTADO B+C) 2 IMAGEN HIBRIDA BICI-MOTO + PIRAMIDE
 	print("APARTADO B+C) 2")
 	print("Construyendo la imagen hibrida Bici-Moto, espere un momento...")
 	hybrid2=showConstructionHybridImage("imagenes/bicycle.bmp","COLOR",1.2,
 								"imagenes/motorcycle.bmp","COLOR",10,
 								1,1)
	print("Construyendo la piramide gaussiana Bici-Moto, espere un momento...")
	showPyramid(hybrid2,"Piramide Gaussiana - Vision por computador - Jose Carlos Martinez Velazquez",1,1,5)
	print("-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.")

#APARTADO B+C) 3 IMAGEN HIBRIDA AVION-AVE + PIRAMIDE
	print("APARTADO B+C) 3")
	print("Construyendo la imagen hibrida Avion-Ave, espere un momento...")
	hybrid3=showConstructionHybridImage("imagenes/plane.bmp","GRAYSCALE",1.7,
								"imagenes/bird.bmp","GRAYSCALE",8,
									1,1)
	print("Construyendo la piramide gaussiana Avion-Ave, espere un momento...")
	showPyramid(hybrid3,"Piramide Gaussiana - Vision por computador - Jose Carlos Martinez Velazquez",1,1,5)
	print("-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.")

#PAPARTADO B+C) 4 IMAGEN HIBRIDA GATO-PERRO + PIRAMIDE
 	print("APARTADO B+C) 4")
 	print("Construyendo la imagen hibrida Gato-Perro, espere un momento...")
 	hybrid4=showConstructionHybridImage("imagenes/cat.bmp","COLOR",2.5,
 								"imagenes/dog.bmp","COLOR",10,
 								1.1,1)
 	print("Construyendo la piramide gaussiana Gato-Perro, espere un momento...")
 	showPyramid(hybrid4,"Piramide Gaussiana - Vision por computador - Jose Carlos Martinez Velazquez",1,1,5)
 	print("-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.")

#APARTADO B+C) 5 IMAGEN HIBRIDA SUBMARINO-PEZ + PIRAMIDE
	print("APARTADO B+C) 5")
	print("Construyendo la imagen hibrida Submarino-Pez, espere un momento...")
	hybrid5=showConstructionHybridImage("imagenes/submarine.bmp","COLOR",2,
								"imagenes/fish.bmp","COLOR",8,
								1,1)
	print("Construyendo la piramide gaussiana Submarino-Pez, espere un momento...")
	showPyramid(hybrid5,"Piramide Gaussiana - Vision por computador - Jose Carlos Martinez Velazquez",1,1,5)
	print("-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.")


	#PRUEBA 0: MOSTRAR LOS DIFERENTES TIPOS DE RELLENO
	# print("A continuacion se mostraran los diferentes tipos de relleno para poder filtrar, espere un momento...")
	# showAllBorders("imagenes/motorcycle.bmp","COLOR",20)

#PRUEBA 1a: MOSTRAR LA CONVOLUCION SOLO EN HORIZONTAL PARA CADA TIPO DE BORDE
	# print("Aplicando convolucion horizontal")
	# image1=loadImage("imagenes/cat.bmp","COLOR")
	# imageborded1=fillImage(image1,7,0)
	# imagesmoothed1=my_imGaussConvol(image1,7,0,True)
	# imageborded2=fillImage(image1,7,1)
	# imagesmoothed2=my_imGaussConvol(image1,7,1,True)
	# imageborded3=fillImage(image1,7,2)
	# imagesmoothed3=my_imGaussConvol(image1,7,2,True)
	# paintMatrixImages(
	# 	[[imageborded1,imagesmoothed1],[imageborded2,imagesmoothed3],[imageborded3,imagesmoothed2]],
	# 	[["UNIFORM 0","SMOOTH_U0"],["REFLECT","SMOOTH_UC"],["UNIFORM COPY","SMOOTH_REF"] ],
	#  	"Practica 1 - Vision por computador - Jose Carlos Martinez Velazquez"
	# )

#PRUEBA 1c: MOSTRAR IMAGEN COMPLETAMENTE CONVOLUCIONADA
	# print("Aplicando convolucion completa")
	# image1=loadImage("imagenes/cat.bmp","COLOR")
	# imageborded1=fillImage(image1,7,0)
	# imagesmoothed1=my_imGaussConvol(image1,7,0,False)
	# imageborded2=fillImage(image1,7,1)
	# imagesmoothed2=my_imGaussConvol(image1,7,1,False)
	# imageborded3=fillImage(image1,7,2)
	# imagesmoothed3=my_imGaussConvol(image1,7,2,False)
	# paintMatrixImages(
	# 	[[imageborded1,imagesmoothed1],[imageborded2,imagesmoothed3],[imageborded3,imagesmoothed2]],
	# 	[["UNIFORM 0","SMOOTH_U0"],["REFLECT","SMOOTH_UC"],["UNIFORM COPY","SMOOTH_REF"] ],
	# 	"Practica 1 - Vision por computador - Jose Carlos Martinez Velazquez"
	# )


#PRUEBA 1c: MOSTRAR LA IMAGEN EN ALTAS FRECUENCIAS
	# print("A continuacion se mostrara el proceso de construccion de una imagen a altas frecuencias, espere un momento...")
	# imagen=loadImage("imagenes/motorcycle.bmp","GRAYSCALE")
	# smoothed=my_imGaussConvol(imagen,2,1)
	# imagenhf=getHighFrequences(imagen,smoothed,1)
	# paintMatrixImages(
	# 	[[imagen,smoothed,imagenhf]],
	# 	[["ORIGINAL","SMOOTHED","Hi-FREQUENCES"]],
	# 	"Practica 1 - Vision por computador - Jose Carlos Martinez Velazquez"
	# )
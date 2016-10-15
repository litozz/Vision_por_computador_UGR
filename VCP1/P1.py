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
	#r,g,b=cv2.split(image)
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

	senialBordeRV=np.transpose(senialBordeRV) 
	senialBordeGV=np.transpose(senialBordeGV)
	senialBordeBV=np.transpose(senialBordeBV) 	
 		
 	imagereflected=cv2.merge([negative(senialBordeRV),negative(senialBordeGV),negative(senialBordeBV)])
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

def normalize(shape,min,max):
	if(min<max):
		maxvalue=-256
		minvalue=256
		nshape=np.zeros( (len(shape),len(shape[0])) )
		for i in xrange(len(shape)):
			for j in xrange(len(shape[0])):
				if(shape[i][j]>maxvalue):
					maxvalue=shape[i][j]
				elif(shape[i][j]<minvalue):
					minvalue=shape[i][j]
		for i in xrange(len(shape)):
			for j in xrange(len(shape[0])):
				nshape[i][j] = ((max-min/(maxvalue-minvalue))*(shape[i][j]-minvalue))
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
El primer paso del apartado 2)consiste en hacer los calculos de convolucion en un vector. 
Coge un vector y una mascara y calcula el filtrado del vector
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
La funcion createAuxVector corresponde al segundo paso del apartado 2).
De lo que se trata es de copiar los bordes ya sea constante a
cero o con reflejo o copia del ultimo pixel.
"""

def createAuxVector(mask,vect,borderType):
	result=np.array([0 for i in xrange(len(vect)+(len(mask)-1))])
	startPosition=len(mask)/2
	finishPosition=len(result)-startPosition
	result[startPosition:finishPosition]=vect

	#if(borderType==0): #Borde a ceros

	if(borderType==1): #Borde reflejo
		result[0:startPosition]=result[2*startPosition:startPosition:-1]
		result[finishPosition:-1]=result[finishPosition-1:finishPosition-startPosition:-1]
		

	elif(borderType==2): #Borde copia
		result[0:startPosition]=result[startPosition]
		result[finishPosition:-1]=result[finishPosition-1]

	return result






"""
La funcion my_imGaussConvol realiza la convolucion de una imagen.
Los parametros que se usan son la imagen que se quiere convolucionar,
la descomposicion en dos valores de la mascara que se usa, la HORIZONTAL
y la VERTICAL y el tipo de borde: 0 -> borde negro, 1 -> borde reflejo,
2-> borde copia

"""

def my_imGaussConvol(image,sigma,bordertype,only_horizontal=False):
	mask=getMask(sigma)
	r,g,b=cv2.split(image)

	for i in xrange(len(r)):
 		r[i]=convolution2Vectors(mask,createAuxVector(mask,r[i],bordertype))
 		g[i]=convolution2Vectors(mask,createAuxVector(mask,g[i],bordertype))
 		b[i]=convolution2Vectors(mask,createAuxVector(mask,b[i],bordertype))

#Comentar a partir de aqui para ver la diferencia entre solo horizontal y vertical tambien
	if(not only_horizontal):
	 	r=np.transpose(r)
	 	g=np.transpose(g)
	 	b=np.transpose(b)

	   	for i in xrange(len(r)):
  			r[i]=convolution2Vectors(mask,createAuxVector(mask,r[i],bordertype))
  			g[i]=convolution2Vectors(mask,createAuxVector(mask,g[i],bordertype))
  			b[i]=convolution2Vectors(mask,createAuxVector(mask,b[i],bordertype))
	 
   		r=np.transpose(r)
   		g=np.transpose(g)
   		b=np.transpose(b)
#Fin de comentarios

 	return cv2.merge([r,g,b])


"""
La funcion getHighFrequences toma dos veces la misma imagen,
la original y la convolucionada. Resta la convolucionada a
la imagen original y obtenemos los detalles (altas frecuencias).
Se puede aplicar un filtro laplaciano, por lo que se pasa un coeficiente
laplaciano.
"""

def getHighFrequences(image,imageconv,hFfactor):
	r,g,b=cv2.split(image)
	rconv,gconv,bconv=cv2.split(imageconv)
	
	r=hFfactor*r - rconv
	g=hFfactor*g - gconv
	b=hFfactor*b - bconv

	for i in xrange(len(r)):
		for j in xrange(len(r[0])):
			if(r[i][j]<0):r[i][j]=0
			if(r[i][j]>255):r[i][j]=255
			if(g[i][j]<0):g[i][j]=0
			if(g[i][j]>255):g[i][j]=255
			if(b[i][j]<0):b[i][j]=0
			if(b[i][j]>255):b[i][j]=255

	return cv2.merge([r,g,b])





"""
La funcion getHybridImage construye una imagen hibrida
a partir de dos imagenes. El primer parametro debe ser
una imagen correspondiente a las altas frecuencias 
(extraidas previamente) y el segundo, una imagen alisada
convenientemente.
"""
def getHybridImage(imageHF,imageLF):

	rHF,gHF,bHF=cv2.split(imageHF)
	rLF,gLF,bLF=cv2.split(imageLF)

	rend=(rLF+rHF)/2
	gend=(gLF+gHF)/2
	bend=(bLF+bHF)/2

	hybridImage=cv2.merge([rend,gend,bend])
	return hybridImage




"""
La funcion scaleDownImage se encarga de escalar a mas pequeña
una imagen 
"""
def scaleDownImage(image,sigma,borderType):
	img=my_imGaussConvol(image,sigma,borderType)
	r,g,b=cv2.split(img)

	nrow=len(r)/2
	ncol=len(r[0])/2

	
	for i in xrange(0,nrow):
		r=np.delete(r,i,0)
		g=np.delete(g,i,0)
		b=np.delete(b,i,0)

	for i in xrange(0,ncol):
		r=np.delete(r,i,1)
		g=np.delete(g,i,1)
		b=np.delete(b,i,1)

	return cv2.merge([r,g,b])




def showPyramid5(imagen,windowtitle,sigma,border):
	
	fig = plt.figure()
	fig.canvas.set_window_title(windowtitle)
	
	nrow=len(imagen)
	ncol=len(imagen[0])
	
	ax1=plt.subplot2grid((nrow,ncol+ncol/2), (0,0), rowspan=nrow,colspan=ncol)
	plt.xticks([]),plt.yticks([])
	plt.imshow(imagen)
	
	i=0
	rowini=i
	row_span=nrow/2**(i+1)
	col_span=nrow/2**(i+1)

	img1=scaleDownImage(imagen,sigma+i,border)	
	ax2 = plt.subplot2grid((nrow,ncol+ncol/2), (0,ncol), rowspan=row_span,colspan=col_span)
	plt.xticks([]),plt.yticks([])
	plt.imshow(imagen)
	
	i=1
	rowini+=row_span
	row_span=nrow/2**(i+1)
	col_span=nrow/2**(i+1)

	img2=scaleDownImage(img1,sigma,border)
	ax3 = plt.subplot2grid((nrow,ncol+ncol/2), (rowini, ncol), rowspan=row_span, colspan=col_span)
	plt.xticks([]),plt.yticks([])
	plt.imshow(img2)
	
	i=2
	rowini+=row_span
	row_span=nrow/2**(i+1)
	col_span=nrow/2**(i+1)

	img3=scaleDownImage(img2,sigma,border)
	ax4 = plt.subplot2grid((nrow,ncol+ncol/2), (rowini, ncol), rowspan=row_span, colspan=row_span)
	plt.xticks([]),plt.yticks([])
	plt.imshow(img3)

	i=3
	rowini+=row_span
	row_span=nrow/2**(i+1)
	col_span=nrow/2**(i+1)

	img4=scaleDownImage(img3,sigma,border)
	ax5 = plt.subplot2grid((nrow,ncol+ncol/2), (rowini, ncol), rowspan=row_span, colspan=col_span)
	plt.xticks([]),plt.yticks([])
	plt.imshow(img3)

	plt.show()






"""------------FUNCIONES PARA MOSTRAR LOS DIFERENTES APARTADOS------------"""
def showAllBorders(ruta,color,sigma):
	imagen=loadImage(ruta,color)
	im0=fillImage(imagen,sigma,0)
	im1=fillImage(imagen,sigma,1)
	im2=fillImage(imagen,sigma,2)

	print("Esta viendo los diferentes rellenos de imagen para poder filtrar, cierre la ventana para continuar.")

	paintMatrixImages(
		[[imagen,im0,im1,im2]],
		[["ORIGINAL","BLACK PADDING","REFLECT","UNIFORM COPY"]],
		"Practica 1 - Vision por computador - Jose Carlos Martinez Velazquez"
	)



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
		[[imagenAltas,imconv,imAltas],[imagenBajas,imBajas],[hybridimage]],
		[["ORIGINAL_HF","SMOOTHED","Hi-FREQUENCES"],["ORIGINAL_LF","Lo-FREQUENCES"],["HYBRID IMAGE"]],
		"Practica 1 - Vision por computador"
	)

	print("Esta viendo la imagen hibrida, cierre la ventana para continuar.")

	paintMatrixImages(
		[[hybridimage]],
		[["HYBRID IMAGE"]],
		"Practica 1 - Vision por computador -  - Jose Carlos Martinez Velazquez"
	)

	return hybridimage

"""-----------------------------------------------------------------------"""

























if __name__=='__main__':
	# os.system('cls' if os.name == 'nt' else 'clear')


#PRUEBA 0: MOSTRAR LOS DIFERENTES TIPOS DE RELLENO
	#print("A continuacion se mostraran los diferentes tipos de relleno para poder filtrar, espere un momento...")
	#showAllBorders("imagenes/motorcycle.bmp","COLOR",10)

#PRUEBA 1: MOSTRAR UNA IMAGEN CONVOLUCIONADA SOLO EN HORIZONTAL Y COMPLETAMENTE
	#print("A continuacion se mostrara el proceso de construccion del suavizado, espere un momento...")
	#showSmoothedImage("imagenes/cat.bmp","COLOR",7,0)
	#os.system('cls' if os.name == 'nt' else 'clear')

#PRUEBA 2 IMAGEN HIBRIDA EINSTEIN-MARILYN
	#showConstructionHybridImage(rutaAltas,colorAltas,sigmaAltas,rutaBajas,colorBajas,sigmaBajas,factorLaplaciano,border):
	#print("Construyendo la imagen hibrida Einstein-Marilyn, espere un momento...")
	#hybrid1=showConstructionHybridImage("imagenes/einstein.bmp","GRAYSCALE",1.5,
	#							"imagenes/marilyn.bmp","GRAYSCALE",3,
	#							1,1)
	#print("Construyendo la piramide gaussiana Einstein-Marilyn, espere un momento...")
	#showPyramid5(hybrid1,"Piramide Gaussiana - Vision por computador - Jose Carlos Martinez Velazquez",1,1)
	#os.system('cls' if os.name == 'nt' else 'clear')
	

#PRUEBA 3: IMAGEN HIBRIDA BICI-MOTO
	#showConstructionHybridImage(rutaAltas,colorAltas,sigmaAltas,rutaBajas,colorBajas,sigmaBajas,factorLaplaciano,border):
	#print("Construyendo la imagen hibrida Bici-Moto, espere un momento...")
	#hybrid2=showConstructionHybridImage("imagenes/bicycle.bmp","COLOR",1.3,
	#							"imagenes/motorcycle.bmp","COLOR",3,
	#							1,1)
	#print("Construyendo la piramide gaussiana Bici-Moto, espere un momento...")
	#showPyramid5(hybrid2,"Piramide Gaussiana - Vision por computador - Jose Carlos Martinez Velazquez",1,1)
	#os.system('cls' if os.name == 'nt' else 'clear')

#PRUEBA 4: IMAGEN HIBRIDA AVION-AVE
	#showConstructionHybridImage(rutaAltas,colorAltas,sigmaAltas,rutaBajas,colorBajas,sigmaBajas,factorLaplaciano,border):
	#print("Construyendo la imagen hibrida Avion-Ave, espere un momento...")
	#hybrid3=showConstructionHybridImage("imagenes/plane.bmp","COLOR",1.5, #Tambien funciona con 0.85
	#							"imagenes/bird.bmp","COLOR",3, #Tambien funciona con 2.2, 2.7 y 3.5
	#								1,1)
	#print("Construyendo la piramide gaussiana Avion-Ave, espere un momento...")
	#showPyramid5(hybrid3,"Piramide Gaussiana - Vision por computador - Jose Carlos Martinez Velazquez",1,1)
	#os.system('cls' if os.name == 'nt' else 'clear')

#PRUEBA 5: IMAGEN HIBRIDA GATO-PERRO
	#showConstructionHybridImage(rutaAltas,colorAltas,sigmaAltas,rutaBajas,colorBajas,sigmaBajas,factorLaplaciano,border):
	#print("Construyendo la imagen hibrida Gato-Perro, espere un momento...")
	#hybrid4=showConstructionHybridImage("imagenes/cat.bmp","COLOR",3,
	#							"imagenes/dog.bmp","COLOR",6,
	#							1,1)
	#print("Construyendo la piramide gaussiana Gato-Perro, espere un momento...")
	#showPyramid5(hybrid4,"Piramide Gaussiana - Vision por computador - Jose Carlos Martinez Velazquez",1,1)
	#os.system('cls' if os.name == 'nt' else 'clear')

#PRUEBA 6: IMAGEN HIBRIDA SUBMARINO-PEZ
	#showConstructionHybridImage(rutaAltas,colorAltas,sigmaAltas,rutaBajas,colorBajas,sigmaBajas,factorLaplaciano,border):
	#print("Construyendo la imagen hibrida Submarino-Pez, espere un momento...")
	#hybrid5=showConstructionHybridImage("imagenes/submarine.bmp","COLOR",3,
	#							"imagenes/fish.bmp","COLOR",6,
	#							1,1)
	#print("Construyendo la piramide gaussiana Submarino-Pez, espere un momento...")
	#showPyramid5(hybrid5,"Piramide Gaussiana - Vision por computador - Jose Carlos Martinez Velazquez",1,1)
	#os.system('cls' if os.name == 'nt' else 'clear')
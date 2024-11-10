import re
import math

def crear_matriz_linea_linea_archivo(archivo):
    matriz_file = []
    # Abre el archivo en modo lectura
    with open(archivo, 'r') as file:
        # Lee cada linea a linea del archivo
        for line in file:
            matriz_file.append(line.strip())
    return matriz_file

def imprimir_linea_matriz(matriz_file):
    for line in matriz_file:
        print(line)

def procesar_data(matriz_file):
    matriz_process = []
    for line in matriz_file:
        if line.find(",") != -1:
            datosline = line.split(",")
            datoslinecast = []
            for dato in datosline:
                if re.match("[0-9]+.[0-9]*",dato):
                    datoslinecast.append(float(dato))
                else:
                    datoslinecast.append(dato)
            matriz_process.append((datoslinecast))
    return matriz_process

def separar_caracteristicas_clases(matri_process):
    caracteristicas = [row[:-1] for row in matri_process]  
    clases = [row[-1] for row in matri_process]   
    return caracteristicas,clases

if __name__ == "__main__":
    matriz_file = crear_matriz_linea_linea_archivo('iris/bezdekIris.data')
    imprimir_linea_matriz(matriz_file)
    matriz_process = procesar_data(matriz_file)
    imprimir_linea_matriz(matriz_process)
    caracteristicas,clases = separar_caracteristicas_clases(matriz_process)
    imprimir_linea_matriz(caracteristicas)

